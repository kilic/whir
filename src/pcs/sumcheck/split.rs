use crate::p3_field_prelude::*;
use crate::pcs::sumcheck::svo::{SvoAccumulators, calculate_accumulators};
use crate::poly::{Point, Poly, eval_eq_xy, eval_poly_reference};
use crate::transcript::{Challenge, Writer};
use crate::utils::TwoAdicSlice;
use itertools::Itertools;
use p3_util::log2_strict_usize;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

pub fn log3_strict_usize(n: usize) -> usize {
    assert_ne!(n, 0);
    let mut res = 0usize;
    let mut t = n;
    loop {
        t /= 3;
        if t == 0 {
            break;
        }
        res += 1;
    }
    assert_eq!(3usize.pow(res as u32), n,);
    res
}

#[derive(Debug, Clone)]
pub(crate) enum SplitPoint<F: Field, Ext: ExtensionField<F>> {
    DoSplit {
        suffix: Point<Ext>,
        eq0: Poly<Ext::ExtensionPacking>,
        eq1: Poly<Ext>,
        #[cfg(test)]
        original: Point<Ext>,
    },
    Fallback {
        point: Point<Ext>,
    },
}

impl<F: Field, Ext: ExtensionField<F>> SplitPoint<F, Ext> {
    pub(crate) fn can_pack(k: usize, l0: usize) -> Option<usize> {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        k.checked_sub(2 * k_pack).map(|l0_max| l0.min(l0_max))
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn new(point: Point<Ext>, l0: Option<usize>) -> Self {
        let k = point.k();
        if let Some(l0) = l0 {
            if let Some(l0_adjusted) = Self::can_pack(k, l0) {
                let (z_split, suffix) = point.split_at(k - l0_adjusted);
                let (z0, z1) = z_split.split_at((k - l0_adjusted) / 2);
                let eq0 = z0.eq_packed(Ext::ONE);
                let eq1 = z1.eq(Ext::ONE);
                Self::DoSplit {
                    suffix,
                    eq0,
                    eq1,
                    #[cfg(test)]
                    original: point,
                }
            } else {
                Self::Fallback { point }
            }
        } else {
            Self::Fallback { point }
        }
    }

    pub(crate) fn k(&self) -> usize {
        self.k_split() + self.l0().unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn original(&self) -> Point<Ext> {
        match self {
            SplitPoint::DoSplit { original, .. } => original.clone(),
            SplitPoint::Fallback { point } => point.clone(),
        }
    }

    pub(crate) fn l0(&self) -> Option<usize> {
        match self {
            SplitPoint::DoSplit { suffix, .. } => Some(suffix.k()),
            _ => None,
        }
    }

    pub(crate) fn k_split(&self) -> usize {
        match self {
            SplitPoint::DoSplit { eq0, eq1, .. } => {
                eq0.k() + eq1.k() + log2_strict_usize(F::Packing::WIDTH)
            }
            SplitPoint::Fallback { point } => point.k(),
        }
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn eval(&self, poly: &Poly<F>) -> (SvoAccumulators<Ext>, Ext) {
        match self {
            SplitPoint::DoSplit {
                suffix, eq0, eq1, ..
            } => {
                let chunk_size = 1 << self.k_split();
                let partial_evals = poly
                    .chunks(chunk_size)
                    .map(|poly| {
                        let poly = F::Packing::pack_slice(poly);
                        let sum = poly
                            .par_chunks(eq0.len())
                            .zip_eq(eq1.par_iter())
                            .map(|(poly, &eq1)| {
                                poly.iter()
                                    .zip_eq(eq0.iter())
                                    .map(|(&f, &eq0)| eq0 * f)
                                    .sum::<Ext::ExtensionPacking>()
                                    * eq1
                            })
                            .sum::<Ext::ExtensionPacking>();
                        Ext::ExtensionPacking::to_ext_iter([sum]).sum::<Ext>()
                    })
                    .collect::<Vec<_>>();
                let eval = eval_poly_reference(&partial_evals, suffix);
                let accumulators = calculate_accumulators(&partial_evals, suffix);
                (accumulators, eval)
            }
            SplitPoint::Fallback { point } => {
                let eval = poly.eval(point);
                (vec![], eval)
            }
        }
    }

    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len())))]
    pub(crate) fn combine_into_packed(
        &self,
        out: &mut [Ext::ExtensionPacking],
        alpha: Ext,
        rs: &Point<Ext>,
    ) {
        let k = self.k();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);

        let l0 = rs.len();
        assert!(k >= l0 + k_pack);
        assert_eq!(out.k(), k - l0 - k_pack);
        assert_eq!(out.k() + k_pack, self.k_split());

        match self {
            SplitPoint::DoSplit {
                suffix, eq0, eq1, ..
            } => {
                assert_eq!(l0, self.l0().unwrap());
                let scale = alpha * eval_eq_xy(suffix, &rs.reversed());
                out.par_chunks_mut(eq0.len())
                    .zip(eq1.par_iter())
                    .for_each(|(chunk, &eq1)| {
                        chunk
                            .iter_mut()
                            .zip(eq0.iter())
                            .for_each(|(out, &eq0)| *out += eq0 * eq1 * scale);
                    });
            }
            _ => panic!("must be split"),
        }
    }

    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len())))]
    pub(crate) fn combine_into(&self, out: &mut [Ext], alpha: Ext, rs: &Point<Ext>) {
        let k = self.k();
        let l0 = rs.len();
        assert!(k >= l0);
        assert_eq!(out.k(), k - l0);

        match self {
            SplitPoint::Fallback { point } => {
                let eq = point.eq(alpha).partial_eval(&rs.reversed());
                out.iter_mut()
                    .zip(eq.iter())
                    .for_each(|(out, &eq)| *out += eq);
            }
            _ => panic!("must be not split"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SplitEqClaim<F: Field, Ext: ExtensionField<F>> {
    point: SplitPoint<F, Ext>,
    pub(crate) accumulators: Vec<[Vec<Ext>; 2]>,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> SplitEqClaim<F, Ext> {
    #[tracing::instrument(skip_all)]
    pub(crate) fn new(point: Point<Ext>, l0: Option<usize>, poly: &Poly<F>) -> Self {
        assert_eq!(point.k(), poly.k());
        let point = SplitPoint::new(point, l0);
        assert_eq!(point.k(), poly.k());
        let (accumulators, eval) = point.eval(poly);
        Self {
            point,
            accumulators,
            eval,
        }
    }

    #[cfg(test)]
    pub(crate) fn normalize(self) -> crate::pcs::sumcheck::EqClaim<Ext, Ext> {
        crate::pcs::sumcheck::EqClaim::new(self.point.original(), self.eval)
    }

    pub(crate) fn l0(&self) -> Option<usize> {
        self.point.l0()
    }

    pub(crate) fn eval(&self) -> &Ext {
        &self.eval
    }

    pub(crate) fn accumulator(&self) -> &Vec<[Vec<Ext>; 2]> {
        &self.accumulators
    }

    pub(crate) fn k(&self) -> usize {
        self.point.k()
    }

    #[tracing::instrument(skip_all, fields(k = log2_strict_usize(out.len())))]
    pub(crate) fn combine_into_packed(
        &self,
        out: &mut [Ext::ExtensionPacking],

        alpha: Ext,
        rs: &Point<Ext>,
    ) {
        self.point.combine_into_packed(out, alpha, rs);
    }

    pub(crate) fn combine_into(&self, out: &mut [Ext], alpha: Ext, rs: &Point<Ext>) {
        self.point.combine_into(out, alpha, rs);
    }
}

#[derive(Clone, Debug)]
pub struct PolyEvaluator<F: Field, Ext: ExtensionField<F>> {
    pub(crate) poly: Poly<F>,
    pub(crate) claims: Vec<SplitEqClaim<F, Ext>>,
    pub(crate) l0: Option<usize>,
}

impl<F: Field, Ext: ExtensionField<F>> PolyEvaluator<F, Ext> {
    pub fn new(poly: Poly<F>, l0: usize) -> Self {
        let adjusted_l0 = SplitPoint::<F, Ext>::can_pack(poly.k(), l0);
        Self {
            poly,
            claims: vec![],
            l0: adjusted_l0,
        }
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn k(&self) -> usize {
        self.poly.k()
    }

    pub fn eval(&mut self, point: Point<Ext>) -> Ext {
        let claim = SplitEqClaim::new(point, self.l0, &self.poly);
        assert_eq!(claim.k(), self.poly.k());
        assert_eq!(claim.l0(), self.l0);
        let eval = claim.eval;
        self.claims.push(claim);
        eval
    }

    #[tracing::instrument(skip_all)]
    pub fn add_claim<Transcript>(&mut self, transcript: &mut Transcript) -> Result<(), crate::Error>
    where
        Transcript: Writer<Ext> + Challenge<F, Ext>,
    {
        let point = transcript.draw();
        let point = Point::expand(self.k(), point);
        let eval = self.eval(point);
        transcript.write(eval)
    }

    #[cfg(test)]
    pub fn normalize(&self) -> Vec<crate::pcs::sumcheck::EqClaim<Ext, Ext>> {
        self.claims
            .iter()
            .cloned()
            .map(SplitEqClaim::normalize)
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use p3_koala_bear::KoalaBear;
    use rand::Rng;

    #[test]
    fn test_split_points() {
        type F = KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type PackedF = <F as Field>::Packing;
        type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;
        let mut rng = crate::test::rng(1);

        let alpha: Ext = rng.random();
        let k_pack = log2_strict_usize(PackedF::WIDTH);
        for k in 1..12 {
            let poly = Poly::<F>::rand(&mut rng, k);
            for l0 in 0..=k {
                let point = Point::<Ext>::rand(&mut rng, k);
                let e0 = poly.eval(&point);

                let split = SplitPoint::<F, Ext>::new(point.clone(), Some(l0));
                let fallback = SplitPoint::<F, Ext>::new(point.clone(), None);

                assert_eq!(split.k(), k);
                assert_eq!(fallback.k(), k);

                if let Some(l0_adjusted) = SplitPoint::<F, Ext>::can_pack(k, l0) {
                    assert_eq!(split.l0().unwrap(), l0_adjusted);

                    let (accumulator, e1) = split.eval(&poly);
                    assert_eq!(e0, e1);
                    assert_eq!(accumulator.len(), l0_adjusted);
                    let (accumulator, e1) = fallback.eval(&poly);
                    assert_eq!(e0, e1);
                    assert!(accumulator.is_empty());

                    let rs = Point::rand(&mut rng, l0_adjusted);
                    let out0 = point.eq(alpha).partial_eval(&rs.reversed());

                    let mut out1 = Poly::<PackedExt>::zero(k - l0_adjusted - k_pack);
                    split.combine_into_packed(&mut out1, alpha, &rs);
                    assert_eq!(out0, out1.unpack::<F, Ext>());

                    let mut out1 = Poly::<Ext>::zero(k - l0_adjusted);
                    fallback.combine_into(&mut out1, alpha, &rs);
                    assert_eq!(out0, out1);
                } else {
                    assert_eq!(split.l0(), None);
                    let (accumulator, e1) = split.eval(&poly);
                    assert_eq!(e0, e1);
                    assert_eq!(accumulator.len(), 0);
                    let rs = Point::rand(&mut rng, l0);

                    let out0 = point.eq(alpha).partial_eval(&rs.reversed());
                    assert_eq!(out0.k(), k - l0);
                    let mut out1 = Poly::<Ext>::zero(k - l0);
                    split.combine_into(&mut out1, alpha, &rs);
                    assert_eq!(out0, out1);
                }
            }
        }
    }

    #[test]
    fn test_combine_split_points() {
        type F = KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type PackedF = <F as Field>::Packing;
        type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;
        let mut rng = crate::test::rng(1);
        let k = 12;
        let d = 2;
        let n = 10;
        let k_pack = log2_strict_usize(PackedF::WIDTH);

        let points = (0..n)
            .map(|_| Point::<Ext>::rand(&mut rng, k))
            .collect::<Vec<_>>();
        let alpha: Ext = rng.random();
        let rs = Point::rand(&mut rng, d);

        let splits = points
            .iter()
            .map(|point| SplitPoint::<F, Ext>::new(point.clone(), Some(d)))
            .collect::<Vec<_>>();
        let mut out0 = Poly::<PackedExt>::zero(k - d - k_pack);

        splits
            .iter()
            .zip(alpha.powers())
            .for_each(|(split, alpha)| split.combine_into_packed(&mut out0, alpha, &rs));

        let splits = points
            .iter()
            .map(|point| SplitPoint::<F, Ext>::new(point.clone(), None))
            .collect::<Vec<_>>();
        let mut out1 = Poly::<Ext>::zero(k - d);

        splits
            .iter()
            .zip(alpha.powers())
            .for_each(|(split, alpha)| split.combine_into(&mut out1, alpha, &rs));
        assert_eq!(out0.unpack(), out1);
    }
}
