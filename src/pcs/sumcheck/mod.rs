use crate::{
    p3_field_prelude::*,
    pcs::{
        EqClaim, PowClaim,
        sumcheck::compress::{compress_claims, compress_claims_packed},
    },
    poly::{Point, Poly, eval_eq_xy, eval_pow_xy},
    transcript::{Challenge, Reader, Writer},
    utils::{VecOps, extrapolate, unpack, unpack_into},
};
use p3_util::log2_strict_usize;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

mod compress;

pub(crate) const PACK_THRESHOLD: usize = 4;

#[tracing::instrument(skip_all, fields(k = poly.k()))]
fn initial_round_packed<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
    poly: &Poly<F>,
    weights: &mut Poly<Ext::ExtensionPacking>,
) -> Result<Ext, crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    let poly = F::Packing::pack_slice(poly);
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip_eq(phi.par_iter())
        .zip_eq(elo.par_iter().zip_eq(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ext::ExtensionPacking::ZERO, Ext::ExtensionPacking::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    let v0 = Ext::ExtensionPacking::to_ext_iter([v0]).sum::<Ext>();
    let v2 = Ext::ExtensionPacking::to_ext_iter([v2]).sum::<Ext>();
    transcript.write(v0)?;
    transcript.write(v2)?;

    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&round_poly, r);
    weights.fix_var_mut(r);
    Ok(r)
}

#[tracing::instrument(skip_all, fields(k = poly.k()))]
fn initial_round<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
    poly: &Poly<F>,
    weights: &mut Poly<Ext>,
) -> Result<Ext, crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip_eq(phi.par_iter())
        .zip_eq(elo.par_iter().zip_eq(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ext::ZERO, Ext::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    transcript.write(v0)?;
    transcript.write(v2)?;

    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&round_poly, r);
    weights.fix_var_mut(r);
    Ok(r)
}

#[tracing::instrument(skip_all, fields(k = poly.k() + log2_strict_usize(F::Packing::WIDTH)))]
fn round_packed<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
    poly: &mut Poly<Ext::ExtensionPacking>,
    weights: &mut Poly<Ext::ExtensionPacking>,
) -> Result<Ext, crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip_eq(phi.par_iter())
        .zip_eq(elo.par_iter().zip_eq(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ext::ExtensionPacking::ZERO, Ext::ExtensionPacking::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    let v0 = Ext::ExtensionPacking::to_ext_iter([v0]).sum::<Ext>();
    let v2 = Ext::ExtensionPacking::to_ext_iter([v2]).sum::<Ext>();
    transcript.write(v0)?;
    transcript.write(v2)?;

    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&round_poly, r);

    poly.fix_var_mut(r);
    weights.fix_var_mut(r);
    Ok(r)
}

#[tracing::instrument(skip_all, fields(k = poly.k()))]
fn round<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
    poly: &mut Poly<Ext>,
    weights: &mut Poly<Ext>,
) -> Result<Ext, crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip_eq(phi.par_iter())
        .zip_eq(elo.par_iter().zip_eq(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ext::ZERO, Ext::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    transcript.write(v0)?;
    transcript.write(v2)?;

    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&round_poly, r);

    poly.fix_var_mut(r);
    weights.fix_var_mut(r);
    Ok(r)
}

#[derive(Debug, Clone)]
enum MaybePacked<F: Field, Ext: ExtensionField<F>> {
    Packed {
        poly: Poly<Ext::ExtensionPacking>,
        weights: Poly<Ext::ExtensionPacking>,
    },
    Small {
        poly: Poly<Ext>,
        weights: Poly<Ext>,
    },
}

impl<F: Field, Ext: ExtensionField<F>> MaybePacked<F, Ext> {
    fn new_packed(poly: Poly<Ext::ExtensionPacking>, weights: Poly<Ext::ExtensionPacking>) -> Self {
        // TODO: assert sizes
        Self::Packed { poly, weights }
    }

    fn new_small(poly: Poly<Ext>, weights: Poly<Ext>) -> Self {
        // TODO: assert sizes
        Self::Small { poly, weights }
    }

    fn k(&self) -> usize {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        match self {
            MaybePacked::Packed { poly, weights } => {
                let k = poly.k();
                assert_eq!(k, weights.k());
                // assert!(k >= PACK_THRESHOLD);
                poly.k() + k_pack
            }
            MaybePacked::Small { poly, weights } => {
                let k = poly.k();
                assert_eq!(k, weights.k());
                // assert!(k < PACK_THRESHOLD);
                poly.k()
            }
        }
    }

    fn eval(&self, point: &Point<Ext>) -> Ext {
        match self {
            MaybePacked::Packed { poly, .. } => poly.eval_packed(point),
            MaybePacked::Small { poly, .. } => poly.eval(point),
        }
    }

    #[cfg(debug_assertions)]
    fn eval_univariate(&self, var: Ext) -> Ext {
        match self {
            MaybePacked::Packed { poly, .. } => poly.eval_univariate_packed(var),
            MaybePacked::Small { poly, .. } => poly.eval_univariate(var),
        }
    }

    #[cfg(debug_assertions)]
    fn prod(&self) -> Ext {
        match self {
            MaybePacked::Packed { poly, weights } => {
                let sum_packed = dot_product(poly.iter().cloned(), weights.iter().cloned());
                Ext::ExtensionPacking::to_ext_iter([sum_packed]).sum::<Ext>()
            }
            MaybePacked::Small { poly, weights } => {
                dot_product(poly.iter().cloned(), weights.iter().cloned())
            }
        }
    }

    fn transition(&mut self) {
        let k = self.k();
        match self {
            MaybePacked::Packed { poly, weights } if k < PACK_THRESHOLD => {
                let poly = unpack::<F, Ext>(poly).into();
                let weights = unpack::<F, Ext>(weights).into();
                *self = MaybePacked::Small { poly, weights };
            }
            _ => {}
        }
    }

    fn round<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        sum: &mut Ext,
    ) -> Result<Ext, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let r = match self {
            MaybePacked::Packed { poly, weights } => {
                let r = round_packed::<_, F, Ext>(transcript, sum, poly, weights)?;
                Ok(r)
            }
            MaybePacked::Small { poly, weights } => {
                let r = round::<_, F, Ext>(transcript, sum, poly, weights)?;
                Ok(r)
            }
        }?;

        #[cfg(debug_assertions)]
        assert_eq!(*sum, self.prod());

        self.transition();
        Ok(r)
    }

    #[tracing::instrument(skip_all)]
    fn unpack_into(&self, unpacked: &mut [Ext]) {
        match self {
            MaybePacked::Packed { poly, .. } => {
                assert_eq!(
                    unpacked.len(),
                    1 << (poly.k() + log2_strict_usize(F::Packing::WIDTH))
                );
                unpack_into::<F, Ext>(unpacked, poly);
            }
            MaybePacked::Small { poly, .. } => {
                unpacked.copy_from_slice(poly);
            }
        }
    }

    fn compress_claims(
        &mut self,
        sum: &mut Ext,
        alpha: Ext,
        eq_claims: &[EqClaim<Ext, Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) {
        match self {
            MaybePacked::Packed { weights, .. } => {
                compress_claims_packed::<F, Ext>(weights, sum, alpha, eq_claims, pow_claims);
            }
            MaybePacked::Small { weights, .. } => {
                compress_claims::<F, Ext>(weights, sum, alpha, eq_claims, pow_claims);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sumcheck<F: Field, Ext: ExtensionField<F>> {
    sum: Ext,
    rs: Point<Ext>,
    polys: MaybePacked<F, Ext>,
    poly_unpacked: Poly<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> Sumcheck<F, Ext> {
    pub fn k(&self) -> usize {
        self.polys.k()
    }

    pub fn poly_unpacked(&self) -> &Poly<Ext> {
        &self.poly_unpacked
    }

    pub fn eval(&self, point: &Point<Ext>) -> Ext {
        self.polys.eval(point)
    }

    #[cfg(debug_assertions)]
    pub(crate) fn eval_univariate(&self, var: Ext) -> Ext {
        self.polys.eval_univariate(var)
    }

    pub fn rs(&self) -> Point<Ext> {
        self.rs.clone()
    }

    pub fn sum(&self) -> Ext {
        self.sum
    }

    #[tracing::instrument(skip_all, fields(k = poly.k(), d = d, eqs = eq_claims.len()))]
    pub fn new<Transcript>(
        transcript: &mut Transcript,
        d: usize,
        eq_claims: &[EqClaim<Ext, Ext>],
        poly: &Poly<F>,
    ) -> Result<Self, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = poly.k();
        assert!(
            d > 0,
            "must be initialize with at least one round for base to ext transition"
        );
        assert!(
            d <= k,
            "number of rounds must be less than or equal to instance size"
        );

        let alpha = transcript.draw();
        let mut sum = Ext::ZERO;

        let (mut polys, r) = if k > PACK_THRESHOLD {
            let mut weights: Poly<_> = Ext::ExtensionPacking::zero_vec(1 << (k - k_pack)).into();
            compress_claims_packed(&mut weights, &mut sum, alpha, eq_claims, &[]);
            let r = initial_round_packed(transcript, &mut sum, poly, &mut weights)?;
            let poly = poly.fix_var_packed(r);
            (MaybePacked::new_packed(poly, weights), r)
        } else {
            let mut weights: Poly<_> = Ext::zero_vec(1 << k).into();
            compress_claims(&mut weights, &mut sum, alpha, eq_claims, &[]);
            let r = initial_round(transcript, &mut sum, poly, &mut weights)?;
            let poly = poly.fix_var(r);
            (MaybePacked::new_small(poly, weights), r)
        };
        #[cfg(debug_assertions)]
        assert_eq!(polys.prod(), sum);

        let rs = std::iter::once(Ok(r))
            .chain((1..d).map(|_| polys.round(transcript, &mut sum)))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        let mut poly_unpacked = Poly::<Ext>::zero(polys.k());
        polys.unpack_into(&mut poly_unpacked);

        Ok(Self {
            sum,
            rs,
            polys,
            poly_unpacked,
            alpha,
        })
    }

    #[tracing::instrument(skip_all, fields(k = self.k(), d = d, eqs = eq_claims.len(), pows = pow_claims.len()))]
    #[allow(clippy::too_many_arguments)]
    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        eq_claims: &[EqClaim<Ext, Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        self.alpha = transcript.draw();
        self.polys
            .compress_claims(&mut self.sum, self.alpha, eq_claims, pow_claims);
        let rs = (0..d)
            .map(|_| self.polys.round(transcript, &mut self.sum))
            .collect::<Result<Vec<_>, _>>()?;

        let k = self.k();
        self.poly_unpacked.truncate(1 << k);
        self.polys.unpack_into(&mut self.poly_unpacked);

        self.rs.extend(rs.iter());
        Ok(rs.into())
    }
}

#[derive(Debug, Clone)]
struct MultiRound<F: Field, Ext: ExtensionField<F>> {
    eq_points: Vec<Point<Ext>>,
    pow_points: Vec<Point<F>>,
    rs: Point<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> MultiRound<F, Ext> {
    fn new(
        eq_points: Vec<Point<Ext>>,
        pow_points: Vec<Point<F>>,
        rs: Point<Ext>,
        alpha: Ext,
    ) -> Self {
        Self {
            eq_points,
            pow_points,
            rs,
            alpha,
        }
    }

    fn extend(&mut self, rs: &Point<Ext>) {
        self.rs.extend(rs.iter());
    }

    fn weights(&self, poly: &Poly<Ext>) -> Ext {
        let rs = &self.rs.reversed();
        let weights = self
            .eq_points
            .iter()
            .map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);
                eval_eq_xy(&zs1, rs) * poly.eval(&zs0.as_ext::<Ext>())
            })
            .chain(self.pow_points.iter().map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);
                if off == 0 {
                    eval_pow_xy(&zs1, rs) * poly.constant().unwrap()
                } else {
                    eval_pow_xy(&zs1, rs) * poly.eval_univariate(Ext::from(*zs0.first().unwrap()))
                }
            }))
            .collect::<Vec<Ext>>();
        weights.iter().horner(self.alpha)
    }
}

fn reduce<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
) -> Result<Ext, crate::Error>
where
    Transcript: Reader<Ext> + Challenge<F, Ext>,
{
    let v0: Ext = transcript.read()?;
    let v2 = transcript.read()?;
    let sumcheck_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&sumcheck_poly, r);
    Ok(r)
}

pub struct SumcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    pub k: usize,
    sum: Ext,
    multi_rounds: Vec<MultiRound<F, Ext>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            sum: Ext::ZERO,
            multi_rounds: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        eq_claims: &[EqClaim<Ext, Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        eq_claims.iter().for_each(|o| assert_eq!(o.k(), self.k));
        pow_claims.iter().for_each(|o| assert_eq!(o.k, self.k));

        let alpha: Ext = transcript.draw();
        self.sum += eq_claims
            .iter()
            .map(EqClaim::<Ext, Ext>::eval)
            .chain(pow_claims.iter().map(PowClaim::<F, Ext>::eval))
            .horner(alpha);

        let round_rs: Point<Ext> = (0..d)
            .map(|_| reduce::<_, F, Ext>(transcript, &mut self.sum))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        self.multi_rounds
            .iter_mut()
            .for_each(|round| round.extend(&round_rs));

        let eq_points = eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>();
        let pow_points = pow_claims.iter().map(PowClaim::point).collect::<Vec<_>>();
        self.multi_rounds.push(MultiRound::new(
            eq_points,
            pow_points,
            round_rs.clone(),
            alpha,
        ));

        self.k -= d;
        Ok(round_rs)
    }

    pub fn finalize<Transcript>(
        self,
        transcript: &mut Transcript,
        poly: Option<Poly<Ext>>,
    ) -> Result<(), crate::Error>
    where
        Transcript: Reader<Ext>,
    {
        let poly = poly.map_or_else(|| Ok(Poly::new(transcript.read_many(1 << self.k)?)), Ok)?;
        let weigths = self
            .multi_rounds
            .iter()
            .map(|round| round.weights(&poly))
            .sum::<Ext>();
        (self.sum == weigths)
            .then_some(())
            .ok_or(crate::Error::Verify)
    }
}

#[cfg(test)]
mod test {
    use crate::p3_field_prelude::*;

    use crate::pcs::test::{make_eq_claims, make_eq_claims_base, make_pow_claims};
    use crate::transcript::test_transcript::{TestReader, TestWriter};
    use crate::utils::{VecOps, unpack};
    use crate::{
        pcs::{
            EqClaim, PowClaim,
            sumcheck::{MultiRound, Sumcheck, compress_claims, compress_claims_packed},
            test::{get_eq_claims, get_pow_claims},
        },
        poly::{Point, Poly},
        transcript::{Challenge, Writer},
    };
    use p3_util::log2_strict_usize;
    use rand::Rng;

    type F = p3_koala_bear::KoalaBear;
    type PackedF = <F as Field>::Packing;
    type Ext = BinomialExtensionField<F, 4>;
    type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

    fn compress_naive<F: Field, Ext: ExtensionField<F>>(
        k: usize,
        alpha: Ext,
        points: &[Point<Ext>],
        vars: &[F],
    ) -> Vec<Ext> {
        let eqs = points
            .iter()
            .map(|point| point.eq(Ext::ONE))
            .collect::<Vec<_>>();

        let pows = vars
            .iter()
            .map(|&x| x.powers().take(1 << k).collect())
            .collect::<Vec<_>>();

        let mut acc = Ext::zero_vec(1 << k);
        acc.iter_mut()
            .enumerate()
            .for_each(|(i, acc)| *acc += eqs.iter().map(|eq| &eq[i]).horner(alpha));

        let shift = alpha.exp_u64(points.len() as u64);
        acc.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += pows.iter().map(|pow| &pow[i]).horner_shifted(alpha, shift)
        });

        acc
    }

    #[test]
    fn test_compress_packed() {
        let mut rng = crate::test::rng(1);
        let alpha: Ext = rng.random();

        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        let k_pack = log2_strict_usize(PackedF::WIDTH);
        for k in 4..10 {
            let poly = Poly::<Ext>::rand(&mut rng, k).pack::<F>();
            for _ in 0..100 {
                let n_eqs = rng.random_range(0..3);
                let n_pows = rng.random_range(0..3);
                let unpacked = poly.unpack::<F, Ext>();
                let eq_claims =
                    make_eq_claims::<_, F, Ext>(&mut transcript, n_eqs, &unpacked).unwrap();
                let pow_claims =
                    make_pow_claims::<_, F, Ext>(&mut transcript, n_pows, &unpacked).unwrap();

                let points = eq_claims
                    .iter()
                    .map(|c| c.point().clone())
                    .collect::<Vec<_>>();
                let vars = pow_claims.iter().map(|c| c.var()).collect::<Vec<_>>();
                let acc0 = compress_naive(k, alpha, &points, &vars);

                {
                    let mut acc1 = Ext::zero_vec(1 << k);
                    let mut sum = Ext::ZERO;
                    compress_claims::<F, Ext>(&mut acc1, &mut sum, alpha, &eq_claims, &pow_claims);
                    assert_eq!(acc0, acc1);
                }

                {
                    let mut acc1 = PackedExt::zero_vec(1 << (k - k_pack));
                    let mut sum = Ext::ZERO;
                    compress_claims_packed::<F, Ext>(
                        &mut acc1,
                        &mut sum,
                        alpha,
                        &eq_claims,
                        &pow_claims,
                    );
                    assert_eq!(acc0, unpack::<F, Ext>(&acc1));
                }
            }
        }
    }

    #[test]
    fn test_sumcheck() {
        {
            for k in 1..8 {
                let mut rng = crate::test::rng(1);
                let poly = Poly::<F>::rand(&mut rng, k);
                for d in 1..=k {
                    let n_eqs = 10;
                    let (proof, checkpoint_prover) = {
                        let mut transcript = TestWriter::<Vec<u8>, F>::init();
                        let eq_claims =
                            make_eq_claims_base::<_, F, Ext>(&mut transcript, n_eqs, &poly)
                                .unwrap();
                        let sc =
                            Sumcheck::<F, Ext>::new(&mut transcript, d, &eq_claims, &poly).unwrap();

                        assert_eq!(sc.k(), k - d);
                        {
                            let round = MultiRound::<F, Ext>::new(
                                eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>(),
                                vec![],
                                sc.rs(),
                                sc.alpha,
                            );
                            assert_eq!(round.weights(sc.poly_unpacked()), sc.sum);
                        }

                        transcript.write_many(sc.poly_unpacked()).unwrap();
                        let checkpoint: F = transcript.draw();
                        let proof = transcript.finalize();
                        (proof, checkpoint)
                    };

                    let checkpoint_verifier = {
                        let mut transcript = TestReader::<&[u8], F>::init(&proof);
                        let eq_claims_ext =
                            get_eq_claims::<_, F, Ext>(&mut transcript, k, n_eqs).unwrap();
                        let mut verifier = super::SumcheckVerifier::<F, Ext>::new(k);
                        verifier
                            .fold(&mut transcript, d, &eq_claims_ext, &[])
                            .unwrap();
                        verifier.finalize(&mut transcript, None).unwrap();
                        transcript.draw()
                    };
                    assert_eq!(checkpoint_prover, checkpoint_verifier);
                }
            }
        }

        {
            let mut rng = crate::test::rng(1);
            for n_fold in 1..5 {
                for _ in 0..100 {
                    let n_eqs = (0..n_fold)
                        .map(|_| rng.random_range(0..4))
                        .collect::<Vec<usize>>();

                    let n_pows = (0..n_fold)
                        .map(|i| if i == 0 { 0 } else { rng.random_range(0..4) })
                        .collect::<Vec<usize>>();

                    let ds = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();

                    let k = ds.iter().sum::<usize>();
                    let poly = Poly::<F>::rand(&mut rng, k);
                    let (proof, checkpoint_prover) = {
                        let mut transcript = TestWriter::<Vec<u8>, F>::init();
                        let mut k_folding = k;
                        let mut round = 0;

                        let (mut sc, mut rounds) = {
                            let n_eqs = n_eqs[0];
                            let n_pows = n_pows[0];
                            assert_eq!(n_pows, 0);
                            let d = ds[0];

                            let eq_claims =
                                make_eq_claims_base(&mut transcript, n_eqs, &poly).unwrap();

                            k_folding -= d;
                            round += d;

                            let sc = Sumcheck::new(&mut transcript, d, &eq_claims, &poly).unwrap();

                            assert_eq!(sc.k(), k_folding);
                            assert_eq!(k - sc.k(), round);

                            let eq_points =
                                eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>();
                            let round = super::MultiRound::<F, Ext>::new(
                                eq_points,
                                vec![],
                                sc.rs.clone(),
                                sc.alpha,
                            );

                            assert_eq!(round.weights(sc.poly_unpacked()), sc.sum);
                            (sc, vec![round])
                        };

                        n_eqs
                            .iter()
                            .zip(n_pows.iter())
                            .zip(ds.iter())
                            .skip(1)
                            .for_each(|((&n_eq_ext, &n_pow_base), &d)| {
                                k_folding -= d;
                                round += d;

                                let eq_claims = make_eq_claims::<_, F, Ext>(
                                    &mut transcript,
                                    n_eq_ext,
                                    sc.poly_unpacked(),
                                )
                                .unwrap();

                                let pow_claims = make_pow_claims::<_, F, Ext>(
                                    &mut transcript,
                                    n_pow_base,
                                    sc.poly_unpacked(),
                                )
                                .unwrap();

                                let rs = sc
                                    .fold(&mut transcript, d, &eq_claims, &pow_claims)
                                    .unwrap();

                                assert_eq!(sc.k(), k_folding);
                                assert_eq!(k - sc.k(), round);

                                {
                                    rounds.iter_mut().for_each(|round| round.extend(&rs));

                                    let eq_points =
                                        eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>();

                                    let pow_points =
                                        pow_claims.iter().map(PowClaim::point).collect::<Vec<_>>();

                                    rounds.push(super::MultiRound::<F, Ext>::new(
                                        eq_points, pow_points, rs, sc.alpha,
                                    ));
                                }

                                assert_eq!(
                                    rounds
                                        .iter()
                                        .map(|round| round.weights(sc.poly_unpacked()))
                                        .sum::<Ext>(),
                                    sc.sum
                                );
                            });

                        assert_eq!(round, k);
                        assert_eq!(k_folding, 0);

                        transcript.write_many(sc.poly_unpacked()).unwrap();
                        let checkpoint: F = transcript.draw();
                        let proof = transcript.finalize();
                        (proof, checkpoint)
                    };

                    let checkpoint_verifier = {
                        let mut transcript = TestReader::<&[u8], F>::init(&proof);
                        let mut verifier = super::SumcheckVerifier::<F, Ext>::new(k);

                        n_eqs.iter().zip(n_pows.iter()).zip(ds.iter()).for_each(
                            |((&n_eq_ext, &n_pow_base), &d)| {
                                let eq_claims: Vec<EqClaim<Ext, Ext>> = get_eq_claims::<_, F, Ext>(
                                    &mut transcript,
                                    verifier.k,
                                    n_eq_ext,
                                )
                                .unwrap();

                                let pow_claims: Vec<PowClaim<F, Ext>> =
                                    get_pow_claims::<_, F, Ext>(
                                        &mut transcript,
                                        verifier.k,
                                        n_pow_base,
                                    )
                                    .unwrap();

                                verifier
                                    .fold(&mut transcript, d, &eq_claims, &pow_claims)
                                    .unwrap();
                            },
                        );

                        verifier.finalize(&mut transcript, None).unwrap();
                        let checkpoint: F = transcript.draw();
                        checkpoint
                    };
                    assert_eq!(checkpoint_prover, checkpoint_verifier);
                }
            }
        }
    }
}
