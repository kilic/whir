use p3_util::log2_strict_usize;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    p3_field_prelude::*,
    pcs::sumcheck::{
        combine::{combine_claims, combine_claims_packed},
        extrapolate_012,
        split::{PolyEvaluator, SplitEqClaim},
        svo::lagrange_weights_012_multi,
        {EqClaim, PowClaim},
    },
    poly::{Point, Poly},
    transcript::{Challenge, Writer},
    utils::{TwoAdicSlice, VecOps, unpack, unpack_into},
};

#[tracing::instrument(skip_all, fields(k = poly.k()))]
pub fn coeffs<A, B>(poly: &[A], weights: &[B]) -> (B, B)
where
    A: Copy + Send + Sync + PrimeCharacteristicRing,
    B: Copy + Send + Sync + Algebra<A>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);
    let (c0, c2) = plo
        .par_iter()
        .zip_eq(phi.par_iter())
        .zip_eq(elo.par_iter().zip_eq(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (B::ZERO, B::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );
    (c0, c2)
}

#[derive(Debug, Clone)]
enum ProdPoly<F: Field, Ext: ExtensionField<F>> {
    Packed {
        poly: Poly<Ext::ExtensionPacking>,
        weights: Poly<Ext::ExtensionPacking>,
    },
    Small {
        poly: Poly<Ext>,
        weights: Poly<Ext>,
    },
}

impl<F: Field, Ext: ExtensionField<F>> ProdPoly<F, Ext> {
    fn new_packed(poly: Poly<Ext::ExtensionPacking>, weights: Poly<Ext::ExtensionPacking>) -> Self {
        let mut poly = Self::Packed { poly, weights };
        poly.transition();
        poly
    }

    fn new_small(poly: Poly<Ext>, weights: Poly<Ext>) -> Self {
        Self::Small { poly, weights }
    }

    fn k(&self) -> usize {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        match self {
            Self::Packed { poly, weights } => {
                let k = poly.k();
                assert_eq!(k, weights.k());
                poly.k() + k_pack
            }
            Self::Small { poly, weights } => {
                let k = poly.k();
                assert_eq!(k, weights.k());
                poly.k()
            }
        }
    }

    fn eval(&self, point: &Point<Ext>) -> Ext {
        match self {
            Self::Packed { poly, .. } => poly.eval_packed(point),
            Self::Small { poly, .. } => poly.eval(point),
        }
    }

    fn transition(&mut self) {
        if let Self::Packed { poly, weights } = self {
            let k = poly.k();
            assert_eq!(k, weights.k());
            if k == 0 {
                let poly = unpack::<F, Ext>(poly).into();
                let weights = unpack::<F, Ext>(weights).into();
                *self = Self::Small { poly, weights };
            }
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
        let (c0, c2) = match self {
            Self::Packed { poly, weights } => {
                let (c0, c2) = coeffs(poly, weights);
                let c0 = Ext::ExtensionPacking::to_ext_iter([c0]).sum::<Ext>();
                let c2 = Ext::ExtensionPacking::to_ext_iter([c2]).sum::<Ext>();
                (c0, c2)
            }
            Self::Small { poly, weights } => coeffs(poly, weights),
        };

        transcript.write_many(&[c0, c2])?;

        let r = Challenge::<F, Ext>::draw(transcript);
        self.fix_var(r);
        *sum = extrapolate_012(c0, *sum - c0, c2, r);

        debug_assert_eq!(*sum, self.prod());
        Ok(r)
    }

    #[tracing::instrument(skip_all)]
    fn unpack_into(&self, unpacked: &mut [Ext]) {
        match self {
            Self::Packed { poly, .. } => {
                assert_eq!(
                    unpacked.len(),
                    1 << (poly.k() + log2_strict_usize(F::Packing::WIDTH))
                );
                unpack_into::<F, Ext>(unpacked, poly);
            }
            Self::Small { poly, .. } => {
                unpacked.copy_from_slice(poly);
            }
        }
    }

    #[tracing::instrument(skip_all)]
    fn fix_var(&mut self, r: Ext) {
        match self {
            Self::Packed { poly, weights } => {
                poly.fix_var_mut(r);
                weights.fix_var_mut(r);
            }
            Self::Small { poly, weights } => {
                poly.fix_var_mut(r);
                weights.fix_var_mut(r);
            }
        }
        self.transition();
    }

    fn combine_claims(
        &mut self,
        sum: &mut Ext,
        alpha: Ext,
        eq_claims: &[EqClaim<Ext, Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) {
        match self {
            Self::Packed { weights, .. } => {
                combine_claims_packed::<F, Ext>(weights, sum, alpha, eq_claims, pow_claims);
            }
            Self::Small { weights, .. } => {
                combine_claims::<F, Ext>(weights, sum, alpha, eq_claims, pow_claims);
            }
        }
    }

    #[cfg(debug_assertions)]
    fn eval_univariate(&self, var: Ext) -> Ext {
        match self {
            Self::Packed { poly, .. } => poly.eval_univariate_packed(var),
            Self::Small { poly, .. } => poly.eval_univariate(var),
        }
    }

    fn prod(&self) -> Ext {
        match self {
            Self::Packed { poly, weights } => {
                let sum_packed = dot_product(poly.iter().cloned(), weights.iter().cloned());
                Ext::ExtensionPacking::to_ext_iter([sum_packed]).sum::<Ext>()
            }
            Self::Small { poly, weights } => {
                dot_product(poly.iter().cloned(), weights.iter().cloned())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sumcheck<F: Field, Ext: ExtensionField<F>> {
    sum: Ext,
    rs: Point<Ext>,
    polys: ProdPoly<F, Ext>,
    poly_unpacked: Poly<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> Sumcheck<F, Ext> {
    #[tracing::instrument(skip_all, fields(k = ev.k(), d = ev.l0, eqs = ev.len()))]
    pub fn new<Transcript>(
        transcript: &mut Transcript,
        ev: PolyEvaluator<F, Ext>,
        d: usize,
    ) -> Result<Self, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let alpha = transcript.draw();
        let mut sum = ev.claims.iter().map(SplitEqClaim::eval).horner(alpha);
        let mut rs = Point::default();
        assert!(d <= ev.k());

        if let Some(l0) = ev.l0 {
            assert!(l0 <= d);

            let accumulators = ev
                .claims
                .iter()
                .map(SplitEqClaim::accumulator)
                .collect::<Vec<_>>();

            for round_idx in 0..l0 {
                let (mut c0, mut c2): (Ext, Ext) = Default::default();
                let weights = lagrange_weights_012_multi(rs.as_slice());
                for (accumulators, alpha) in accumulators.iter().zip(alpha.powers()) {
                    let acc0 = &accumulators[round_idx][0];
                    let acc2 = &accumulators[round_idx][1];
                    c0 += alpha
                        * dot_product::<Ext, _, _>(acc0.iter().copied(), weights.iter().copied());
                    c2 += alpha
                        * dot_product::<Ext, _, _>(acc2.iter().copied(), weights.iter().copied());
                }
                transcript.write_many(&[c0, c2])?;
                let r: Ext = transcript.draw();
                sum = extrapolate_012(c0, sum - c0, c2, r);
                rs.push(r);
            }

            let poly = ev.poly.partial_eval_into_packed(&rs.reversed());
            let weights = tracing::info_span!("combine weights").in_scope(|| {
                let mut weights = Poly::<Ext::ExtensionPacking>::zero(poly.k());
                ev.claims
                    .iter()
                    .zip(alpha.powers())
                    .for_each(|(claim, alpha)| claim.combine_into_packed(&mut weights, alpha, &rs));
                weights
            });
            let mut polys = ProdPoly::new_packed(poly, weights);
            debug_assert_eq!(polys.prod(), sum);

            for _ in l0..d {
                rs.push(polys.round(transcript, &mut sum)?);
            }

            let mut poly_unpacked = Poly::<Ext>::zero(polys.k());
            polys.unpack_into(&mut poly_unpacked);

            Ok(Self {
                sum,
                rs,
                polys,
                poly_unpacked,
                alpha,
            })
        } else {
            let mut weights = Poly::<Ext>::zero(ev.poly.k());
            ev.claims
                .iter()
                .zip(alpha.powers())
                .for_each(|(claim, alpha)| claim.combine_into(&mut weights, alpha, &rs));
            let (c0, c2) = coeffs(&ev.poly, &weights);
            transcript.write_many(&[c0, c2])?;
            let r: Ext = transcript.draw();
            sum = extrapolate_012(c0, sum - c0, c2, r);
            let poly = ev.poly.fix_var(r);
            weights.fix_var_mut(r);
            let mut polys = ProdPoly::new_small(poly, weights);
            debug_assert_eq!(polys.prod(), sum);

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
    }

    pub fn k(&self) -> usize {
        self.polys.k()
    }

    pub fn poly_unpacked(&self) -> &Poly<Ext> {
        &self.poly_unpacked
    }

    pub fn eval(&self, point: &Point<Ext>) -> Ext {
        self.polys.eval(point)
    }

    pub fn alpha(&self) -> Ext {
        self.alpha
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
            .combine_claims(&mut self.sum, self.alpha, eq_claims, pow_claims);
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
