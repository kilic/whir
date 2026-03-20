use itertools::Itertools;
use p3_util::log2_strict_usize;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::sumcheck::{
    EqClaim, PowClaim,
    combine::{combine_claims, combine_claims_packed},
    extrapolate_012,
};
use common::{
    field::*,
    utils::{unpack, unpack_into},
};
use poly::Poly;
use transcript::{Challenge, Writer};

pub(crate) fn coeffs_lo_var<A, B>(poly: &[A], weights: &[B]) -> (B, B)
where
    A: Copy + Send + Sync + PrimeCharacteristicRing,
    B: Copy + Send + Sync + Algebra<A>,
{
    assert_eq!(poly.len(), weights.len());
    poly.par_chunks(2)
        .zip(weights.par_chunks(2))
        .map(|(f, w)| (w[0] * f[0], (w[1].double() - w[0]) * (f[1].double() - f[0])))
        .reduce(
            || (B::ZERO, B::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        )
}

pub(crate) fn coeffs_hi_var<A, B>(poly: &[A], weights: &[B]) -> (B, B)
where
    A: Copy + Send + Sync + PrimeCharacteristicRing,
    B: Copy + Send + Sync + Algebra<A>,
{
    let len = [poly.len(), weights.len()]
        .into_iter()
        .all_equal_value()
        .unwrap();

    let mid = len / 2;
    let (poly_lo, poly_hi) = poly.split_at(mid);
    let (weights_lo, weights_hi) = weights.split_at(mid);

    poly_lo
        .par_iter()
        .zip(poly_hi.par_iter())
        .zip(weights_lo.par_iter().zip(weights_hi.par_iter()))
        .map(|((&f0, &f1), (&w0, &w1))| (w0 * f0, (w1.double() - w0) * (f1.double() - f0)))
        .reduce(
            || (B::ZERO, B::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        )
}

#[derive(Debug, Clone)]
pub(crate) struct Expression<F: Field, Ext: ExtensionField<F>> {
    poly: Poly<Ext>,
    pub(crate) weights: Poly<Ext>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> Expression<F, Ext> {
    pub(crate) fn new(poly: Poly<Ext>, weights: Poly<Ext>) -> Self {
        Self {
            poly,
            weights,
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn k(&self) -> usize {
        let k = self.poly.k();
        assert_eq!(k, self.weights.k());
        k
    }

    pub(crate) fn round_lo_var<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        sum: &mut Ext,
    ) -> Result<Ext, common::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let (c0, c2) = coeffs_lo_var(&self.poly, &self.weights);

        transcript.write_many(&[c0, c2])?;

        let r = Challenge::<F, Ext>::draw(transcript);
        self.fix_lo_var(r);
        *sum = extrapolate_012(c0, *sum - c0, c2, r);

        debug_assert_eq!(*sum, self.prod());
        Ok(r)
    }

    // fn eval(&self, point: &Point<Ext>) -> Ext {
    //     self.poly.eval_ext(point)
    // }

    // pub(crate) fn round_hi_var<Transcript>(
    //     &mut self,
    //     transcript: &mut Transcript,
    //     sum: &mut Ext,
    // ) -> Result<Ext, common::Error>
    // where
    //     Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    // {
    //     let (c0, c2) = coeffs_hi_var(&self.poly, &self.weights);

    //     transcript.write_many(&[c0, c2])?;

    //     let r = Challenge::<F, Ext>::draw(transcript);
    //     self.fix_hi_var(r);
    //     *sum = extrapolate_012(c0, *sum - c0, c2, r);

    //     debug_assert_eq!(*sum, self.prod());
    //     Ok(r)
    // }

    pub(crate) fn poly(&self) -> &Poly<Ext> {
        &self.poly
    }

    // pub(crate) fn coeffs_hi_var(&self) -> (Ext, Ext) {
    //     coeffs_hi_var(&self.poly, &self.weights)
    // }

    pub(crate) fn coeffs_lo_var(&self) -> (Ext, Ext) {
        coeffs_lo_var(&self.poly, &self.weights)
    }

    pub(crate) fn write_poly<Transcript>(
        &self,
        transcript: &mut Transcript,
    ) -> Result<(), common::Error>
    where
        Transcript: Writer<Ext>,
    {
        transcript.write_many(&self.poly)
    }

    pub fn combine_claims(
        &mut self,
        sum: &mut Ext,
        alpha: Ext,
        eq_claims: &[EqClaim<Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) {
        combine_claims::<F, Ext>(&mut self.weights, sum, alpha, eq_claims, pow_claims);
    }

    pub(crate) fn prod(&self) -> Ext {
        dot_product(self.poly.iter().copied(), self.weights.iter().copied())
    }

    pub(crate) fn fix_lo_var(&mut self, r: Ext) {
        self.poly.fix_lo_var_mut(r);
        self.weights.fix_lo_var_mut(r);
    }

    // pub(crate) fn fix_hi_var(&mut self, r: Ext) {
    //     self.poly.fix_hi_var_mut(r);
    //     self.weights.fix_hi_var_mut(r);
    // }
}

#[derive(Debug, Clone)]
pub(crate) enum ExpressionPacked<F: Field, Ext: ExtensionField<F>> {
    Packed {
        poly: Poly<Ext::ExtensionPacking>,
        weights: Poly<Ext::ExtensionPacking>,
    },
    Small {
        poly: Poly<Ext>,
        weights: Poly<Ext>,
    },
}

impl<F: Field, Ext: ExtensionField<F>> ExpressionPacked<F, Ext> {
    pub(crate) fn new_packed(
        poly: Poly<Ext::ExtensionPacking>,
        weights: Poly<Ext::ExtensionPacking>,
    ) -> Self {
        Self::Packed { poly, weights }
    }

    // pub(crate) fn new_unpacked(poly: Poly<Ext>, weights: Poly<Ext>) -> Self {
    //     Self::Small { poly, weights }
    // }

    pub(crate) fn k(&self) -> usize {
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

    pub(crate) fn round<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        sum: &mut Ext,
    ) -> Result<Ext, common::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let (c0, c2) = self.coeffs();
        transcript.write_many(&[c0, c2])?;

        let r = Challenge::<F, Ext>::draw(transcript);
        self.fix(r);
        *sum = extrapolate_012(c0, *sum - c0, c2, r);

        debug_assert_eq!(*sum, self.prod());
        Ok(r)
    }

    pub(crate) fn coeffs(&self) -> (Ext, Ext) {
        match self {
            Self::Packed { poly, weights } => {
                let (c0, c2) = coeffs_hi_var(poly, weights);
                let c0 = Ext::ExtensionPacking::to_ext_iter([c0]).sum::<Ext>();
                let c2 = Ext::ExtensionPacking::to_ext_iter([c2]).sum::<Ext>();
                (c0, c2)
            }
            Self::Small { poly, weights } => coeffs_hi_var(poly, weights),
        }
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn unpack_poly_into(&self, unpacked: &mut [Ext]) {
        match self {
            Self::Packed { poly, .. } => {
                assert_eq!(
                    unpacked.len(),
                    1 << (poly.k() + log2_strict_usize(F::Packing::WIDTH))
                );
                unpack_into::<F, Ext>(unpacked, poly);
            }
            Self::Small { poly, .. } => unpacked.copy_from_slice(poly),
        }
    }

    pub(crate) fn combine_claims(
        &mut self,
        sum: &mut Ext,
        alpha: Ext,
        eq_claims: &[EqClaim<Ext>],
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

    pub(crate) fn fix(&mut self, r: Ext) {
        match self {
            Self::Packed { poly, weights } => {
                poly.fix_hi_var_mut(r);
                weights.fix_hi_var_mut(r);
            }
            Self::Small { poly, weights } => {
                poly.fix_hi_var_mut(r);
                weights.fix_hi_var_mut(r);
            }
        }
        self.transition();
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

    pub(crate) fn prod(&self) -> Ext {
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
