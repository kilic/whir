use itertools::Itertools;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};

use crate::{
    sumcheck::combine::combine_claims,
    sumcheck::{EqClaim, PowClaim, extrapolate_012},
};
use common::field::*;
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

// TODO: move combine claims here?
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

    pub(crate) fn round_hi_var<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        sum: &mut Ext,
    ) -> Result<Ext, common::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let (c0, c2) = coeffs_hi_var(&self.poly, &self.weights);

        transcript.write_many(&[c0, c2])?;

        let r = Challenge::<F, Ext>::draw(transcript);
        self.fix_hi_var(r);
        *sum = extrapolate_012(c0, *sum - c0, c2, r);

        debug_assert_eq!(*sum, self.prod());
        Ok(r)
    }

    pub(crate) fn poly(&self) -> &Poly<Ext> {
        &self.poly
    }

    pub(crate) fn coeffs_hi_var(&self) -> (Ext, Ext) {
        coeffs_hi_var(&self.poly, &self.weights)
    }

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

    pub(crate) fn fix_hi_var(&mut self, r: Ext) {
        self.poly.fix_hi_var_mut(r);
        self.weights.fix_hi_var_mut(r);
    }
}
