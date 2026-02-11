mod combine;
pub mod prover;
pub mod split;
pub(crate) mod svo;
#[cfg(test)]
mod test;
pub mod verifier;

use crate::p3_field_prelude::*;
use crate::poly::{Point, Poly};
pub use prover::Sumcheck;
pub use verifier::SumcheckVerifier;

pub(super) fn lagrange_weights_012<F: Field>(r: F) -> [F; 3] {
    let inv_two = F::TWO.inverse();
    let l0 = (r - F::ONE) * (r - F::TWO) * inv_two;
    let l1 = r * (F::TWO - r);
    let l2 = (r * (r - F::ONE)) * inv_two;
    [l0, l1, l2]
}

pub fn extrapolate_012<F: Field>(e0: F, e1: F, e2: F, r: F) -> F {
    let [w0, w1, w2] = lagrange_weights_012(r);
    e0 * w0 + e1 * w1 + e2 * w2
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EqClaim<F: Field, Ext: ExtensionField<F>> {
    pub(crate) point: Point<F>,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> EqClaim<F, Ext> {
    pub fn new(point: Point<F>, eval: Ext) -> Self {
        Self { point, eval }
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }

    pub fn eq(&self) -> Poly<F> {
        self.point.eq(F::ONE)
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn point(&self) -> Point<F> {
        self.point.clone()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PowClaim<F: Field, Ext: ExtensionField<F>> {
    pub(crate) k: usize,
    pub(crate) var: F,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> PowClaim<F, Ext> {
    pub fn new(var: F, eval: Ext, k: usize) -> Self {
        Self { var, eval, k }
    }

    pub fn pow(&self, k: usize) -> Poly<F> {
        self.var.powers().take(k).collect().into()
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn var(&self) -> F {
        self.var
    }

    pub fn point(&self) -> Point<F> {
        Point::expand(self.k, self.var)
    }
}
