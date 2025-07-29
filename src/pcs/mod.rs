use p3_field::{ExtensionField, Field};

use crate::poly::{Point, Poly};

pub mod sumcheck;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Claim<F> {
    point: Point<F>,
    eval: F,
}

impl<F: Field> Claim<F> {
    pub fn new(point: Point<F>, eval: F) -> Self {
        Self { point, eval }
    }

    pub fn evaluate<BaseField: Field>(poly: &Poly<BaseField>, point: &Point<F>) -> Self
    where
        F: ExtensionField<BaseField>,
    {
        let eval = poly.eval_lagrange(point);
        Self::new(point.clone(), eval)
    }

    pub fn point(&self) -> &Point<F> {
        &self.point
    }

    pub fn eval(&self) -> F {
        self.eval
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }
}
