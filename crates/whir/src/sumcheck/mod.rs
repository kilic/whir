use common::field::*;
use poly::{Point, Poly, eq::SplitEq};
use transcript::{Challenge, Reader};

mod combine;
mod expr;
pub mod prover;
pub mod prover2;
mod svo;
#[cfg(test)]
pub(crate) mod test;
#[cfg(test)]
pub(crate) mod test2;
pub mod verifier;
pub mod verifier2;

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Selector(Vec<bool>);

impl Selector {
    pub(super) fn new(k: usize, index: usize) -> Self {
        assert!(index < (1 << k));
        Self((0..k).map(|i| (index >> i) & 1 == 1).collect::<Vec<_>>())
    }

    pub(super) fn index(&self) -> usize {
        self.0
            .iter()
            .enumerate()
            .map(|(i, &bit)| (bit as usize) << i)
            .sum()
    }

    pub(super) fn point<F: Field>(&self) -> Point<F> {
        Point::hypercube(self.index(), self.k())
    }

    pub(super) fn k(&self) -> usize {
        self.0.len()
    }

    pub(super) fn lift<Ext: Field>(&self, other: &Point<Ext>) -> Point<Ext> {
        other.concat(self.point())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EqClaim<Ext: Field> {
    pub(crate) point: Point<Ext>,
    pub(crate) eval: Ext,
}

#[derive(Debug, Clone)]
pub struct SplitEqClaim<F: Field, Ext: ExtensionField<F>> {
    pub(crate) split: SplitEq<F, Ext>,
    pub(crate) point: Point<Ext>,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> PartialEq for SplitEqClaim<F, Ext> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.eval == other.eval
    }
}

impl<F: Field, Ext: ExtensionField<F>> Eq for SplitEqClaim<F, Ext> {}

impl<F: Field, Ext: ExtensionField<F>> SplitEqClaim<F, Ext> {
    pub fn new(point: Point<Ext>, poly: &Poly<F>) -> Self {
        let split = SplitEq::new_packed(&point, Ext::ONE);
        let eval = split.eval_base(poly);
        Self { split, point, eval }
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }

    pub fn eq(&self) -> Poly<Ext> {
        self.point.eq(Ext::ONE)
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn point(&self) -> &SplitEq<F, Ext> {
        &self.split
    }
    // pub fn split(&self) -> &SplitEq<F, Ext> {
    //     &self.point

    // }
}

impl<Ext: Field> EqClaim<Ext> {
    pub fn new(point: Point<Ext>, eval: Ext) -> Self {
        Self { point, eval }
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }

    pub fn eq(&self) -> Poly<Ext> {
        self.point.eq(Ext::ONE)
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn point(&self) -> &Point<Ext> {
        &self.point
    }

    pub fn read<Transcript, F: Field>(
        transcript: &mut Transcript,
        k: usize,
    ) -> Result<Self, common::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
        Ext: ExtensionField<F>,
    {
        let point = Point::expand(k, transcript.draw());
        let eval: Ext = transcript.read()?;
        Ok(Self::new(point, eval))
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
