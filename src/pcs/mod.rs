use p3_field::{ExtensionField, Field};

use crate::{
    poly::{Point, Poly},
    transcript::{Challenge, Reader, Writer},
};

pub mod params;
pub mod stir;
pub mod sumcheck;
pub mod whir;

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

pub(crate) fn make_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    n: usize,
    poly: &Poly<F>,
) -> Result<Vec<Claim<Ext>>, crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    (0..n)
        .map(|_| {
            let var: Ext = transcript.draw();
            let point = Point::expand(poly.k(), var);
            let claim = Claim::evaluate(poly, &point);
            transcript.write(claim.eval())?;
            Ok(claim)
        })
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn get_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    k: usize,
    n: usize,
) -> Result<Vec<Claim<Ext>>, crate::Error>
where
    Transcript: Reader<Ext> + Challenge<F, Ext>,
{
    (0..n)
        .map(|_| {
            let var: Ext = transcript.draw();
            let point = Point::expand(k, var);
            let eval = transcript.read()?;
            Ok(Claim::new(point, eval))
        })
        .collect::<Result<Vec<_>, _>>()
}
