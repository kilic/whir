use crate::{
    p3_field_prelude::*,
    pcs::sumcheck::{EqClaim, PowClaim, extrapolate_012},
    poly::{Point, Poly, eval_eq_xy, eval_pow_xy},
    transcript::{Challenge, Reader},
    utils::VecOps,
};

#[derive(Debug, Clone)]
pub(super) struct MultiRound<F: Field, Ext: ExtensionField<F>> {
    eq_points: Vec<Point<Ext>>,
    pow_points: Vec<Point<F>>,
    rs: Point<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> MultiRound<F, Ext> {
    pub(super) fn new(
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

    pub(super) fn extend(&mut self, rs: &Point<Ext>) {
        self.rs.extend(rs.iter());
    }

    pub(super) fn eval(&self, poly: &Poly<Ext>) -> Ext {
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
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate_012(v0, *sum - v0, v2, r);
    Ok(r)
}

pub struct SumcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    pub k: usize,
    sum: Ext,
    multi_rounds: Vec<MultiRound<F, Ext>>,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            sum: Ext::ZERO,
            multi_rounds: Vec::new(),
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
        let sum = self
            .multi_rounds
            .iter()
            .map(|round| round.eval(&poly))
            .sum::<Ext>();
        (self.sum == sum).then_some(()).ok_or(crate::Error::Verify)
    }
}
