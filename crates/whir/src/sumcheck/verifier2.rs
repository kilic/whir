use crate::sumcheck::{EqClaim, PowClaim, Selector, extrapolate_012};
use common::{field::*, utils::VecOps};
use p3_util::log2_ceil_usize;
use poly::{eval_eq_xy, eval_pow_xy, prelude::*};
use std::collections::BTreeMap;
use transcript::{Challenge, Reader};

fn reduce<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
) -> Result<Ext, common::Error>
where
    Transcript: Reader<Ext> + Challenge<F, Ext>,
{
    let v0: Ext = transcript.read()?;
    let v2 = transcript.read()?;
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate_012(v0, *sum - v0, v2, r);
    Ok(r)
}

struct FoldingRounds<F: Field, Ext: ExtensionField<F>> {
    eq_points: Vec<Point<Ext>>,
    pow_points: Vec<Point<F>>,
    rs: Point<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> FoldingRounds<F, Ext> {
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

    fn eval(&self, poly: &Poly<Ext>) -> Ext {
        let rs = &self.rs.reversed();
        let weights = self
            .eq_points
            .iter()
            .map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (z_poly, z_fix) = zs.split_at(off);
                eval_eq_xy(&z_fix, rs) * poly.eval_ext(&z_poly.as_ext::<Ext>())
            })
            .chain(self.pow_points.iter().map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (z_poly, z_fix) = zs.split_at(off);
                eval_pow_xy(&z_fix, &rs)
                    * poly.eval_univariate(Ext::from(z_poly.first().copied().unwrap()))
            }))
            .collect::<Vec<_>>();
        weights.iter().horner(self.alpha)
    }
}

#[derive(Debug, Clone)]
struct PolyEvaluator<Ext: Field> {
    k: usize,
    claims: Vec<EqClaim<Ext>>,
}

impl<Ext: Field> PolyEvaluator<Ext> {
    fn new(k: usize) -> Self {
        PolyEvaluator { claims: vec![], k }
    }

    fn k(&self) -> usize {
        self.k
    }

    fn size(&self) -> usize {
        1 << self.k()
    }

    fn read_eval<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        point: Point<Ext>,
    ) -> Result<Ext, common::Error>
    where
        Transcript: Reader<Ext>,
    {
        assert_eq!(point.k(), self.k());
        let eval = transcript.read()?;
        self.claims.push(EqClaim::new(point, eval));
        Ok(eval)
    }
}

pub struct VerifierStack<F: Field, Ext: ExtensionField<F>> {
    evs: Vec<PolyEvaluator<Ext>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> VerifierStack<F, Ext> {
    pub fn new() -> Self {
        Self {
            evs: vec![],
            _marker: std::marker::PhantomData,
        }
    }

    pub fn k_poly(&self, id: usize) -> usize {
        self.evs[id].k()
    }

    pub fn register(&mut self, k: usize) -> usize {
        self.evs.push(PolyEvaluator::new(k));
        self.evs.len() - 1
    }

    pub fn read_eval<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        id: usize,
        point: &Point<Ext>,
    ) -> Result<Ext, common::Error>
    where
        Transcript: Reader<Ext>,
    {
        self.evs[id].read_eval(transcript, point.clone())
    }

    pub fn layout(self) -> VerifierLayout<F, Ext> {
        VerifierLayout::new(self.evs)
    }
}

#[derive(Debug, Clone)]
pub struct VerifierLayout<F: Field, Ext: ExtensionField<F>> {
    k: usize,
    layout: BTreeMap<Selector, PolyEvaluator<Ext>>,
    virtual_claims: Vec<EqClaim<Ext>>,
    off: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> VerifierLayout<F, Ext> {
    fn new(mut evs: Vec<PolyEvaluator<Ext>>) -> Self {
        evs.sort_by_key(PolyEvaluator::k);
        let k = log2_ceil_usize(evs.iter().map(PolyEvaluator::size).sum::<usize>());

        let mut off = 0usize;
        let mut layout = BTreeMap::<Selector, PolyEvaluator<Ext>>::new();
        for ev in evs.into_iter().rev() {
            let k_ev = ev.k();
            let size = 1usize << k_ev;
            let selector = Selector::new(k - k_ev, off >> k_ev);
            assert!(layout.insert(selector, ev).is_none());
            off += size;
        }
        Self {
            k,
            layout,
            off,
            virtual_claims: vec![],
            _marker: std::marker::PhantomData,
        }
    }

    pub fn n_columns(&self, folding: usize) -> usize {
        let width = 1 << (self.k - folding);
        self.off.div_ceil(width)
    }

    pub fn read_virtual_eval<Transcript>(
        &mut self,
        transcript: &mut Transcript,
    ) -> Result<Ext, common::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        let point = transcript.draw();
        let point = Point::expand(self.k(), point);
        let eval = transcript.read()?;
        let claim = EqClaim::new(point, eval);
        self.virtual_claims.push(claim);
        Ok(eval)
    }

    pub fn k(&self) -> usize {
        self.k
    }

    fn sum(&self, alpha: Ext) -> Ext {
        self.layout
            .values()
            .flat_map(|ev| ev.claims.iter().map(EqClaim::eval))
            .chain(self.virtual_claims.iter().map(EqClaim::eval))
            .horner(alpha)
    }

    fn lift(&self) -> Vec<Point<Ext>> {
        self.layout
            .iter()
            .flat_map(|(selector, ev)| ev.claims.iter().map(|claim| selector.lift(claim.point())))
            .chain(
                self.virtual_claims
                    .iter()
                    .map(|claim| claim.point().clone()),
            )
            .collect::<Vec<_>>()
    }
}

pub struct SumcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    multi_rounds: Vec<FoldingRounds<F, Ext>>,
    k: usize,
    sum: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> {
    pub fn new<Transcript>(
        transcript: &mut Transcript,
        layout: VerifierLayout<F, Ext>,
        d: usize,
    ) -> Result<(Self, Point<Ext>), common::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        let alpha: Ext = transcript.draw();
        let mut sum = layout.sum(alpha);

        let round_rs: Point<Ext> = (0..d)
            .map(|_| reduce::<_, F, Ext>(transcript, &mut sum))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        let k = layout.k();
        let points = layout.lift();
        let multi_rounds = vec![FoldingRounds::new(points, vec![], round_rs.clone(), alpha)];

        Ok((
            Self {
                multi_rounds,
                k: k - d,
                sum,
            },
            round_rs,
        ))
    }

    pub fn k(&self) -> usize {
        self.k
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        eq_claims: &[EqClaim<Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) -> Result<Point<Ext>, common::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        eq_claims.iter().for_each(|o| assert_eq!(o.k(), self.k));
        pow_claims.iter().for_each(|o| assert_eq!(o.k, self.k));

        let alpha: Ext = transcript.draw();
        self.sum += eq_claims
            .iter()
            .map(EqClaim::<Ext>::eval)
            .chain(pow_claims.iter().map(PowClaim::<F, Ext>::eval))
            .horner(alpha);

        let round_rs: Point<Ext> = (0..d)
            .map(|_| reduce::<_, F, Ext>(transcript, &mut self.sum))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        self.multi_rounds
            .iter_mut()
            .for_each(|round| round.extend(&round_rs));

        let eq_points = eq_claims
            .iter()
            .map(EqClaim::point)
            .cloned()
            .collect::<Vec<_>>();
        let pow_points = pow_claims.iter().map(PowClaim::point).collect::<Vec<_>>();
        self.multi_rounds.push(FoldingRounds::new(
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
    ) -> Result<(), common::Error>
    where
        Transcript: Reader<Ext>,
    {
        let poly = poly.map_or_else(|| Ok(Poly::new(transcript.read_many(1 << self.k)?)), Ok)?;
        let sum = self
            .multi_rounds
            .iter()
            .map(|round| round.eval(&poly))
            .sum::<Ext>();
        (self.sum == sum).then_some(()).ok_or(common::Error::Verify)
    }
}
