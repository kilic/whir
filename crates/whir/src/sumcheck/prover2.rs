use std::collections::{BTreeMap, HashMap};

use crate::{
    commit::commit_base_interleaved,
    sumcheck::{
        EqClaim, PowClaim, Selector, SplitEqClaim,
        expr::{Expression, coeffs_hi_var},
        extrapolate_012,
    },
};
use common::{field::*, utils::VecOps};
use merkle::{MerkleData, MerkleTree};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_util::log2_ceil_usize;
use poly::{Point, Poly, eq::SplitEq};
use transcript::{Challenge, Writer};

#[derive(Debug, Clone)]
pub struct ProverStack<F: Field, Ext: ExtensionField<F>> {
    polys: Vec<Poly<F>>,
    claims: Vec<SplitEqClaim<F, Ext>>,
    point_to_claim_map: HashMap<Point<Ext>, Vec<usize>>, // TODO: disallow duplicate, remove hashmap
    poly_to_claim_map: HashMap<usize, Vec<usize>>,       // TODO: disallow duplicate, remove hashmap
}

impl<F: TwoAdicField, Ext: ExtensionField<F>> ProverStack<F, Ext> {
    pub fn new() -> Self {
        ProverStack {
            polys: Default::default(),
            claims: Default::default(),
            point_to_claim_map: Default::default(),
            poly_to_claim_map: Default::default(),
        }
    }

    pub fn k_poly(&self, id: usize) -> usize {
        self.polys[id].k()
    }

    pub fn poly(&self, id: usize) -> &Poly<F> {
        &self.polys[id]
    }

    pub fn register(&mut self, poly: Poly<F>) -> usize {
        self.polys.push(poly);
        self.polys.len() - 1
    }

    pub fn eval<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        id: usize,
        point: &Point<Ext>,
    ) -> Result<(), common::Error>
    where
        Transcript: Writer<Ext>,
    {
        assert_eq!(point.k(), self.k_poly(id));

        self.point_to_claim_map
            .entry(point.clone())
            .or_default()
            .push(self.claims.len());

        self.poly_to_claim_map
            .entry(id)
            .or_default()
            .push(self.claims.len());

        let claim = SplitEqClaim::new(point.clone(), &self.polys[id]);
        transcript.write(claim.eval)?;
        self.claims.push(claim);
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub fn commit<Transcript, Dft, MT>(
        self,
        dft: &Dft,
        transcript: &mut Transcript,
        mt: &MT,
        folding: usize,
        rate: usize,
    ) -> Result<(ProverLayout<F, Ext>, MT::MerkleData), common::Error>
    where
        Transcript:
            Writer<Ext> + Writer<<MT::MerkleData as MerkleData<F>>::Digest> + Challenge<F, Ext>,
        Dft: TwoAdicSubgroupDft<F>,
        MT: MerkleTree<F>,
    {
        let layout = self.layout();
        let width = 1 << (layout.k - folding);
        let pad_size = layout.off.div_ceil(width) * width;
        let data =
            commit_base_interleaved(dft, transcript, &layout.poly[..pad_size], rate, folding, mt)?;

        Ok((layout, data))
    }

    #[tracing::instrument(skip_all)]
    pub fn layout(self) -> ProverLayout<F, Ext> {
        let mut order = (0..self.polys.len()).collect::<Vec<usize>>();
        order.sort_by_key(|&i| self.k_poly(i));
        let k = log2_ceil_usize(self.polys.iter().map(|p| p.len()).sum::<usize>());

        let mut off = 0usize;
        let mut layout = BTreeMap::new();
        order.iter().rev().for_each(|&id| {
            let k_poly = self.k_poly(id);
            let size = 1usize << k_poly;
            let selector = Selector::new(k - k_poly, off >> k_poly);
            assert!(layout.insert(selector, id).is_none());
            off += size;
        });

        let mut poly = Poly::zero(k);
        for (selector, &id) in layout.iter() {
            let src = &self.polys[id];
            let off = selector.index() << src.k();
            poly[off..off + src.len()].copy_from_slice(src);
        }

        ProverLayout {
            k,
            poly,
            off,
            stack: self,
            layout,
            virtual_claims: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProverLayout<F: Field, Ext: ExtensionField<F>> {
    k: usize,
    poly: Poly<F>, // TODO: we should keep this as vec
    off: usize,
    stack: ProverStack<F, Ext>,
    layout: BTreeMap<Selector, usize>,
    virtual_claims: Vec<SplitEqClaim<F, Ext>>,
}

impl<F: Field, Ext: ExtensionField<F>> ProverLayout<F, Ext> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n_columns(&self, folding: usize) -> usize {
        let width = 1 << (self.k - folding);
        self.off.div_ceil(width)
    }

    pub fn n_claims(&self) -> usize {
        self.stack.claims.len() + self.virtual_claims.len()
    }

    pub fn poly(&self) -> &Poly<F> {
        &self.poly
    }

    fn claims_in_layout_order(&self) -> impl Iterator<Item = (&Selector, usize)> {
        self.layout.iter().flat_map(move |(selector, &poly_id)| {
            self.stack
                .poly_to_claim_map
                .get(&poly_id)
                .into_iter()
                .flatten()
                .copied()
                .map(move |claim_idx| (selector, claim_idx))
        })
    }

    fn claim_alpha_powers(&self, alpha: Ext) -> Vec<Ext> {
        let mut alphas = vec![Ext::ZERO; self.stack.claims.len()];
        self.claims_in_layout_order()
            .map(|(_, claim_id)| claim_id)
            .zip(alpha.powers())
            .for_each(|(claim_id, alpha_i)| alphas[claim_id] = alpha_i);
        alphas
    }

    fn sum(&self, alpha: Ext) -> Ext {
        self.claims_in_layout_order()
            .map(|(_, claim_id)| self.stack.claims[claim_id].eval())
            .chain(self.virtual_claims.iter().map(SplitEqClaim::eval))
            .collect::<Vec<_>>()
            .into_iter()
            .horner(alpha)
    }

    #[tracing::instrument(skip_all)]
    fn combine_eqs(&self, alpha: Ext) -> Poly<Ext> {
        let mut out = Poly::<Ext>::zero(self.k);
        let mut alpha_i = Ext::ONE;

        self.claims_in_layout_order()
            .for_each(|(selector, claim_idx)| {
                let point = self.stack.claims[claim_idx].point();
                let size = 1 << point.k();
                let off = selector.index() * size;
                point.combine_into(&mut out[off..off + size], Some(alpha_i));
                alpha_i *= alpha;
            });

        self.virtual_claims.iter().for_each(|claim| {
            tracing::info_span!("combine virtual")
                .in_scope(|| claim.point().combine_into(&mut out, Some(alpha_i)));
            alpha_i *= alpha;
        });
        out
    }

    #[tracing::instrument(skip_all)]
    pub fn write_virtual_eval<Transcript>(
        &mut self,
        transcript: &mut Transcript,
    ) -> Result<(), common::Error>
    where
        Transcript: Writer<Ext> + Challenge<F, Ext>,
    {
        let point = Point::expand(self.k(), transcript.draw());
        let split = SplitEq::new_packed(&point, Ext::ONE);
        let eval = split.eval_base(self.poly());
        transcript.write(eval)?;
        self.virtual_claims
            .push(SplitEqClaim { point, split, eval });
        Ok(())
    }

    #[tracing::instrument(skip_all, fields(k = self.k()))]
    pub fn new_prover<Transcript>(
        self,
        transcript: &mut Transcript,
        d: usize,
    ) -> Result<SumcheckProver<F, Ext>, common::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        assert!(d <= self.k());
        let alpha: Ext = transcript.draw();
        let alphas = self.claim_alpha_powers(alpha);
        debug_assert_eq!(alphas.len(), self.stack.claims.len());

        let mut sum = self.sum(alpha);
        let mut weights = self.combine_eqs(alpha);

        let (c0, c2) = coeffs_hi_var(self.poly(), &weights);
        transcript.write_many(&[c0, c2])?;
        let r: Ext = transcript.draw();
        sum = extrapolate_012(c0, sum - c0, c2, r);
        let poly = self.poly().fix_hi_var(r);
        weights.fix_hi_var_mut(r);
        let mut expr = Expression::new(poly, weights);

        let rs = core::iter::once(Ok(r))
            .chain((1..d).map(|_| {
                let (c0, c2) = expr.coeffs_hi_var();
                transcript.write_many(&[c0, c2])?;
                let r: Ext = transcript.draw();
                sum = extrapolate_012(c0, sum - c0, c2, r);
                expr.fix_hi_var(r);
                debug_assert_eq!(expr.prod(), sum);
                Ok(r)
            }))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        Ok(SumcheckProver {
            sum,
            rs,
            expr,
            alpha,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SumcheckProver<F: Field, Ext: ExtensionField<F>> {
    sum: Ext,
    rs: Point<Ext>,
    expr: Expression<F, Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckProver<F, Ext> {
    pub fn rs(&self) -> &Point<Ext> {
        &self.rs
    }

    pub fn alpha(&self) -> Ext {
        self.alpha
    }

    pub(crate) fn poly(&self) -> &Poly<Ext> {
        self.expr.poly()
    }

    #[tracing::instrument(skip_all, fields(k = self.k(), eq = eq_claims.len(), pow = pow_claims.len()))]
    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        eq_claims: &[EqClaim<Ext>],
        pow_claims: &[PowClaim<F, Ext>],
    ) -> Result<Point<Ext>, common::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        self.alpha = transcript.draw();
        self.expr
            .combine_claims(&mut self.sum, self.alpha, eq_claims, pow_claims);
        let rs = (0..d)
            .map(|_| self.expr.round_hi_var(transcript, &mut self.sum))
            .collect::<Result<Vec<_>, _>>()?;

        self.rs.extend(rs.iter());
        Ok(rs.into())
    }

    pub fn write_poly<Transcript>(&self, transcript: &mut Transcript) -> Result<(), common::Error>
    where
        Transcript: Writer<Ext>,
    {
        self.expr.write_poly(transcript)
    }

    pub fn k(&self) -> usize {
        self.expr.k()
    }
}
