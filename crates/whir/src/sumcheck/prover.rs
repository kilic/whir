use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::sumcheck::{
    EqClaim, PowClaim, Selector,
    combine::pow::combine_pows_scaled,
    expr::Expression,
    extrapolate_012,
    svo::{SvoAccumulators, SvoClaim, SvoPoint, lagrange_weights_012_multi},
};
use common::{field::*, utils::VecOps};
use p3_field::TwoAdicField;
use p3_matrix::{Matrix, dense::RowMajorMatrixView};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use poly::{Point, Poly, eq::SplitEq, eval_eq_xy};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use transcript::{Challenge, Writer};

#[derive(Debug, Clone)]
struct VirtualClaim<Ext: Field> {
    point: Point<Ext>,
    accumulators: SvoAccumulators<Ext>,
    eval: Ext,
}

impl<Ext: Field> VirtualClaim<Ext> {
    fn eval(&self) -> &Ext {
        &self.eval
    }
}

#[derive(Debug, Clone)]
pub struct ProverStack<F: Field, Ext: ExtensionField<F>> {
    k_svo: usize,
    polys: Vec<Poly<F>>,
    claims: Vec<SvoClaim<F, Ext>>,
    point_to_claim_map: HashMap<Point<Ext>, Vec<usize>>, // TODO: disallow duplicate, remove hashmap
    poly_to_claim_map: HashMap<usize, Vec<usize>>,       // TODO: disallow duplicate, remove hashmap
}

impl<F: Field, Ext: ExtensionField<F>> ProverStack<F, Ext> {
    pub fn new(k_svo: usize) -> Self {
        ProverStack {
            k_svo,
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

        let point = SvoPoint::new(self.k_svo, point);
        let claim = SvoClaim::new(point, &self.polys[id]);
        transcript.write(claim.eval)?;
        self.claims.push(claim);
        Ok(())
    }

    pub fn layout(self) -> ProverLayout<F, Ext> {
        let mut order = (0..self.polys.len()).collect::<Vec<usize>>();
        order.sort_by_key(|&i| self.k_poly(i));

        let k = log2_ceil_usize(self.polys.iter().map(|p| p.len()).sum::<usize>());
        assert!(k >= self.k_svo);

        let mut off = 0usize;
        let mut layout = BTreeMap::new();
        order.iter().rev().for_each(|&id| {
            let k_poly = self.k_poly(id);
            let size = 1usize << k_poly;
            let selector = Selector::new(k - k_poly, off >> k_poly);
            assert!(layout.insert(selector, id).is_none());
            off += size;
        });

        ProverLayout {
            k,
            stack: self,
            layout,
            virtual_claims: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProverLayout<F: Field, Ext: ExtensionField<F>> {
    k: usize,
    stack: ProverStack<F, Ext>,
    layout: BTreeMap<Selector, usize>,
    virtual_claims: Vec<VirtualClaim<Ext>>,
}

impl<F: Field, Ext: ExtensionField<F>> ProverLayout<F, Ext> {
    pub fn k(&self) -> usize {
        self.k
    }

    pub fn k_svo(&self) -> usize {
        self.stack.k_svo
    }

    pub fn n_claims(&self) -> usize {
        self.stack.claims.len() + self.virtual_claims.len()
    }

    fn claims_in_layout_order(&self) -> impl Iterator<Item = (&Selector, usize)> + '_ {
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
            .chain(self.virtual_claims.iter().map(VirtualClaim::eval))
            .collect::<Vec<_>>()
            .into_iter()
            .horner(alpha)
    }

    #[tracing::instrument(skip_all)]
    fn combine_eqs(&self, rs: &Point<Ext>, alpha: Ext) -> Poly<Ext> {
        assert!(rs.k() <= self.k);
        let mut out = Poly::<Ext>::zero(self.k - rs.len());
        let mut alpha_i = Ext::ONE;

        self.claims_in_layout_order()
            .for_each(|(selector, claim_idx)| {
                let point = self.stack.claims[claim_idx].point();
                let size = 1 << point.k_split();
                let off = selector.index() * size;
                point.combine_into(&mut out[off..off + size], rs, alpha_i);
                alpha_i *= alpha;
            });

        self.virtual_claims.iter().for_each(|claim| {
            tracing::info_span!("combine virtual").in_scope(|| {
                let (svo, rest) = claim.point.split_at(rs.k());
                let scale = alpha_i * eval_eq_xy(&svo, rs);
                SplitEq::new(&rest, scale).combine_into(&mut out, None);
            });
            alpha_i *= alpha;
        });
        out
    }

    #[tracing::instrument(skip_all)]
    fn compress_stacked(&self, rs: &Point<Ext>) -> Poly<Ext> {
        assert!(rs.k() <= self.k);
        let mut out = Poly::<Ext>::zero(self.k - rs.k());
        for (selector, &id) in self.layout.iter() {
            assert!(rs.k() <= self.stack.k_poly(id));
            let poly = &self.stack.polys[id];
            let compressed = poly.compress_lo(rs, Ext::ONE);
            let off = selector.index() << (poly.k() - rs.k());
            out[off..off + compressed.len()].copy_from_slice(&compressed);
        }
        out
    }

    #[tracing::instrument(skip_all)]
    pub fn expression(self, rs: &Point<Ext>, alpha: Ext) -> Expression<F, Ext> {
        let poly = self.compress_stacked(rs);
        let weights = self.combine_eqs(rs, alpha);
        Expression::new(poly, weights)
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
        let mut eval = Ext::ZERO;

        let (claims, weights): (Vec<_>, Vec<_>) = self
            .layout
            .iter()
            .map(|(selector, &poly_id)| {
                let poly = self.stack.poly(poly_id);
                let (local_point, rest) = point.split_at(poly.k());
                let local_point = SvoPoint::new(self.k_svo(), &local_point);
                let weight = eval_eq_xy(&selector.point(), &rest);
                let claim = SvoClaim::new(local_point, poly);
                eval += weight * claim.eval;
                (claim, weight)
            })
            .unzip();
        let accumulators = SvoClaim::calculate_accumulators(&claims, &weights);

        #[cfg(debug_assertions)]
        {
            let poly = self.compress_stacked(&vec![].into());
            let point = SvoPoint::<Ext, Ext>::new(self.k_svo(), &point);
            let claim = SvoClaim::<Ext, Ext>::new(point, &poly);
            assert_eq!(eval, claim.eval);
            let claim_accumulators = SvoClaim::calculate_accumulators(&[claim], &[Ext::ONE]);
            assert_eq!(accumulators, claim_accumulators);
        }

        let claim = VirtualClaim {
            point,
            accumulators,
            eval,
        };

        transcript.write(claim.eval)?;
        self.virtual_claims.push(claim);
        Ok(())
    }

    #[tracing::instrument(skip_all, fields(k = self.k()))]
    pub fn new_prover<Transcript>(
        self,
        transcript: &mut Transcript,
    ) -> Result<SumcheckProver<F, Ext>, common::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        assert!(self.k_svo() <= self.k());
        let alpha: Ext = transcript.draw();
        let alphas = self.claim_alpha_powers(alpha);
        debug_assert_eq!(alphas.len(), self.stack.claims.len());

        let mut sum = self.sum(alpha);
        let mut rs = Point::<Ext>::default();

        let accumulators0 = self
            .stack
            .point_to_claim_map
            .values()
            .map(|claim_ids| {
                let claims = claim_ids
                    .iter()
                    .map(|&idx| self.stack.claims[idx].clone())
                    .collect::<Vec<_>>();
                let alphas = claim_ids.iter().map(|&idx| alphas[idx]).collect::<Vec<_>>();
                SvoClaim::calculate_accumulators(&claims, &alphas)
            })
            .collect::<Vec<_>>();

        for round_idx in 0..self.k_svo() {
            let weights = lagrange_weights_012_multi(rs.as_slice());
            let mut acc0 = vec![Ext::ZERO; weights.len()];
            let mut acc2 = vec![Ext::ZERO; weights.len()];

            for accumulators in accumulators0.iter() {
                acc0.iter_mut()
                    .zip(accumulators[round_idx][0].iter())
                    .for_each(|(acc, &w)| *acc += w);
                acc2.iter_mut()
                    .zip(accumulators[round_idx][1].iter())
                    .for_each(|(acc, &w)| *acc += w);
            }

            for (vc, alpha) in self
                .virtual_claims
                .iter()
                .zip(alpha.powers().skip(alphas.len()))
            {
                acc0.iter_mut()
                    .zip(vc.accumulators[round_idx][0].iter())
                    .for_each(|(acc, &w)| *acc += alpha * w);
                acc2.iter_mut()
                    .zip(vc.accumulators[round_idx][1].iter())
                    .for_each(|(acc, &w)| *acc += alpha * w);
            }

            let c0 = dot_product::<Ext, _, _>(acc0.iter().copied(), weights.iter().copied());
            let c2 = dot_product::<Ext, _, _>(acc2.iter().copied(), weights.iter().copied());

            transcript.write_many(&[c0, c2])?;
            let r: Ext = transcript.draw();
            sum = extrapolate_012(c0, sum - c0, c2, r);
            rs.push(r);
        }

        let expr = self.expression(&rs, alpha);
        debug_assert_eq!(expr.prod(), sum);
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
            .map(|_| self.expr.round(transcript, &mut self.sum))
            .collect::<Result<Vec<_>, _>>()?;

        self.rs.extend(rs.iter());
        Ok(rs.into())
    }

    #[tracing::instrument(skip_all, fields(k = self.k(), eq = eq_claims.len(), pow = indexes.len()))]
    pub fn fold_with_code<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        eq_claims: &[EqClaim<Ext>],

        indexes: &[usize],
        data: RowMajorMatrixView<'_, F>,
    ) -> Result<Point<Ext>, common::Error>
    where
        F: TwoAdicField,
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        self.alpha = transcript.draw();
        let k_data = log2_strict_usize(data.height());
        assert!(d <= k_data);
        self.expr
            .combine_claims(&mut self.sum, self.alpha, eq_claims, &[]);

        let mut scales = self
            .alpha
            .powers()
            .skip(eq_claims.len())
            .take(indexes.len())
            .collect::<Vec<_>>();

        {
            let eq = self.rs().eq(Ext::ONE);
            for (&alpha, &index) in scales.iter().zip(indexes) {
                self.sum += alpha
                    * dot_product::<Ext, _, _>(
                        eq.iter().copied(),
                        data.row_slice(index).unwrap().iter().copied(),
                    );
            }
        }

        fn g<F: TwoAdicField, Ext: ExtensionField<F>>(
            k: usize,
            data: &BTreeMap<usize, Ext>,
            round: usize,
            index: usize,
            rs_cur: &[Ext],
            two_inv: F,
        ) -> Ext {
            if round == 0 {
                *data.get(&index).unwrap()
            } else {
                let l = 1 << (k - (round - 1));
                let mid = l / 2;
                let i_lo = index & (mid - 1);
                let i_hi = i_lo + mid;

                let g_lo = g(k, data, round - 1, i_lo, rs_cur, two_inv);
                let g_hi = g(k, data, round - 1, i_hi, rs_cur, two_inv);

                let omega = F::two_adic_generator(k - (round - 1));
                let even = (g_lo + g_hi) * two_inv;
                let odd = (g_lo - g_hi) * two_inv * omega.exp_u64(i_lo as u64).inverse();
                even + rs_cur[round - 1] * (odd - even)
            }
        }

        let omega = F::two_adic_generator(k_data);
        let mut ws = tracing::info_span!("twiddles1").in_scope(|| {
            indexes
                .iter()
                .map(|&index| omega.exp_u64(index as u64))
                .collect::<Vec<_>>()
        });

        let mut ts = tracing::info_span!("twiddles2").in_scope(|| {
            let l = 1 << k_data;
            let mid = l / 2;
            indexes
                .iter()
                .map(|&index| {
                    let i_lo = index & (mid - 1);
                    omega.exp_u64(i_lo as u64).inverse()
                })
                .collect::<Vec<_>>()
        });

        let mut recursive_indexes = BTreeSet::new();
        tracing::info_span!("flat indexes").in_scope(|| {
            (0..d).for_each(|round| {
                let stride = 1usize << (k_data - round - 1);
                let count = 1usize << (round + 1);
                indexes.iter().for_each(|&index| {
                    let base = index & (stride - 1);
                    for t in 0..count {
                        recursive_indexes.insert(base + t * stride);
                    }
                });
            });
        });

        let evals: BTreeMap<usize, Ext> = tracing::info_span!("evals").in_scope(|| {
            let eq = self.rs.eq(Ext::ONE);
            recursive_indexes
                .par_iter()
                .map(|&index| {
                    let row = data.row_slice(index).unwrap();
                    (index, dot_product(eq.iter().copied(), row.iter().copied()))
                })
                .collect()
        });

        let two_inv = F::TWO.inverse();
        let mut rs = Point::empty();
        for round in 0..d {
            let (mut c0, mut c2) = self.expr.coeffs();

            let l = 1 << (k_data - round);
            let mid = l / 2;

            for (((&index, &s), &wi), &ti) in indexes
                .iter()
                .zip(scales.iter())
                .zip(ws.iter())
                .zip(ts.iter())
            {
                let i_lo = index & (mid - 1);
                let i_hi = i_lo + mid;

                let g_lo = g(k_data, &evals, round, i_lo, &rs, two_inv);
                let g_hi = g(k_data, &evals, round, i_hi, &rs, two_inv);

                let even = (g_lo + g_hi) * two_inv;
                let odd = (g_lo - g_hi) * two_inv * ti;

                c0 += s * even;
                c2 += s * (odd.double() - even) * (wi.double() - F::ONE);
            }

            transcript.write_many(&[c0, c2])?;
            let r: Ext = transcript.draw();
            self.sum = extrapolate_012(c0, self.sum - c0, c2, r);
            rs.push(r);

            self.expr.fix_var(r);

            ws.iter_mut()
                .zip(ts.iter_mut())
                .zip(scales.iter_mut())
                .zip(indexes.iter())
                .for_each(|(((wi, ti), si), &index)| {
                    *si *= Ext::ONE + r * (-Ext::ONE + *wi);
                    *wi = wi.square();
                    *ti = ti.square();
                    if (index >> (k_data - round - 2)) & 1 == 1 {
                        *ti = -*ti;
                    }
                });
        }

        combine_pows_scaled(&mut self.expr.weights, &ws, &scales);

        debug_assert_eq!(self.expr.prod(), self.sum);
        self.rs.extend(rs.iter());
        Ok(rs)
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

#[cfg(test)]
mod test {
    use std::collections::BTreeMap;
    use std::collections::BTreeSet;

    use common::field::BinomialExtensionField;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_dft::TwoAdicSubgroupDft;
    use p3_field::Field;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::TwoAdicField;
    use p3_field::dot_product;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;
    use poly::Point;
    use poly::Poly;
    use rand::RngExt;
    use rayon::prelude::*;

    use crate::sumcheck::expr::coeffs;

    #[test]
    fn test_fold_with_code() {
        type F = KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;

        let dft = Radix2DFTSmallBatch::default();
        let k = 25;
        let folding = 5;
        let rate = 1;

        let mut rng = common::test::rng(1);
        let poly = Poly::rand(&mut rng, k);
        let width = 1 << (k - folding);
        let mut mat = RowMajorMatrix::new(poly.to_vec(), poly.len() / width);
        mat.pad_to_height(1 << (poly.k() + rate - folding), F::ZERO);
        let codeword = dft.dft_batch(mat).to_row_major_matrix();

        let k_data = log2_strict_usize(codeword.height());
        let rs: Point<Ext> = Point::expand(folding, rng.random());
        let mut poly = poly.compress_lo(&rs, Ext::ONE);

        let two_inv = F::TWO.inverse();

        let n_claims = 300;
        let indexes: Vec<usize> = (0..n_claims)
            .map(|_| rng.random_range(0..1 << k_data))
            .collect::<Vec<_>>();

        let mut recursive_indexes = BTreeSet::new();
        tracing::info_span!("flat indexes").in_scope(|| {
            (0..folding).for_each(|round| {
                let stride = 1usize << (k_data - round - 1);
                let count = 1usize << (round + 1);
                indexes.iter().for_each(|&index| {
                    let base = index & (stride - 1);
                    for t in 0..count {
                        recursive_indexes.insert(base + t * stride);
                    }
                });
            });
        });

        let evals: BTreeMap<usize, Ext> = tracing::info_span!("evals").in_scope(|| {
            // let mut evals: BTreeMap<usize, Ext> = Default::default();
            let eq = rs.eq(Ext::ONE);
            recursive_indexes
                .par_iter()
                .map(|&index| {
                    let row = codeword.row_slice(index).unwrap();
                    (index, dot_product(eq.iter().copied(), row.iter().copied()))
                })
                .collect()
        });

        let omega = F::two_adic_generator(k_data);
        let pows: Vec<Poly<F>> = indexes
            .iter()
            .map(|&index| {
                let wi = omega.exp_u64(index as u64);
                wi.powers().take(1 << poly.k()).collect().into()
            })
            .collect::<Vec<_>>();

        let alpha: Ext = rng.random();

        let mut pows_combined = Poly::zero(poly.k());
        for (pow, alpha) in pows.iter().zip(alpha.powers()) {
            pows_combined
                .iter_mut()
                .zip(pow.iter())
                .for_each(|(c_combined, &c_pow)| *c_combined += alpha * c_pow);
        }

        let mut scales = alpha.powers().take(indexes.len()).collect();
        let mut rs_cur = Point::empty();

        fn g(
            k: usize,
            data: &BTreeMap<usize, Ext>,
            round: usize,
            index: usize,
            rs_cur: &[Ext],
            two_inv: F,
        ) -> Ext {
            if round == 0 {
                *data.get(&index).unwrap()
            } else {
                let l = 1 << (k - (round - 1));
                let mid = l / 2;
                let i_lo = index & (mid - 1);
                let i_hi = i_lo + mid;

                let g_lo = g(k, data, round - 1, i_lo, rs_cur, two_inv);
                let g_hi = g(k, data, round - 1, i_hi, rs_cur, two_inv);

                let omega = F::two_adic_generator(k - (round - 1));
                let even = (g_lo + g_hi) * two_inv;
                let odd = (g_lo - g_hi) * two_inv * omega.exp_u64(i_lo as u64).inverse();
                even + rs_cur[round - 1] * (odd - even)
            }
        }

        let mut ws = tracing::info_span!("twiddles1").in_scope(|| {
            indexes
                .iter()
                .map(|&index| omega.exp_u64(index as u64))
                .collect::<Vec<_>>()
        });

        let mut ts = tracing::info_span!("twiddles2").in_scope(|| {
            let l = 1 << k_data;
            let mid = l / 2;
            indexes
                .iter()
                .map(|&index| {
                    let i_lo = index & (mid - 1);
                    omega.exp_u64(i_lo as u64).inverse()
                })
                .collect::<Vec<_>>()
        });

        for round in 0..folding {
            tracing::info_span!("Folding").in_scope(|| {
                let l = 1 << (k_data - round);
                let mid = l / 2;

                let mut c0_pow = Ext::ZERO;
                let mut c2_pow = Ext::ZERO;
                for (((&index, &s), &wi), &ti) in indexes
                    .iter()
                    .zip(scales.iter())
                    .zip(ws.iter())
                    .zip(ts.iter())
                {
                    let i_lo = index & (mid - 1);
                    let i_hi = i_lo + mid;

                    let g_lo = g(k_data, &evals, round, i_lo, &rs_cur, two_inv);
                    let g_hi = g(k_data, &evals, round, i_hi, &rs_cur, two_inv);

                    let even = (g_lo + g_hi) * two_inv;
                    let odd = (g_lo - g_hi) * two_inv * ti;

                    c0_pow += s * even;
                    c2_pow += s * (odd.double() - even) * (wi.double() - F::ONE);
                }

                let r: Ext = rng.random();
                rs_cur.push(r);

                let (c0, c2) = coeffs(&poly, &pows_combined);
                assert_eq!(c0_pow, c0);
                assert_eq!(c2_pow, c2);
                poly.fix_lo_var_mut(r);
                pows_combined.fix_lo_var_mut(r);

                ws.iter_mut()
                    .zip(ts.iter_mut())
                    .zip(scales.iter_mut())
                    .zip(indexes.iter())
                    .for_each(|(((wi, ti), si), &index)| {
                        *si *= Ext::ONE + r * (-Ext::ONE + *wi);
                        *wi = wi.square();
                        *ti = ti.square();
                        if (index >> (k_data - round - 2)) & 1 == 1 {
                            *ti = -*ti;
                        }
                    });
            });
        }
    }
}
