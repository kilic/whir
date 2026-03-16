use crate::sumcheck::lagrange_weights_012;
use common::field::*;
use itertools::Itertools;
use p3_util::log2_strict_usize;
use poly::{Point, Poly, eq::SplitEq, eval_eq_xy, eval_poly_reference};

pub(super) fn lagrange_weights_012_multi<F: Field>(rs: &[F]) -> Vec<F> {
    let mut weights = vec![F::ONE];
    rs.iter().for_each(|&r| {
        weights = lagrange_weights_012(r)
            .iter()
            .flat_map(|li| weights.iter().map(|&w| w * *li))
            .collect();
    });
    weights
}

fn svo_partial_evals<F: Field, Ext: ExtensionField<F>>(
    l: usize,
    compressed: &[Ext],
    svo: &Point<Ext>,
) -> Poly<Ext> {
    assert_eq!(log2_strict_usize(compressed.len()), svo.len());
    let (svo_active, svo_rest) = svo.split_at(l);
    let eq_active = svo_active.eq(Ext::ONE);
    let eq_rest = svo_rest.eq(Ext::ONE);

    let mut out = Poly::<Ext>::zero(eq_active.k());
    compressed
        .chunks(eq_active.len())
        .zip(eq_rest.iter())
        .for_each(|(chunk, &w)| {
            out.iter_mut()
                .zip_eq(chunk.iter())
                .for_each(|(out, &f)| *out += f * w);
        });
    out
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SvoAccumulators<Ext: Field>(Vec<[Vec<Ext>; 2]>);

impl<Ext: Field> core::ops::Deref for SvoAccumulators<Ext> {
    type Target = Vec<[Vec<Ext>; 2]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Ext: Field> core::ops::DerefMut for SvoAccumulators<Ext> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SvoPoint<F: Field, Ext: ExtensionField<F>> {
    svo: Point<Ext>,
    split: SplitEq<F, Ext>,
}

impl<F: Field, Ext: ExtensionField<F>> SvoPoint<F, Ext> {
    pub(crate) fn new(k_svo: usize, point: &Point<Ext>) -> Self {
        assert!(k_svo <= point.k());
        let (svo, z_split) = point.split_at(k_svo);
        let split = SplitEq::new_unpacked(&z_split);
        SvoPoint { svo, split }
    }

    pub(crate) fn combine_into(&self, out: &mut [Ext], rs: &Point<Ext>, scale: Ext) {
        assert_eq!(rs.k(), self.k_svo());
        self.split
            .combine_into(out, Some(scale * eval_eq_xy(&self.svo, rs)))
    }

    pub(crate) fn eval(self, poly: &Poly<F>) -> SvoClaim<F, Ext> {
        assert_eq!(self.k(), poly.k());

        let compressed = self.split.compress_hi(poly);
        let eval = eval_poly_reference(&compressed, &self.svo);
        let partial_evals = (1..=self.k_svo())
            .map(|i| svo_partial_evals(i, &compressed, &self.svo))
            .collect::<Vec<_>>();

        SvoClaim {
            point: self,
            partial_evals,
            eval,
        }
    }

    pub(crate) fn k_svo(&self) -> usize {
        self.svo.k()
    }

    pub(crate) fn k_split(&self) -> usize {
        self.split.k()
    }

    pub(crate) fn k(&self) -> usize {
        self.svo.k() + self.split.k()
    }

    pub(crate) fn svo(&self) -> &Point<Ext> {
        &self.svo
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SvoClaim<F: Field, Ext: ExtensionField<F>> {
    point: SvoPoint<F, Ext>,
    partial_evals: Vec<Poly<Ext>>,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> SvoClaim<F, Ext> {
    pub(crate) fn new(point: SvoPoint<F, Ext>, poly: &Poly<F>) -> Self {
        point.eval(poly)
    }

    pub(crate) fn eval(&self) -> &Ext {
        &self.eval
    }

    pub(crate) fn point(&self) -> &SvoPoint<F, Ext> {
        &self.point
    }

    fn partial_evals(&self) -> &[Poly<Ext>] {
        &self.partial_evals
    }

    fn k_svo(&self) -> usize {
        self.point.k_svo()
    }

    fn svo(&self) -> &Point<Ext> {
        self.point.svo()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SvoCoeffs<F: Field> {
    eq0: Vec<Poly<F>>,
    eq2: Vec<Poly<F>>,
    points0: Vec<Vec<F>>,
    points2: Vec<Vec<F>>,
}

#[derive(Debug, Clone)]
pub(crate) struct Svo<F: Field> {
    coeffs: Vec<SvoCoeffs<F>>,
}

impl<F: Field> Svo<F> {
    #[tracing::instrument(skip_all, name = "Svo::new")]
    pub(crate) fn new(l: usize) -> Self {
        fn extend_eq<F: Field>(prev: &Poly<F>, zi: F) -> Poly<F> {
            let mut eq = F::zero_vec(prev.len() * 2);
            let (lo, hi) = eq.split_at_mut(prev.len());
            lo.iter_mut()
                .zip(hi.iter_mut())
                .zip(prev.iter())
                .for_each(|((lo, hi), &prev)| {
                    *lo = prev * (F::ONE - zi);
                    *hi = prev * zi;
                });
            eq.into()
        }

        // This contains eq0/eq1/eq2 points; we only persist eq0/eq2 per layer.
        let mut prev_eqs = vec![Poly::new(vec![F::ONE])];
        let mut prev_points = vec![vec![]];

        let coeffs = (0..l)
            .map(|_| {
                let prev_len = prev_eqs.len();
                let mut next_eqs = Vec::with_capacity(prev_len * 3);
                let mut next_points = Vec::with_capacity(prev_len * 3);

                for i in 0..prev_len {
                    let eq = extend_eq(&prev_eqs[i], F::ZERO);
                    let mut point = prev_points[i].clone();
                    point.push(F::ZERO);

                    next_eqs.push(eq);
                    next_points.push(point);
                }

                for i in 0..prev_len {
                    let eq = extend_eq(&prev_eqs[i], F::ONE);
                    let mut point = prev_points[i].clone();
                    point.push(F::ONE);

                    next_eqs.push(eq);
                    next_points.push(point);
                }

                for i in 0..prev_len {
                    let eq = extend_eq(&prev_eqs[i], F::TWO);
                    let mut point = prev_points[i].clone();
                    point.push(F::TWO);

                    next_eqs.push(eq);
                    next_points.push(point);
                }

                let coeff = SvoCoeffs {
                    eq0: next_eqs[0..prev_len].to_vec(),
                    eq2: next_eqs[2 * prev_len..3 * prev_len].to_vec(),
                    points0: next_points[0..prev_len].to_vec(),
                    points2: next_points[2 * prev_len..3 * prev_len].to_vec(),
                };
                prev_eqs = next_eqs;
                prev_points = next_points;
                coeff
            })
            .collect();

        Self { coeffs }
    }

    #[tracing::instrument(skip_all, name = "Svo::calculate_accumulators")]
    pub(crate) fn calculate_accumulators<Ext: ExtensionField<F>>(
        &self,
        claims: &[SvoClaim<F, Ext>],
        alphas: &[Ext],
    ) -> SvoAccumulators<Ext> {
        assert_eq!(claims.len(), alphas.len());
        let k = claims
            .iter()
            .map(SvoClaim::k_svo)
            .all_equal_value()
            .unwrap();
        let svo = claims.iter().map(SvoClaim::svo).all_equal_value().unwrap();

        let out = (0..k)
            .map(|round_idx| {
                let l = round_idx + 1;
                let mut combined = Poly::<Ext>::zero(l);
                claims
                    .iter()
                    .zip(alphas.iter())
                    .for_each(|(claim, &alpha)| {
                        combined
                            .iter_mut()
                            .zip_eq(claim.partial_evals()[round_idx].iter())
                            .for_each(|(out, &f)| *out += alpha * f);
                    });
                let svo_active = svo.range(..l);

                let acc0 = self.coeffs[round_idx]
                    .points0
                    .iter()
                    .zip(self.coeffs[round_idx].eq0.iter())
                    .map(|(u, eq)| {
                        eval_eq_xy(u, &svo_active)
                            * dot_product::<Ext, _, _>(combined.iter().copied(), eq.iter().copied())
                    })
                    .collect::<Vec<_>>();

                let acc2 = self.coeffs[round_idx]
                    .points2
                    .iter()
                    .zip(self.coeffs[round_idx].eq2.iter())
                    .map(|(u, eq)| {
                        eval_eq_xy(u, &svo_active)
                            * dot_product::<Ext, _, _>(combined.iter().copied(), eq.iter().copied())
                    })
                    .collect::<Vec<_>>();

                [acc0, acc2]
            })
            .collect();

        SvoAccumulators(out)
    }
}

#[cfg(test)]
mod test {
    use super::{Svo, SvoPoint};
    use common::field::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use p3_koala_bear::KoalaBear;
    use poly::{Point, Poly, eval_poly_reference};

    #[test]
    fn test_accumulators() {
        type F = KoalaBear;
        type Ext = F;
        let k = 12;
        let mut rng = common::test::rng(1);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<Ext>::rand(&mut rng, k);
        let eq: Poly<Ext> = point.eq(Ext::ONE);
        let e0 = poly.eval_base(&point);

        for l0 in 0..k / 2 {
            let point = SvoPoint::<F, Ext>::new(l0, &point);
            let claim = point.eval(&poly);
            assert_eq!(e0, claim.eval);
            let svo = Svo::<F>::new(l0);
            let accumulators = svo.calculate_accumulators(&[claim], &[Ext::ONE]);

            for (i, accumulator) in accumulators.iter().enumerate() {
                for (us, accs) in [
                    (svo.coeffs[i].points0.iter(), accumulator[0].iter()),
                    (svo.coeffs[i].points2.iter(), accumulator[1].iter()),
                ] {
                    us.zip(accs).for_each(|(u, &acc)| {
                        let u = Point::new(u.clone());
                        let poly = poly.compress_lo(&u, Ext::ONE);
                        let eq = eq.compress_lo(&u, Ext::ONE);
                        assert_eq!(acc, dot_product(poly.iter().copied(), eq.iter().copied()));
                    });
                }
            }
        }
    }

    #[test]
    fn test_svo_eval() {
        type F = KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        let k = 20;
        let k_svo = 7;
        let mut rng = common::test::rng(1);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<Ext>::rand(&mut rng, k);

        let e0 = eval_poly_reference(&poly, &point);

        let n_iter = 100;
        common::test::bench(
            "eval base",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| poly.eval_base(&point),
            |_, e1| assert_eq!(e0, e1),
        );

        common::test::bench(
            "ref",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| eval_poly_reference(&poly, &point),
            |_, e1| assert_eq!(e0, e1),
        );

        common::test::bench(
            "svo",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| SvoPoint::<F, Ext>::new(k_svo, &point),
            |_, point| assert_eq!(e0, *point.eval(&poly).eval()),
        );

        common::test::bench(
            "svo & eval",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| {
                let point = SvoPoint::<F, Ext>::new(k_svo, &point);
                *point.eval(&poly).eval()
            },
            |_, e1| assert_eq!(e0, e1),
        );
    }
}
