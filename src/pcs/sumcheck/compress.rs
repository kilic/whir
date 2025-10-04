use crate::pcs::PowClaim;
use crate::utils::{TwoAdicSlice, VecOps};
use crate::{pcs::EqClaim, poly::Point};
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

pub fn compress_claims<F: Field, Ext: ExtensionField<F>>(
    out: (&mut [Ext], &mut Ext, bool),

    alpha: Ext,

    eq_claims_base: &[EqClaim<F, Ext>],
    eq_claims_ext: &[EqClaim<Ext, Ext>],

    pow_claims_base: &[PowClaim<F, Ext>],
    pow_claims_ext: &[PowClaim<Ext, Ext>],
) {
    use crate::utils::TwoAdicSlice;
    let weights = out.0;
    let sum = out.1;
    let initialized = out.2;
    let k = weights.k();

    eq_claims_base.iter().for_each(|o| assert_eq!(o.k(), k));
    eq_claims_ext.iter().for_each(|o| assert_eq!(o.k(), k));
    pow_claims_base.iter().for_each(|o| assert_eq!(o.k, k));
    pow_claims_ext.iter().for_each(|o| assert_eq!(o.k, k));

    *sum = std::iter::once(&sum.clone())
        .chain(eq_claims_base.iter().map(EqClaim::<F, Ext>::eval))
        .chain(pow_claims_base.iter().map(PowClaim::<F, Ext>::eval))
        .chain(eq_claims_ext.iter().map(EqClaim::<Ext, Ext>::eval))
        .chain(pow_claims_ext.iter().map(PowClaim::<Ext, Ext>::eval))
        .horner(alpha);

    let points_base = eq_claims_base
        .iter()
        .map(EqClaim::point)
        .collect::<Vec<_>>();

    let points_ext = eq_claims_ext.iter().map(EqClaim::point).collect::<Vec<_>>();

    let vars_base = pow_claims_base
        .iter()
        .map(PowClaim::var)
        .collect::<Vec<_>>();

    let vars_ext = pow_claims_ext.iter().map(PowClaim::var).collect::<Vec<_>>();

    compress_mixed(
        (weights, initialized),
        k,
        &points_base,
        &vars_base,
        alpha,
        alpha,
    );

    let off = points_base.len() + vars_base.len() + 1;
    compress(
        (weights, true),
        k,
        &points_ext,
        &vars_ext,
        alpha,
        alpha.exp_u64(off as u64),
    );
}

#[tracing::instrument(skip_all, fields(points = points.len(), k = out.0.k()))]
pub(super) fn compress<F: Field>(
    out: (&mut [F], bool),
    k: usize,
    points: &[Point<F>],
    vars: &[F],
    alpha: F,
    shift: F,
) {
    if points.is_empty() && vars.is_empty() {
        return;
    }

    points.iter().for_each(|point| assert_eq!(point.len(), k));
    let n_points = points.len();
    let n_vars = vars.len();
    let n = n_points + n_vars;

    let var_combos = vars
        .par_iter()
        .cloned()
        .map(|mut var| {
            (0..k)
                .map(|_| {
                    let ret = var;
                    var = var.square();
                    ret
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut acc = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);

    let alphas = alpha
        .powers()
        .take(n)
        .map(|alpha| alpha * shift)
        .collect::<Vec<_>>();

    acc[..n].copy_from_slice(&alphas);

    for i in 0..k {
        let (lo, hi) = acc.split_at_mut((1 << i) * n);
        let points = points.to_vec();
        let combos = var_combos.iter().map(|c| c[i]).collect::<Vec<_>>();

        lo.par_chunks_mut(n)
            .zip(hi.par_chunks_mut(n))
            .for_each(|(lo, hi)| {
                points
                    .iter()
                    .zip(lo.iter_mut())
                    .zip(hi.iter_mut())
                    .for_each(|((point, a0), a1)| {
                        *a1 = *a0 * point[i];
                        *a0 -= *a1;
                    });
                combos
                    .iter()
                    .zip(lo.iter().zip(hi.iter_mut()).skip(n_points))
                    .for_each(|(&combo, (&lo, hi))| *hi = lo * combo);
            });
    }

    // accumulate rows, add if initialized, assign otherwises
    let initialized = out.1;
    if initialized {
        acc.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, eq)| *eq += row.iter().fold(F::ZERO, |acc, &v| acc + v));
    } else {
        acc.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, eq)| *eq = row.iter().fold(F::ZERO, |acc, &v| acc + v));
    }
}

#[tracing::instrument(skip_all, fields(points = points.len(), k = out.0.k()))]
pub(super) fn compress_mixed<F: Field, Ext: ExtensionField<F>>(
    out: (&mut [Ext], bool),
    k: usize,
    points: &[Point<F>],
    vars: &[F],
    alpha: Ext,
    shift: Ext,
) {
    if points.is_empty() && vars.is_empty() {
        return;
    }

    points.iter().for_each(|point| assert_eq!(point.len(), k));
    let n_points = points.len();
    let n_vars = vars.len();
    let n = n_points + n_vars;

    let var_combos = vars
        .par_iter()
        .cloned()
        .map(|mut var| {
            (0..k)
                .map(|_| {
                    let ret = var;
                    var = var.square();
                    ret
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut acc: Vec<F> = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);

    acc[..n].copy_from_slice(&vec![F::ONE; n]);

    for i in 0..k {
        let (lo, hi) = acc.split_at_mut((1 << i) * n);
        let points = points.to_vec();
        let combos = var_combos.iter().map(|c| c[i]).collect::<Vec<_>>();

        lo.par_chunks_mut(n)
            .zip(hi.par_chunks_mut(n))
            .for_each(|(lo, hi)| {
                points
                    .iter()
                    .zip(lo.iter_mut().zip(hi.iter_mut()))
                    .for_each(|(point, (a0, a1))| {
                        *a1 = *a0 * point[i];
                        *a0 -= *a1;
                    });
                combos
                    .iter()
                    .zip(lo.iter().zip(hi.iter_mut()).skip(n_points))
                    .for_each(|(&combo, (&lo, hi))| *hi = lo * combo);
            });
    }

    let alphas = alpha
        .powers()
        .take(n)
        .map(|alpha| alpha * shift)
        .collect::<Vec<_>>();

    // accumulate rows, add if initialized, assign otherwises
    let initialized = out.1;
    if initialized {
        acc.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, out)| {
                *out += row
                    .iter()
                    .zip(alphas.iter())
                    .fold(Ext::ZERO, |acc, (&v, &alpha)| acc + alpha * v)
            });
    } else {
        acc.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, out)| {
                *out = row
                    .iter()
                    .zip(alphas.iter())
                    .fold(Ext::ZERO, |acc, (&v, &alpha)| acc + alpha * v)
            });
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            sumcheck::compress::{compress, compress_claims, compress_mixed},
            test::{
                make_eq_claims_base, make_eq_claims_ext, make_pow_claims_base, make_pow_claims_ext,
            },
            EqClaim,
        },
        poly::{Eval, Point, Poly},
        transcript::test_transcript::TestWriter,
        utils::VecOps,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};

    type F = BabyBear;
    type Ext = BinomialExtensionField<F, 4>;
    use p3_field::PrimeCharacteristicRing;
    use rand::Rng;

    fn compress_naive<F: Field, Ext: ExtensionField<F>>(
        k: usize,
        alpha: Ext,
        shift: Ext,
        points_base: &[Point<F>],
        points_ext: &[Point<Ext>],
        vars_base: &[F],
        vars_ext: &[Ext],
    ) -> Poly<Ext, Eval> {
        let eqs_base = points_base
            .iter()
            .map(|point| point.eq(F::ONE))
            .collect::<Vec<_>>();

        let pows_base = vars_base
            .iter()
            .map(|&x| x.powers().take(1 << k).collect())
            .collect::<Vec<_>>();

        let eqs_ext = points_ext
            .iter()
            .map(|point| point.eq(Ext::ONE))
            .collect::<Vec<_>>();

        let pows_ext = vars_ext
            .iter()
            .map(|&x| x.powers().take(1 << k).collect())
            .collect::<Vec<_>>();

        let mut acc = vec![Ext::ZERO; 1 << k];

        let n = 0;
        let alpha_shift = alpha.exp_u64(n as u64) * shift;
        acc.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += eqs_base
                .iter()
                .map(|eq| &eq[i])
                .horner_shifted(alpha, alpha_shift)
        });

        let n = n + eqs_base.len();
        let alpha_shift = alpha.exp_u64(n as u64) * shift;
        acc.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += pows_base
                .iter()
                .map(|eq| &eq[i])
                .horner_shifted(alpha, alpha_shift)
        });

        let n = n + pows_base.len();
        let alpha_shift = alpha.exp_u64(n as u64) * shift;
        acc.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += eqs_ext
                .iter()
                .map(|eq| &eq[i])
                .horner_shifted(alpha, alpha_shift)
        });

        let n = n + eqs_ext.len();
        let alpha_shift = alpha.exp_u64(n as u64) * shift;
        acc.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += pows_ext
                .iter()
                .map(|eq| &eq[i])
                .horner_shifted(alpha, alpha_shift)
        });

        acc.into()
    }

    #[test]
    fn test_compress() {
        let k = 4;
        let mut rng = crate::test::rng(1);
        let poly = Poly::<Ext, Eval>::rand(&mut rng, k);

        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        for _ in 0..1000 {
            let alpha: Ext = rng.random();

            let n_eq_base = rng.random_range(0..2);
            let n_pow_base = rng.random_range(0..2);
            let n_eq_ext = rng.random_range(0..2);
            let n_pow_ext = rng.random_range(0..2);

            let eq_claims_base: Vec<EqClaim<F, Ext>> =
                make_eq_claims_base(&mut transcript, n_eq_base, &poly).unwrap();
            let pow_claims_base = make_pow_claims_base(&mut transcript, n_pow_base, &poly).unwrap();
            let eq_claims_ext =
                make_eq_claims_ext::<_, F, Ext, Ext>(&mut transcript, n_eq_ext, &poly).unwrap();
            let pow_claims_ext =
                make_pow_claims_ext::<_, F, Ext, Ext>(&mut transcript, n_pow_ext, &poly).unwrap();

            let points_base = eq_claims_base
                .iter()
                .map(|c| c.point().clone())
                .collect::<Vec<_>>();
            let points_ext = eq_claims_ext
                .iter()
                .map(|c| c.point().clone())
                .collect::<Vec<_>>();
            let vars_base = pow_claims_base.iter().map(|c| c.var()).collect::<Vec<_>>();
            let vars_ext = pow_claims_ext.iter().map(|c| c.var()).collect::<Vec<_>>();

            let acc0 = compress_naive(
                k,
                alpha,
                alpha,
                &points_base,
                &points_ext,
                &vars_base,
                &vars_ext,
            );

            let mut acc1 = Poly::<Ext, Eval>::zero(k);
            compress_mixed((&mut acc1, true), k, &points_base, &vars_base, alpha, alpha);
            compress(
                (&mut acc1, true),
                k,
                &points_ext,
                &vars_ext,
                alpha,
                alpha * alpha.exp_u64((vars_base.len() + points_base.len()) as u64),
            );
            assert_eq!(acc0, acc1);

            let mut acc2 = Poly::<Ext, Eval>::zero(k);
            let mut sum = Ext::ZERO;
            compress_claims::<F, Ext>(
                (&mut acc2, &mut sum, true),
                alpha,
                &eq_claims_base,
                &eq_claims_ext,
                &pow_claims_base,
                &pow_claims_ext,
            );
            assert_eq!(acc0, acc2);
        }
    }
}
