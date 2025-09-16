use crate::poly::Point;
use crate::utils::TwoAdicSlice;
use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

#[tracing::instrument(skip_all, fields(points = points.len(), k = out.0.k()))]
pub(super) fn compress_eqs_ext<F: Field>(
    out: (&mut [F], bool),
    points: &[Point<F>],
    alpha: F,
    shift: F,
) {
    // TODO: explore SIMD for eq building
    if points.is_empty() {
        return;
    }

    let k = points.first().unwrap().len();
    points.iter().for_each(|point| assert_eq!(point.len(), k));
    let n = points.len();

    let mut eq = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);

    // prepare rlc coeffs
    let alphas = alpha
        .powers()
        .take(n)
        .map(|alpha| alpha * shift)
        .collect::<Vec<_>>();

    // initialize first row with randomness factors
    eq.iter_mut()
        .zip(alphas.iter())
        .for_each(|(e, alpha)| *e = *alpha);

    // build eqs independently
    for i in 0..k {
        let (lo, hi) = eq.split_at_mut((1 << i) * n);
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
            });
    }

    // accumulate rows, add if initialized, assign otherwises
    let initialized = out.1;
    if initialized {
        eq.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, eq)| *eq += row.iter().fold(F::ZERO, |acc, &v| acc + v));
    } else {
        eq.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, eq)| *eq = row.iter().fold(F::ZERO, |acc, &v| acc + v));
    }
}

#[tracing::instrument(skip_all, fields(points = points.len(), k = out.0.k()))]
pub(super) fn compress_eqs_base<F: Field, Ext: ExtensionField<F>>(
    out: (&mut [Ext], bool),
    points: &[Point<F>],
    alpha: Ext,
    shift: Ext,
) {
    // TODO: explore SIMD for eq building
    if points.is_empty() {
        return;
    }
    let k = points.first().unwrap().len();

    // validate num variables
    points.iter().for_each(|point| assert_eq!(point.len(), k));
    let n = points.len();

    // initialize an empty matrix
    let mut eq: Vec<F> = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);

    // prepare rlc coeffs
    let alphas = alpha
        .powers()
        .take(n)
        .map(|alpha| alpha * shift)
        .collect::<Vec<_>>();

    // initialize first row with ones, will apply randomness later
    eq.iter_mut().take(n).for_each(|e| *e = F::ONE);

    // build eqs independently
    for i in 0..k {
        let (lo, hi) = eq.split_at_mut((1 << i) * n);
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
            });
    }

    // accumulate rows, add if initialized, assign otherwises
    let initialized = out.1;
    if initialized {
        eq.par_chunks(n)
            .zip(out.0.par_iter_mut())
            .for_each(|(row, out)| {
                *out += row
                    .iter()
                    .zip(alphas.iter())
                    .fold(Ext::ZERO, |acc, (&v, &alpha)| acc + alpha * v)
            });
    } else {
        eq.par_chunks(n)
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
    use super::{compress_eqs_base, compress_eqs_ext};
    use crate::{
        poly::{Eval, Point, Poly},
        utils::{unsafe_allocate_zero_vec, VecOps},
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};

    type F = BabyBear;
    type Ext = BinomialExtensionField<F, 4>;
    use p3_field::PrimeCharacteristicRing;
    use rayon::{
        iter::{
            IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
            ParallelIterator,
        },
        slice::{ParallelSlice, ParallelSliceMut},
    };

    fn compress_eqs_naive<F: Field, Ext: ExtensionField<F>>(
        k: usize,
        n_base: usize,
        alpha: Ext,
        shift: Ext,
        points_base: &[Point<F>],
        points_ext: &[Point<Ext>],
    ) -> Poly<Ext, Eval> {
        let eqs_base = points_base
            .iter()
            .map(|point| point.eq(F::ONE))
            .collect::<Vec<_>>();

        let mut compressed_eqs = (0..1 << k)
            .map(|i| {
                eqs_base
                    .iter()
                    .map(|eq| &eq[i])
                    .horner_shifted(alpha, shift)
            })
            .collect::<Vec<_>>();

        let eqs_ext = points_ext
            .iter()
            .map(|point| point.eq(shift))
            .collect::<Vec<_>>();

        let alpha_shift = alpha.exp_u64(n_base as u64);
        compressed_eqs.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += eqs_ext
                .iter()
                .map(|eq| &eq[i])
                .horner_shifted(alpha, alpha_shift)
        });
        compressed_eqs.into()
    }

    fn compress_eqs_alternative1<F: Field, Ext: ExtensionField<F>>(
        out: (&mut [Ext], bool),
        alpha: Ext,
        shift: Ext,
        points_base: &[Point<F>],
        points_ext: &[Point<Ext>],
    ) {
        let k = if !points_base.is_empty() {
            points_base.first().unwrap().len()
        } else if !points_ext.is_empty() {
            points_ext.first().unwrap().len()
        } else {
            panic!()
        };

        // validate sizes
        points_base
            .iter()
            .for_each(|point| assert_eq!(point.len(), k));
        points_ext
            .iter()
            .for_each(|point| assert_eq!(point.len(), k));

        let n_base = points_base.len();
        let n = n_base + points_ext.len();

        // initialize an empty matrix
        let mut eq = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);

        // prepare rlc coeffs
        let alphas = alpha
            .powers()
            .take(n)
            .map(|alpha| alpha * shift)
            .collect::<Vec<_>>();

        // initialize first row with randomnesses
        eq.iter_mut()
            .zip(alphas.iter())
            .for_each(|(e, alpha)| *e = *alpha);

        // build eqs independently
        for i in 0..k {
            let (lo, hi) = eq.split_at_mut((1 << i) * n);
            lo.par_chunks_mut(n)
                .zip(hi.par_chunks_mut(n))
                .for_each(|(lo, hi)| {
                    points_base
                        .iter()
                        .zip(lo.iter_mut().zip(hi.iter_mut()))
                        .for_each(|(point, (a0, a1))| {
                            *a1 = *a0 * point[i];
                            *a0 -= *a1;
                        });

                    points_ext
                        .iter()
                        .zip(lo.iter_mut().zip(hi.iter_mut()).skip(n_base))
                        .for_each(|(point, (a0, a1))| {
                            *a1 = *a0 * point[i];
                            *a0 -= *a1;
                        });
                });
        }

        // accumulate rows
        if out.1 {
            // initilized
            eq.par_chunks(n)
                .zip(out.0.par_iter_mut())
                .for_each(|(row, eq)| *eq += row.iter().fold(Ext::ZERO, |acc, &v| acc + v));
        } else {
            // not initilized
            eq.par_chunks(n)
                .zip(out.0.par_iter_mut())
                .for_each(|(row, eq)| *eq = row.iter().fold(Ext::ZERO, |acc, &v| acc + v));
        }
    }

    fn compress_eq_alternative2<F: Field, Ext: ExtensionField<F>>(
        out: (&mut [Ext], bool),
        n: usize,
        alpha: Ext,
        shift: Ext,
        points_base: &[Point<F>],
        points_ext: &[Point<Ext>],
    ) {
        let alphas = alpha.powers().take(n).collect();

        let eqs_base = points_base
            .iter()
            .map(|point| point.eq(F::ONE))
            .collect::<Vec<_>>();

        let mut initialized = out.1;
        for (eq, alpha) in eqs_base.iter().zip(alphas.iter()) {
            let shift = *alpha * shift;
            if initialized {
                out.0
                    .par_iter_mut()
                    .zip(eq.par_iter())
                    .for_each(|(e, &v)| *e += shift * v);
            } else {
                out.0
                    .par_iter_mut()
                    .zip(eq.par_iter())
                    .for_each(|(e, &v)| *e = shift * v);
                initialized = true;
            }
        }

        let eqs_ext = points_ext
            .iter()
            .zip(alphas.iter().skip(eqs_base.len()))
            .map(|(point, alpha)| point.eq(*alpha * shift))
            .collect::<Vec<_>>();

        for eq in eqs_ext.iter() {
            if initialized {
                out.0
                    .par_iter_mut()
                    .zip(eq.par_iter())
                    .for_each(|(e, &v)| *e += v);
            } else {
                out.0
                    .par_iter_mut()
                    .zip(eq.par_iter())
                    .for_each(|(e, &v)| *e = v);
            }
        }
    }

    #[test]
    fn test_compressed_eq() {
        let k = 16;
        let n_base = 20;
        let n_ext = 11;
        let n = n_base + n_ext;
        let mut rng = crate::test::rng(1);
        let alpha: Ext = Ext::ONE;
        let shift: Ext = Ext::ONE;

        // crate::test::init_tracing();
        let points_base = (0..n_base)
            .map(|_| Point::<F>::rand(&mut rng, k))
            .collect::<Vec<_>>();
        let points_ext = (0..n_ext)
            .map(|_| Point::<Ext>::rand(&mut rng, k))
            .collect::<Vec<_>>();

        let eq0 = tracing::info_span!("eq0")
            .in_scope(|| compress_eqs_naive(k, n_base, alpha, shift, &points_base, &points_ext));

        let mut eq1: Poly<Ext, Eval> = unsafe_allocate_zero_vec(1 << k).into();
        tracing::info_span!("eq1").in_scope(|| {
            compress_eq_alternative2(
                (&mut eq1, false),
                n,
                alpha,
                shift,
                &points_base,
                &points_ext,
            )
        });
        assert_eq!(eq0, eq1);

        let mut eq2: Poly<Ext, Eval> = unsafe_allocate_zero_vec(1 << k).into();
        tracing::info_span!("eq2").in_scope(|| {
            compress_eqs_alternative1((&mut eq2, false), alpha, shift, &points_base, &points_ext)
        });
        assert_eq!(eq0, eq2);

        let mut eq3: Poly<Ext, Eval> = unsafe_allocate_zero_vec(1 << k).into();
        tracing::info_span!("eq3").in_scope(|| {
            compress_eqs_base((&mut eq3, false), &points_base, alpha, shift);
            compress_eqs_ext(
                (&mut eq3, true),
                &points_ext,
                alpha,
                shift * alpha.exp_u64(points_base.len() as u64),
            )
        });
        assert_eq!(eq0, eq3);
    }
}
