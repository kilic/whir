use crate::{
    poly::{Point, Poly},
    utils::{log2_strict, TwoAdicSlice},
};
use itertools::Itertools;
use p3_field::{dot_product, Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::Matrix;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use rayon::prelude::*;

#[tracing::instrument(skip_all)]
fn flat_eqs<F: Field, Ext: ExtensionField<F>>(points: &[Point<Ext>], alpha: Ext) -> Vec<Ext> {
    let k = points[0].len();
    let n = points.len();

    let alphas = alpha.powers().take(n).collect();
    let mut acc: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);
    acc[..n].copy_from_slice(&alphas);
    for i in 0..k {
        let (lo, hi) = acc.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i]).collect::<Vec<_>>();
        lo.chunks_mut(n).zip(hi.chunks_mut(n)).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
        });
    }
    acc
}

#[tracing::instrument(skip_all)]
fn packed_flat_eqs<F: Field, Ext: ExtensionField<F>>(
    points: &[Point<Ext>],
    alpha: Ext,
) -> Vec<Ext::ExtensionPacking> {
    let k = points[0].len();
    let k_pack = log2_strict(F::Packing::WIDTH);
    let n = points.len();

    let alphas = alpha.powers().take(n).collect();
    let mut acc_packed: Vec<Ext::ExtensionPacking> =
        crate::utils::unsafe_allocate_zero_vec((1 << (k - k_pack)) * n);
    let mut acc_init: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec((1 << k_pack) * n);
    acc_init[..n].copy_from_slice(&alphas);

    for i in 0..k_pack {
        let (lo, hi) = acc_init.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i]).collect::<Vec<_>>();

        lo.chunks_mut(n).zip(hi.chunks_mut(n)).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
        });
    }

    let mut acc_init_transposed = Ext::zero_vec(acc_init.len());
    transpose::transpose(
        &acc_init,
        &mut acc_init_transposed,
        acc_init.len() / F::Packing::WIDTH,
        F::Packing::WIDTH,
    );
    acc_init_transposed
        .chunks(F::Packing::WIDTH)
        .zip(acc_packed.iter_mut())
        .for_each(|(chunk, packed)| *packed = Ext::ExtensionPacking::from_ext_slice(chunk));

    for i in 0..k - k_pack {
        let (lo, hi) = acc_packed.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i + k_pack]).collect::<Vec<_>>();
        lo.chunks_mut(n).zip(hi.chunks_mut(n)).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
        });
    }

    acc_packed
}

#[tracing::instrument(skip_all)]
pub fn eq_compress_split<F: Field, Ext: ExtensionField<F>, const PAR: bool>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
    split_at: usize,
) {
    let n = points.len();
    let z0s = points
        .iter()
        .map(|point| point.split_at(split_at).0)
        .collect::<Vec<_>>();
    let left = flat_eqs(&z0s, alpha);
    let z1s = points
        .iter()
        .map(|point| point.split_at(split_at).1)
        .collect::<Vec<_>>();
    let right = flat_eqs(&z1s, Ext::ONE);

    if PAR {
        out.par_chunks_mut(left.len() / n)
            .zip_eq(right.par_chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut()
                    .zip_eq(left.chunks(n))
                    .for_each(|(out, left)| {
                        *out =
                            dot_product::<Ext, _, _>(left.iter().cloned(), right.iter().cloned());
                    });
            });
    } else {
        out.chunks_mut(left.len() / n)
            .zip_eq(right.chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut()
                    .zip_eq(left.chunks(n))
                    .for_each(|(out, left)| {
                        *out =
                            dot_product::<Ext, _, _>(left.iter().cloned(), right.iter().cloned());
                    });
            });
    }
}

#[tracing::instrument(skip_all)]
pub fn eq_compress_split_packed<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
    split_at: usize,
) {
    use p3_field::PackedFieldExtension;

    let n = points.len();
    let z0s = points
        .iter()
        .map(|point| point.split_at(split_at).0)
        .collect::<Vec<_>>();
    let left_packed = packed_flat_eqs::<F, Ext>(&z0s, alpha);
    let z1s = points
        .iter()
        .map(|point| point.split_at(split_at).1)
        .collect::<Vec<_>>();
    let right = flat_eqs(&z1s, Ext::ONE);

    out.par_chunks_mut(F::Packing::WIDTH * left_packed.len() / n)
        .zip_eq(right.par_chunks(n))
        .for_each(|(out, right)| {
            let packed_out = left_packed.chunks(n).map(|left| {
                left.iter()
                    .zip_eq(right.iter())
                    .map(|(&left, &right)| left * right)
                    .sum::<Ext::ExtensionPacking>()
            });
            out.iter_mut()
                .zip_eq(Ext::ExtensionPacking::to_ext_iter(packed_out))
                .for_each(|(out, val)| *out = val);
        });
}

#[tracing::instrument(skip_all, fields(points = points.len()))]
pub fn eq_compress_ref<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
) {
    if points.is_empty() {
        return;
    }
    use crate::utils::TwoAdicSlice;
    let k = out.k();
    let n = points.len();

    let alphas = alpha.powers().take(n).collect();
    let mut acc: Vec<Ext> = Ext::zero_vec((1 << k) * n);
    acc[..n].copy_from_slice(&alphas);

    for i in 0..k {
        let (lo, hi) = acc.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|point| point[i]).collect::<Vec<_>>();

        lo.par_chunks_mut(n)
            .zip(hi.par_chunks_mut(n))
            .with_min_len(1 << 14)
            .for_each(|(lo, hi)| {
                vars.iter()
                    .zip(lo.iter_mut().zip(hi.iter_mut()))
                    .for_each(|(&z, (lo, hi))| {
                        *hi = *lo * z;
                        *lo -= *hi;
                    });
            });
    }
    acc.par_chunks(n)
        .zip(out.par_iter_mut())
        .with_min_len(1 << 14)
        .for_each(|(row, out)| *out += row.iter().cloned().sum::<Ext>());
}

#[tracing::instrument(skip_all)]
pub fn eq_compress_p3<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
) {
    let k = out.k();
    let n = points.len();
    let flat_points = points
        .iter()
        .flat_map(|points| points.iter().rev())
        .cloned()
        .collect::<Vec<_>>();
    let flat_points = RowMajorMatrixView::new(&flat_points, k).transpose();
    let alphas = alpha.powers().take(n).collect();
    eval_eq_batch::<F, Ext, false>(flat_points.as_view(), out, &alphas);
}

pub fn eq_compress_recursive_0<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
) {
    let k = out.k();
    let n = points.len();
    let flat_points = points
        .iter()
        .flat_map(|points| points.iter().rev())
        .cloned()
        .collect::<Vec<_>>();
    let flat_points = RowMajorMatrixView::new(&flat_points, k).transpose();
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = Ext::zero_vec(2 * k * n);
    recursive_inner_0::<F, Ext>(out, &mut workspace, &alphas, &flat_points.values);
}

fn recursive_inner_0<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    workspace: &mut [Ext],
    scalars: &[Ext],
    flat_points: &[Ext],
) {
    use crate::utils::TwoAdicSlice;
    let k = out.k();
    let n = scalars.len();
    debug_assert_eq!(k * scalars.len(), flat_points.len());

    match k {
        0 => {
            out[0] += scalars.iter().copied().sum::<Ext>();
        }
        _ => {
            let (zs_cur, zs_rest) = flat_points.split_at(n);
            let (s0_buffer, next_workspace) = workspace.split_at_mut(n);
            let (s1_buffer, next_workspace) = next_workspace.split_at_mut(n);

            for i in 0..n {
                let s = scalars[i];
                let z = zs_cur[i];
                let s1 = s * z;
                s1_buffer[i] = s1;
                s0_buffer[i] = s - s1;
            }

            let (low, high) = out.split_at_mut(out.len() / 2);
            recursive_inner_0::<F, Ext>(low, next_workspace, s0_buffer, zs_rest);
            recursive_inner_0::<F, Ext>(high, next_workspace, s1_buffer, zs_rest);
        }
    }
}

pub fn eq_compress_recursive_1<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
) {
    let k = out.k();
    let n = points.len();
    let flat_points = points
        .iter()
        .flat_map(|points| points.iter().rev())
        .cloned()
        .collect::<Vec<_>>();
    let flat_points = RowMajorMatrixView::new(&flat_points, k).transpose();
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = Ext::zero_vec(2 * k * n);
    recursive_inner_1::<F, Ext>(flat_points.as_view(), &alphas, out, &mut workspace);
}

fn recursive_inner_1<F: Field, Ext: ExtensionField<F>>(
    points: RowMajorMatrixView<'_, Ext>,
    scalars: &[Ext],
    out: &mut [Ext],
    workspace: &mut [Ext],
) {
    let num_vars = points.height();
    let num_points = points.width();
    debug_assert_eq!(out.len(), 1 << num_vars);

    #[inline(always)]
    fn apply<F: Field, Other: Algebra<F> + Copy + Send + Sync>(
        evals: &[Other],
        row: &[Other],
        buf0: &mut [Other],
        buf1: &mut [Other],
    ) {
        debug_assert_eq!(evals.len(), row.len());
        debug_assert_eq!(evals.len(), buf0.len());
        debug_assert_eq!(evals.len(), buf1.len());
        evals
            .iter()
            .zip(row.iter())
            .zip(buf0.iter_mut().zip(buf1.iter_mut()))
            .for_each(|((&sc, &el), (buf0, buf1))| {
                let s1 = sc * el;
                *buf0 = sc - s1;
                *buf1 = s1;
            });
    }

    #[inline(always)]
    fn apply_basic_1<F: Field, Ext: ExtensionField<F>>(
        evals: RowMajorMatrixView<'_, F>,
        scalars: &[Ext],
    ) -> [Ext; 2] {
        debug_assert_eq!(evals.height(), 1);
        debug_assert_eq!(evals.width(), scalars.len());
        let sum: Ext = scalars.iter().cloned().sum();
        let eq_1_sum: Ext = dot_product(scalars.iter().cloned(), evals.values.iter().copied());
        let eq_0_sum = sum - eq_1_sum.clone();
        [eq_0_sum, eq_1_sum]
    }

    #[inline(always)]
    fn apply_basic_2<F: Field, Ext: ExtensionField<F>>(
        evals: RowMajorMatrixView<'_, Ext>,
        scalars: &[Ext],
        workspace: &mut [Ext],
    ) -> [Ext; 4] {
        debug_assert_eq!(evals.height(), 2);
        debug_assert_eq!(evals.width(), scalars.len());

        let (first_row, second_row) = evals.split_rows(1);
        let num_points = evals.width();

        let (eq_0s, remaining) = workspace.split_at_mut(num_points);
        let eq_1s = &mut remaining[..num_points];

        apply::<F, Ext>(scalars, first_row.values, eq_0s, eq_1s);

        let [eq_00, eq_01] = apply_basic_1(second_row, eq_0s);
        let [eq_10, eq_11] = apply_basic_1(second_row, eq_1s);

        [eq_00, eq_01, eq_10, eq_11]
    }

    match num_vars {
        0 => {
            out[0] = scalars.iter().copied().sum();
        }
        1 => {
            out.copy_from_slice(&apply_basic_1(points, scalars));
        }
        2 => {
            let eqs = apply_basic_2::<F, Ext>(points, scalars, workspace);
            out.copy_from_slice(&eqs);
        }
        3 => {
            debug_assert_eq!(points.height(), 3);
            debug_assert_eq!(points.width(), scalars.len());

            let (first_row, remainder) = points.split_rows(1);
            let num_points = points.width();

            let (eq_0s, next_workspace) = workspace.split_at_mut(num_points);
            let (eq_1s, next_workspace) = next_workspace.split_at_mut(num_points);

            apply::<F, Ext>(scalars, first_row.values, eq_0s, eq_1s);

            let (ws0, remaining) = next_workspace.split_at_mut(2 * num_points);
            let ws1 = &mut remaining[..2 * num_points];

            let [eq_000, eq_001, eq_010, eq_011] = apply_basic_2::<F, Ext>(remainder, eq_0s, ws0);
            let [eq_100, eq_101, eq_110, eq_111] = apply_basic_2::<F, Ext>(remainder, eq_1s, ws1);

            out.copy_from_slice(&[
                eq_000, eq_001, eq_010, eq_011, eq_100, eq_101, eq_110, eq_111,
            ]);
        }
        _ => {
            let (low, high) = out.split_at_mut(out.len() / 2);
            let (first_row, remainder) = points.split_rows(1);

            let (s0_buffer, next_workspace) = workspace.split_at_mut(num_points);
            let (s1_buffer, next_workspace) = next_workspace.split_at_mut(num_points);

            apply::<F, Ext>(scalars, first_row.values, s0_buffer, s1_buffer);

            recursive_inner_1(remainder, s0_buffer, low, next_workspace);
            recursive_inner_1(remainder, s1_buffer, high, next_workspace);
        }
    }
}

#[tracing::instrument(skip_all)]
pub fn eq_compress_split_packed_packed<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext::ExtensionPacking],
    points: &[Point<Ext>],
    alpha: Ext,
    split_at: usize,
) {
    let n = points.len();
    let left = points
        .iter()
        .map(|point| point.split_at(split_at).0)
        .collect::<Vec<_>>();
    let left_packed = packed_flat_eqs::<F, Ext>(&left, alpha);
    let right = points
        .iter()
        .map(|point| point.split_at(split_at).1)
        .collect::<Vec<_>>();
    let right = flat_eqs(&right, Ext::ONE);

    out.par_chunks_mut(left_packed.len() / n)
        .zip_eq(right.par_chunks(n))
        .for_each(|(out, right)| {
            out.iter_mut()
                .zip(left_packed.chunks(n))
                .for_each(|(out, left)| {
                    *out = left
                        .iter()
                        .zip_eq(right.iter())
                        .map(|(&left, &right)| left * right)
                        .sum::<Ext::ExtensionPacking>();
                });
        });
}

#[cfg(test)]
mod test {

    fn unpack_ext_packed<F: Field, Ext: ExtensionField<F>>(
        packed: Vec<Ext::ExtensionPacking>,
    ) -> Vec<Ext> {
        Ext::ExtensionPacking::to_ext_iter(packed.into_iter()).collect()
    }

    use super::*;
    use p3_field::{
        extension::{BinomialExtensionField, PackedBinomialExtensionField},
        PrimeCharacteristicRing,
    };
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;

    #[test]
    fn bench_compress_eq() {
        let mut rng = crate::test::rng(1);

        let n_iter = 10;
        // crate::test::init_tracing();

        fn bench_packed<F: Field, Ext: ExtensionField<F>>(
            n_iter: usize,
            k: usize,
            points: &[Point<Ext>],
            alpha: Ext,
        ) -> Vec<Ext> {
            let mut out = Ext::ExtensionPacking::zero_vec(1 << (k - 2));
            eq_compress_split_packed_packed::<F, Ext>(&mut out, &points, alpha, (k - 2) / 2);
            crate::test::bench("packed-only", n_iter, None, || {
                let mut out = Ext::ExtensionPacking::zero_vec(1 << (k - 2));
                eq_compress_split_packed_packed::<F, Ext>(&mut out, &points, alpha, (k - 2) / 2);
            });
            unpack_ext_packed(out)
        }

        for k in 20..=20 {
            println!("--- **** k: {}", k);
            for n in [1, 10, 100] {
                println!(" --- **** n: {}", n);
                let points = (0..n)
                    .map(|_| Point::<Ext>::rand(&mut rng, k))
                    .collect::<Vec<_>>();

                let alpha: Ext = Ext::ONE;

                let mut out_ref = Ext::zero_vec(1 << k);
                eq_compress_ref::<F, Ext>(&mut out_ref, &points, alpha);

                // crate::test::bench("ref", n_iter, Some(&out_ref), || {
                //     let mut out = Ext::zero_vec(1 << k);
                //     eq_compress_ref::<F, Ext>(&mut out, &points, alpha);
                //     out
                // });

                crate::test::bench("p3", n_iter, Some(&out_ref), || {
                    let mut out = Ext::zero_vec(1 << k);
                    eq_compress_p3::<F, Ext>(&mut out, &points, alpha);
                    out
                });

                // crate::test::bench("rec0", n_iter, Some(&out_ref), || {
                //     let mut out = Ext::zero_vec(1 << k);
                //     eq_compress_recursive_0::<F, Ext>(&mut out, &points, alpha);
                //     out
                // });

                // crate::test::bench("rec1", n_iter, Some(&out_ref), || {
                //     let mut out = Ext::zero_vec(1 << k);
                //     eq_compress_recursive_1::<F, Ext>(&mut out, &points, alpha);
                //     out
                // });

                crate::test::bench("splitpar", n_iter, Some(&out_ref), || {
                    let mut out = Ext::zero_vec(1 << k);
                    eq_compress_split::<F, Ext, true>(&mut out, &points, alpha, k / 2);
                    out
                });

                // crate::test::bench("splitser", n_iter, Some(&out_ref), || {
                //     let mut out = Ext::zero_vec(1 << k);
                //     eq_compress_split::<F, Ext, false>(&mut out, &points, alpha, k / 2);
                //     out
                // });

                crate::test::bench("packed", n_iter, Some(&out_ref), || {
                    let mut out = Ext::zero_vec(1 << k);
                    eq_compress_split_packed::<F, Ext>(&mut out, &points, alpha, k / 2);
                    out
                });

                let out_packed = bench_packed::<F, Ext>(n_iter, k, &points, alpha);
                assert_eq!(out_ref, out_packed);

                // let mut out_ref_packed = Ext::zero_vec(1 << (k - 2));
                // eq_compress_split_packed_packed::<F, Ext>(
                //     &mut out_ref_packed,
                //     &points,
                //     alpha,
                //     k / 2,
                // );
            }
        }
    }
}
