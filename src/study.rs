use crate::{
    poly::{Point, Poly},
    utils::log2_strict,
};
use itertools::Itertools;
use p3_field::PackedValue;
use p3_field::{dot_product, PackedFieldExtension};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

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
fn packed_eq<F: Field, Ext: ExtensionField<F>>(point: &Point<Ext>) -> Vec<Ext::ExtensionPacking> {
    let k = point.len();
    let k_pack = log2_strict(F::Packing::WIDTH);

    let mut acc_packed: Vec<Ext::ExtensionPacking> =
        crate::utils::unsafe_allocate_zero_vec(1 << (k - k_pack));
    let mut acc_init: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec(1 << k_pack);
    acc_init[0] = Ext::ONE;

    for i in 0..k_pack {
        let (lo, hi) = acc_init.split_at_mut(1 << i);
        let var = point[i];

        lo.iter_mut().zip(hi.iter_mut()).for_each(|(lo, hi)| {
            *hi = *lo * var;
            *lo -= *hi;
        });
    }

    acc_init
        .chunks(F::Packing::WIDTH)
        .zip(acc_packed.iter_mut())
        .for_each(|(chunk, packed)| *packed = Ext::ExtensionPacking::from_ext_slice(chunk));

    for i in 0..k - k_pack {
        let var = point[i + k_pack];
        let (lo, hi) = acc_packed.split_at_mut(1 << i);
        lo.iter_mut().zip(hi.iter_mut()).for_each(|(lo, hi)| {
            *hi = *lo * var;
            *lo -= *hi;
        });
    }

    acc_packed
}

#[tracing::instrument(skip_all)]
pub fn eq_single_p3<F: Field, Ext: ExtensionField<F>>(point: &Point<Ext>) -> Poly<Ext> {
    let mut out = Ext::zero_vec(1 << point.len());
    let point = point.reversed();
    let point = RowMajorMatrixView::new_col(&point);
    eval_eq_batch::<F, Ext, false>(point.as_view(), &mut out, &[Ext::ONE]);
    out.into()
}

#[tracing::instrument(skip_all)]
pub fn eq_single_split<F: Field, Ext: ExtensionField<F>>(
    point: &Point<Ext>,
    split_at: usize,
) -> Poly<Ext> {
    let (z0, z1) = point.split_at(split_at);

    let left: Poly<Ext> = {
        let mut out = Ext::zero_vec(1 << z0.len());
        let point = z0.reversed();
        let point = RowMajorMatrixView::new_col(&point);
        eval_eq_batch::<F, Ext, false>(point.as_view(), &mut out, &[Ext::ONE]);
        out.into()
    };

    let right: Poly<Ext> = {
        let mut out = Ext::zero_vec(1 << z1.len());
        let point = z1.reversed();
        let point = RowMajorMatrixView::new_col(&point);
        eval_eq_batch::<F, Ext, false>(point.as_view(), &mut out, &[Ext::ONE]);
        out.into()
    };

    let mut out = Ext::zero_vec(1 << point.len());
    out.par_chunks_mut(left.len())
        .zip_eq(right.par_iter())
        .for_each(|(out, &right)| {
            out.iter_mut()
                .zip_eq(left.iter())
                .for_each(|(out, &left)| *out = left * right);
        });
    out.into()
}

#[tracing::instrument(skip_all)]
pub fn eq_single_split_packed<F: Field, Ext: ExtensionField<F>>(
    point: &Point<Ext>,
    split_at: usize,
) -> Poly<Ext> {
    let (z0, z1) = point.split_at(split_at);

    use p3_field::PackedFieldExtension;

    let left = packed_eq(&z0);

    let right: Poly<Ext> = {
        let mut out = Ext::zero_vec(1 << z1.len());
        let point = z1.reversed();
        let point = RowMajorMatrixView::new_col(&point);
        eval_eq_batch::<F, Ext, false>(point.as_view(), &mut out, &[Ext::ONE]);
        out.into()
    };

    let mut out = Ext::zero_vec(1 << point.len());

    let n = out.len();
    out.par_chunks_mut(n / right.len())
        .zip_eq(right.par_iter())
        .for_each(|(out, &right)| {
            let packed = left.iter().map(|&left| left * right);
            // let unpacked = Ext::ExtensionPacking::to_ext_iter(packed).collect::<Vec<_>>();
            let unpacked = Ext::ExtensionPacking::to_ext_iter(packed);
            // out.iter(&unpacked);
            out.iter_mut()
                .zip(unpacked)
                .for_each(|(out, val)| *out = val);
        });
    out.into()
}

#[tracing::instrument(skip_all)]
pub fn eq_multi_naive<F: Field, Ext: ExtensionField<F>>(
    points: &[Point<Ext>],
    alpha: Ext,
) -> Poly<Ext> {
    let k = points[0].len();
    let n = points.len();

    let mut acc: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);
    let alphas = alpha.powers().take(n).collect();
    acc[..n].copy_from_slice(&alphas);

    for i in 0..k {
        let (lo, hi) = acc.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i]).collect::<Vec<_>>();

        lo.par_chunks_mut(n)
            .zip(hi.par_chunks_mut(n))
            .for_each(|(lo, hi)| {
                vars.iter()
                    .zip(lo.iter_mut().zip(hi.iter_mut()))
                    .for_each(|(&var, (lo, hi))| {
                        *hi = *lo * var;
                        *lo -= *hi;
                    });
            });
    }

    acc.par_chunks(n)
        .map(|row| row.iter().cloned().sum::<Ext>())
        .collect::<Vec<_>>()
        .into()
}

#[tracing::instrument(skip_all)]
pub fn eq_multi_p3<F: Field, Ext: ExtensionField<F>>(
    points: &[Point<Ext>],
    alpha: Ext,
) -> Poly<Ext> {
    let k = points[0].len();
    let n = points.len();
    let mut out = Ext::zero_vec(1 << k);
    let flat_points = points
        .iter()
        .flat_map(|powers| powers.iter().rev())
        .cloned()
        .collect::<Vec<_>>();
    let flat_points = RowMajorMatrixView::new(&flat_points, k).transpose();

    let alphas = alpha.powers().take(n).collect();
    eval_eq_batch::<F, Ext, false>(flat_points.as_view(), &mut out, &alphas);
    out.into()
}

#[tracing::instrument(skip_all)]
pub fn eq_multi_split<F: Field, Ext: ExtensionField<F>>(
    points: &[Point<Ext>],
    alpha: Ext,
    split_at: usize,
) -> Poly<Ext> {
    let k = points[0].len();
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

    let mut out = Ext::zero_vec(1 << k);
    out.par_chunks_mut(left.len() / n)
        .zip_eq(right.par_chunks(n))
        .for_each(|(out, right)| {
            out.iter_mut()
                .zip_eq(left.chunks(n))
                .for_each(|(out, left)| {
                    *out += dot_product::<Ext, _, _>(left.iter().cloned(), right.iter().cloned());
                });
        });
    out.into()
}

#[tracing::instrument(skip_all)]
pub fn eq_multi_split_packed<F: Field, Ext: ExtensionField<F>>(
    points: &[Point<Ext>],
    alpha: Ext,
    split_at: usize,
) -> Poly<Ext> {
    use p3_field::PackedFieldExtension;

    let k = points[0].len();
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

    let mut out = Ext::zero_vec(1 << k);
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
    out.into()
}

#[cfg(test)]
mod test {

    use p3_field::{extension::BinomialExtensionField, PrimeCharacteristicRing};

    use crate::study::{
        eq_multi_naive, eq_multi_p3, eq_multi_split, eq_multi_split_packed, eq_single_p3,
        eq_single_split, eq_single_split_packed, Point,
    };
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;

    fn bench<F, T>(name: &str, n_iter: usize, expected: &T, f: F)
    where
        F: Fn() -> T,
        T: PartialEq + std::fmt::Debug,
    {
        let mut lowest = std::time::Duration::MAX;
        let mut total = std::time::Duration::default();

        for _ in 0..n_iter {
            let start = std::time::Instant::now();
            let result = f();
            let this_time = start.elapsed();
            lowest = lowest.min(this_time);
            total += this_time;

            // if expected != &result {
            //     println!("MISMATCH, {}", name)
            // }
            assert_eq!(*expected, result);
        }

        let avg = total / n_iter as u32;
        println!(
            "{:20} Total: {:8.2?} Avg: {:8.2?} Lowest: {:8.2?}",
            name, total, avg, lowest
        );
    }

    #[test]
    fn bench_multi_eq() {
        let mut rng = crate::test::rng(1);

        let n_iter = 100;
        // crate::test::init_tracing();

        for k in 18..=21 {
            println!("--- **** ---");
            for n in [1, 10, 100] {
                println!("---");
                println!("testing k={} n={}", k, n);
                let points = (0..n)
                    .map(|_| Point::<Ext>::rand(&mut rng, k))
                    .collect::<Vec<_>>();
                // let alpha: Ext = rng.random();
                let alpha: Ext = Ext::ONE;

                let eq0 = eq_multi_naive::<F, Ext>(&points, alpha);
                bench("split", n_iter, &eq0, || {
                    eq_multi_split::<F, Ext>(&points, alpha, k / 2)
                });

                bench("p3", n_iter, &eq0, || eq_multi_p3::<F, Ext>(&points, alpha));

                bench("split_pack", n_iter, &eq0, || {
                    eq_multi_split_packed::<F, Ext>(&points, alpha, k / 2)
                });
            }
        }
    }

    #[test]
    fn bench_single_eq() {
        let mut rng = crate::test::rng(1);

        let n_iter = 1000;
        // crate::test::init_tracing();

        for k in 18..21 {
            println!("testing k={}", k);

            let point = Point::<Ext>::rand(&mut rng, k);
            let eq0 = point.eq(Ext::ONE);

            bench("eq_single_p3", n_iter, &eq0, || {
                eq_single_p3::<F, Ext>(&point)
            });

            bench("eq_single_split", n_iter, &eq0, || {
                eq_single_split::<F, Ext>(&point, k / 2)
            });

            bench("eq_multi_split", n_iter, &eq0, || {
                eq_multi_split::<F, Ext>(&[point.clone()], Ext::ONE, k / 2)
            });

            for split in (k / 2) - 5..(k / 2) + 5 {
                bench(format!("packed {split}").as_str(), n_iter, &eq0, || {
                    eq_single_split_packed::<F, Ext>(&point.clone(), split)
                });
            }
        }
    }
}
