use crate::{
    poly::Point,
    utils::{log2_strict, TwoAdicSlice},
};
use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use rayon::prelude::*;

fn binary_powers_reversed<F: Field>(vars: &[F], k: usize) -> Vec<F> {
    let mut flat_powers = F::zero_vec(vars.len() * k);
    let n = vars.len();
    vars.iter().enumerate().for_each(|(i, &var)| {
        let mut cur = var;
        (0..k).for_each(|j| {
            flat_powers[(k - 1 - j) * n + i] = cur;
            cur = cur.square();
        });
    });
    RowMajorMatrix::new(flat_powers, n).values
}

pub fn pow_compress_recursive<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers_reversed(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = Ext::zero_vec(2 * k * n);
    recursive_inner::<F, Ext>(out, &mut workspace, &alphas, &flat_pows);
}

fn recursive_inner<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    workspace: &mut [Ext],
    buffer: &[Ext],
    flat_pows: &[F],
) {
    use crate::utils::TwoAdicSlice;
    let k = out.k();
    let n = buffer.len();
    debug_assert_eq!(k * buffer.len(), flat_pows.len());

    match k {
        0 => {
            out[0] += buffer.iter().copied().sum::<Ext>();
        }
        _ => {
            let (pows_cur, pows_rest) = flat_pows.split_at(n);
            let (s0_buffer, workspace) = workspace.split_at_mut(n);
            let (s1_buffer, workspace) = workspace.split_at_mut(n);

            s0_buffer.copy_from_slice(buffer);
            s1_buffer
                .iter_mut()
                .zip(buffer.iter())
                .zip(pows_cur.iter())
                .for_each(|((s1, &buf), &p)| *s1 = buf * p);

            let (low, high) = out.split_at_mut(out.len() / 2);
            recursive_inner::<F, Ext>(low, workspace, s0_buffer, pows_rest);
            recursive_inner::<F, Ext>(high, workspace, s1_buffer, pows_rest);
        }
    }
}

pub fn pow_compress_recursive2<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers_reversed(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = F::zero_vec(2 * k * n);
    recursive_inner2::<F, Ext>(out, &mut workspace, &vec![F::ONE; n], &flat_pows, &alphas);
}

fn recursive_inner2<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    workspace: &mut [F],
    buffer: &[F],
    flat_pows: &[F],
    challenges: &[Ext],
) {
    use crate::utils::TwoAdicSlice;
    let k = out.k();
    let n = buffer.len();
    debug_assert_eq!(k * buffer.len(), flat_pows.len());

    match k {
        0 => {
            // println!("challenges: {:?}", challenges);
            out[0] += buffer
                .iter()
                .zip(challenges.iter())
                .fold(Ext::ZERO, |acc, (w, &ch)| acc + ch * *w);
        }
        _ => {
            let (pows_cur, pows_rest) = flat_pows.split_at(n);
            let (s0_buffer, workspace) = workspace.split_at_mut(n);
            let (s1_buffer, workspace) = workspace.split_at_mut(n);

            s0_buffer.copy_from_slice(buffer);
            s1_buffer
                .iter_mut()
                .zip(buffer.iter())
                .zip(pows_cur.iter())
                .for_each(|((s1, &buf), &p)| *s1 = buf * p);

            let (low, high) = out.split_at_mut(out.len() / 2);
            recursive_inner2::<F, Ext>(low, workspace, s0_buffer, pows_rest, challenges);
            recursive_inner2::<F, Ext>(high, workspace, s1_buffer, pows_rest, challenges);
        }
    }
}

pub fn pow_compress_recursive_less_mem<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers_reversed(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = Ext::zero_vec((k + 1) * n);
    workspace[0..alphas.len()].copy_from_slice(&alphas);
    recursive_inner_less_mem::<F, Ext>(n, out, &mut workspace, &flat_pows);
}

fn recursive_inner_less_mem<F: Field, Ext: ExtensionField<F>>(
    n: usize,
    out: &mut [Ext],
    workspace: &mut [Ext],
    flat_pows: &[F],
) {
    use crate::utils::TwoAdicSlice;
    let k = out.k();

    match k {
        0 => {
            out[0] += workspace.iter().copied().sum::<Ext>();
        }
        _ => {
            let (pows_cur, pows_rest) = flat_pows.split_at(n);
            let (buffer, workspace) = workspace.split_at_mut(n);

            let (low, high) = out.split_at_mut(out.len() / 2);

            workspace[0..n].copy_from_slice(buffer);
            recursive_inner_less_mem::<F, Ext>(n, low, workspace, pows_rest);

            workspace
                .iter_mut()
                .zip(buffer.iter())
                .zip(pows_cur.iter())
                .for_each(|((w, &buf), &p)| *w = buf * p);
            recursive_inner_less_mem::<F, Ext>(n, high, workspace, pows_rest);
        }
    }
}

pub fn pow_compress_recursive_less_mem2<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers_reversed(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = F::zero_vec((k + 1) * n);
    workspace[0..n].copy_from_slice(&vec![F::ONE; n]);
    recursive_inner_less_mem2::<F, Ext>(n, out, &mut workspace, &alphas, &flat_pows);
}

fn recursive_inner_less_mem2<F: Field, Ext: ExtensionField<F>>(
    n: usize,
    out: &mut [Ext],
    workspace: &mut [F],
    challenges: &[Ext],
    flat_pows: &[F],
) {
    use crate::utils::TwoAdicSlice;
    let k = out.k();

    match k {
        0 => {
            out[0] += workspace
                .iter()
                .zip(challenges.iter())
                .fold(Ext::ZERO, |acc, (w, &ch)| acc + ch * *w);
        }
        _ => {
            let (pows_cur, pows_rest) = flat_pows.split_at(n);
            let (buffer, workspace) = workspace.split_at_mut(n);

            let (low, high) = out.split_at_mut(out.len() / 2);

            workspace[0..n].copy_from_slice(buffer);
            recursive_inner_less_mem2::<F, Ext>(n, low, workspace, challenges, pows_rest);

            workspace
                .iter_mut()
                .zip(buffer.iter())
                .zip(pows_cur.iter())
                .for_each(|((w, &buf), &p)| *w = buf * p);
            recursive_inner_less_mem2::<F, Ext>(n, high, workspace, challenges, pows_rest);
        }
    }
}

#[tracing::instrument(skip_all)]
pub fn pow_compress_ref<F: Field, Ext: ExtensionField<F>>(out: &mut [Ext], vars: &[F], alpha: Ext) {
    let k = out.k();
    let alphas = alpha.powers().take(vars.len()).collect();
    for (&var, &alpha) in vars.iter().zip(alphas.iter()) {
        let pows = var.powers().take(1 << k).collect();
        // out.par_iter_mut()
        //     .zip(pows.par_iter())
        out.iter_mut()
            .zip(pows.iter())
            .for_each(|(acc, &el)| *acc += alpha * el);
    }
}

#[tracing::instrument(skip_all)]
fn packed_flat_pows<F: Field>(points: &[Point<F>]) -> Vec<F::Packing> {
    let k = points[0].len();
    let k_pack = log2_strict(F::Packing::WIDTH);
    let n = points.len();

    let mut acc_init: Vec<F> = crate::utils::unsafe_allocate_zero_vec((1 << k_pack) * n);
    acc_init[..n].copy_from_slice(&vec![F::ONE; n]);

    for i in 0..k_pack {
        let (lo, hi) = acc_init.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i]).collect::<Vec<_>>();

        lo.chunks_mut(n).zip(hi.chunks_mut(n)).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| *hi = *lo * var);
        });
    }

    let mut acc_init_transposed = F::zero_vec(acc_init.len());
    transpose::transpose(
        &acc_init,
        &mut acc_init_transposed,
        acc_init.len() / F::Packing::WIDTH,
        F::Packing::WIDTH,
    );

    let mut acc_packed: Vec<F::Packing> =
        crate::utils::unsafe_allocate_zero_vec((1 << (k - k_pack)) * n);

    acc_init_transposed
        .chunks(F::Packing::WIDTH)
        .zip(acc_packed.iter_mut())
        .for_each(|(chunk, packed)| *packed = *F::Packing::from_slice(chunk));

    for i in 0..k - k_pack {
        let (lo, hi) = acc_packed.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i + k_pack]).collect::<Vec<_>>();
        lo.chunks_mut(n).zip(hi.chunks_mut(n)).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| *hi = *lo * var);
        });
    }

    acc_packed
}

#[tracing::instrument(skip_all)]
fn flat_pows<F: Field>(points: &[Point<F>]) -> Vec<F> {
    let k = points[0].len();
    let n = points.len();

    let mut acc: Vec<F> = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);
    acc[..n].copy_from_slice(&vec![F::ONE; n]);
    for i in 0..k {
        let (lo, hi) = acc.split_at_mut((1 << i) * n);
        let vars = points.iter().map(|c| c[i]).collect::<Vec<_>>();
        lo.chunks_mut(n).zip(hi.chunks_mut(n)).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| *hi = *lo * var);
        });
    }
    acc
}

#[tracing::instrument(skip_all)]
pub fn pow_compress_split<F: Field, Ext: ExtensionField<F>, const PAR: bool>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
    split_at: usize,
) {
    fn binary_powers<F: Field>(vars: &[F], k: usize) -> Vec<Point<F>> {
        vars.iter()
            .cloned()
            .map(|mut var| {
                (0..k)
                    .map(|_| {
                        let ret = var;
                        var = var.square();
                        ret
                    })
                    .collect::<Vec<_>>()
                    .into()
            })
            .collect::<Vec<_>>()
    }

    let k = out.k();
    let n = vars.len();

    let points = binary_powers(vars, k);
    let left = points
        .iter()
        .map(|point| point.split_at(split_at).0)
        .collect::<Vec<_>>();
    let left = flat_pows(&left);
    let right = points
        .iter()
        .map(|point| point.split_at(split_at).1)
        .collect::<Vec<_>>();
    let right = flat_pows(&right);
    let alphas = alpha.powers().take(n).collect();

    if PAR {
        out.par_chunks_mut(left.len() / n)
            .zip_eq(right.par_chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut()
                    .zip_eq(left.chunks(n))
                    .for_each(|(out, left)| {
                        *out = left
                            .iter()
                            .zip_eq(right.iter())
                            .zip_eq(alphas.iter())
                            .map(|((&l, &r), &alpha)| alpha * (l * r))
                            .sum::<Ext>();
                    });
            });
    } else {
        out.chunks_mut(left.len() / n)
            .zip_eq(right.chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut()
                    .zip_eq(left.chunks(n))
                    .for_each(|(out, left)| {
                        *out = left
                            .iter()
                            .zip_eq(right.iter())
                            .zip_eq(alphas.iter())
                            .map(|((&l, &r), &alpha)| alpha * (l * r))
                            .sum::<Ext>();
                    });
            });
    }
}

pub fn pow_compress_packed_packed<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext::ExtensionPacking],
    vars: &[F],
    alpha: Ext,
    split_at: usize,
) {
    let k_pack = log2_strict(F::Packing::WIDTH);
    let k = out.k() + k_pack;
    let n = vars.len();

    fn binary_powers<F: Field>(vars: &[F], k: usize) -> Vec<Point<F>> {
        vars.iter()
            .cloned()
            .map(|mut var| {
                (0..k)
                    .map(|_| {
                        let ret = var;
                        var = var.square();
                        ret
                    })
                    .collect::<Vec<_>>()
                    .into()
            })
            .collect::<Vec<_>>()
    }

    let points = binary_powers(vars, k);
    let left = points
        .iter()
        .map(|point| point.split_at(split_at).0)
        .collect::<Vec<_>>();
    let left_packed = packed_flat_pows(&left);
    let right = points
        .iter()
        .map(|point| point.split_at(split_at).1)
        .collect::<Vec<_>>();
    let right = flat_pows(&right);

    let alphas = alpha.powers().take(n).collect();
    let alphas = alphas
        .iter()
        .map(|&alpha| Ext::ExtensionPacking::from_ext_slice(&vec![alpha; F::Packing::WIDTH]))
        .collect::<Vec<_>>();

    out.par_chunks_mut(left_packed.len() / n)
        .zip_eq(right.par_chunks(n))
        .for_each(|(out, right)| {
            out.iter_mut()
                .zip(left_packed.chunks(n))
                .for_each(|(out, left)| {
                    *out = left
                        .iter()
                        .zip_eq(right.iter())
                        .zip(alphas.iter())
                        .map(|((&left, &right), &alpha)| alpha * (left * right))
                        .sum::<Ext::ExtensionPacking>();
                });
        });
}

#[cfg(test)]
mod test {
    use crate::{
        study::pow::{
            pow_compress_packed_packed, pow_compress_recursive, pow_compress_recursive2,
            pow_compress_recursive_less_mem, pow_compress_recursive_less_mem2, pow_compress_ref,
            pow_compress_split,
        },
        test::unpack_ext,
        utils::log2_strict,
    };
    use p3_field::{extension::BinomialExtensionField, PackedValue};
    use p3_field::{
        // extension::{BinomialExtensionField, PackedBinomialExtensionField},
        ExtensionField,
        Field,
        PrimeCharacteristicRing,
    };
    use rand::Rng;

    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;
    type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

    #[test]
    fn bench_compress_pow() {
        let mut rng = crate::test::rng(1);
        let alpha: Ext = rng.random();
        let n_iter = 10;
        let k_pack = log2_strict(<F as Field>::Packing::WIDTH);

        for k in [10, 18, 21] {
            println!("--- **** k: {}", k);
            for n in [1, 10, 100] {
                let vars: Vec<F> = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
                println!("--- **** n: {}", n);

                let mut out_ref = Ext::zero_vec(1 << k);
                pow_compress_ref::<F, Ext>(&mut out_ref, &vars, alpha);

                crate::test::bench(
                    "ref",
                    n_iter,
                    || (),
                    |_| Ext::zero_vec(1 << k),
                    |_, out| {
                        pow_compress_ref::<F, Ext>(out, &vars, alpha);
                    },
                    |out, _| assert_eq!(out, out_ref),
                );

                crate::test::bench(
                    "rec",
                    n_iter,
                    || (),
                    |_| Ext::zero_vec(1 << k),
                    |_, out| {
                        pow_compress_recursive::<F, Ext>(out, &vars, alpha);
                    },
                    |out, _| assert_eq!(out, out_ref),
                );

                crate::test::bench(
                    "rec2",
                    n_iter,
                    || (),
                    |_| Ext::zero_vec(1 << k),
                    |_, out| {
                        pow_compress_recursive2::<F, Ext>(out, &vars, alpha);
                    },
                    |out, _| assert_eq!(out, out_ref),
                );

                crate::test::bench(
                    "less-mem",
                    n_iter,
                    || (),
                    |_| Ext::zero_vec(1 << k),
                    |_, out| {
                        pow_compress_recursive_less_mem::<F, Ext>(out, &vars, alpha);
                    },
                    |out, _| assert_eq!(out, out_ref),
                );

                crate::test::bench(
                    "less-mem2",
                    n_iter,
                    || (),
                    |_| Ext::zero_vec(1 << k),
                    |_, out| {
                        pow_compress_recursive_less_mem2::<F, Ext>(out, &vars, alpha);
                    },
                    |out, _| assert_eq!(out, out_ref),
                );

                crate::test::bench(
                    "split",
                    n_iter,
                    || (),
                    |_| Ext::zero_vec(1 << k),
                    |_, out| {
                        pow_compress_split::<F, Ext, true>(out, &vars, alpha, k / 2);
                    },
                    |out, _| assert_eq!(out, out_ref),
                );

                crate::test::bench(
                    "split-packed",
                    n_iter,
                    || (),
                    |_| PackedExt::zero_vec(1 << (k - k_pack)),
                    |_, out| {
                        pow_compress_packed_packed::<F, Ext>(out, &vars, alpha, k / 2);
                    },
                    |out, _| assert_eq!(unpack_ext::<F, Ext>(out), out_ref),
                );
            }
        }
    }
}
