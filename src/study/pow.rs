use crate::{
    poly::Point,
    utils::{log2_strict, TwoAdicSlice},
};
use bincode::config::WithOtherEndian;
use itertools::Itertools;
use p3_field::{dot_product, Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::Matrix;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use rand::Rng;
use rayon::prelude::*;

#[inline]
fn binary_powers<F: Field>(vars: &[F], k: usize) -> Vec<F> {
    let flat_powers: Vec<F> = vars
        .iter()
        .cloned()
        .flat_map(|mut var| {
            (0..k)
                .map(|_| {
                    let ret = var;
                    var = var.square();
                    ret
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
        })
        .collect::<Vec<_>>();
    RowMajorMatrix::new(flat_powers, k).transpose().values
}

pub fn pow_compress_recursive<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[Ext],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = Ext::zero_vec(2 * k * n);
    recursive_inner::<F, Ext>(out, &mut workspace, &alphas, &flat_pows);
}

fn recursive_inner<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    workspace: &mut [Ext],
    buffer: &[Ext],
    flat_pows: &[Ext],
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

pub fn pow_compress_recursive_x<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[Ext],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = Ext::zero_vec((k + 1) * n);
    workspace[0..alphas.len()].copy_from_slice(&alphas);
    recursive_inner_x::<F, Ext>(n, out, &mut workspace, &flat_pows);
}

fn recursive_inner_x<F: Field, Ext: ExtensionField<F>>(
    n: usize,
    out: &mut [Ext],
    workspace: &mut [Ext],
    flat_pows: &[Ext],
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
            recursive_inner_x::<F, Ext>(n, low, workspace, pows_rest);

            workspace
                .iter_mut()
                .zip(buffer.iter())
                .zip(pows_cur.iter())
                .for_each(|((w, &buf), &p)| *w = buf * p);
            recursive_inner_x::<F, Ext>(n, high, workspace, pows_rest);
        }
    }
}

pub fn pow_compress_recursive_x_base<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
) {
    let k = out.k();
    let n = vars.len();
    let flat_pows = binary_powers(vars, k);
    let alphas = alpha.powers().take(n).collect();
    let mut workspace = F::zero_vec((k + 1) * n);
    // let mut workspace = Ext::zero_vec(2 * k * n);

    workspace[0..alphas.len()].copy_from_slice(&vec![F::ONE; alphas.len()]);
    recursive_inner_x_base::<F, Ext>(n, out, &mut workspace, &alphas, &flat_pows);
}

fn recursive_inner_x_base<F: Field, Ext: ExtensionField<F>>(
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
            recursive_inner_x_base::<F, Ext>(n, low, workspace, challenges, pows_rest);

            workspace
                .iter_mut()
                .zip(buffer.iter())
                .zip(pows_cur.iter())
                .for_each(|((w, &buf), &p)| *w = buf * p);
            recursive_inner_x_base::<F, Ext>(n, high, workspace, challenges, pows_rest);
        }
    }
}

#[tracing::instrument(skip_all)]
pub fn pow_compress_ref<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[Ext],
    alpha: Ext,
) {
    let k = out.k();
    let alphas = alpha.powers().take(vars.len()).collect();
    for (&var, &alpha) in vars.iter().zip(alphas.iter()) {
        let pows = var.powers().take(1 << k).collect();
        out.par_iter_mut()
            .zip(pows.par_iter())
            .for_each(|(acc, &el)| *acc += alpha * el);
    }
}

#[tracing::instrument(skip_all)]
pub fn pow_compress_base_ref<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
) {
    let k = out.k();
    let alphas = alpha.powers().take(vars.len()).collect();
    for (&var, &alpha) in vars.iter().zip(alphas.iter()) {
        let pows = var.powers().take(1 << k).collect();
        out.par_iter_mut()
            .zip(pows.par_iter())
            .for_each(|(acc, &el)| *acc += alpha * el);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use p3_field::{
        extension::{BinomialExtensionField, PackedBinomialExtensionField},
        PrimeCharacteristicRing,
    };
    type F = p3_koala_bear::KoalaBear;
    // type Ext = p3_koala_bear::KoalaBear;
    // type Ext = BinomialExtensionField<F, 4>;
    type Ext = F;

    #[test]
    fn bench_compress_pow() {
        let mut rng = crate::test::rng(1);
        let alpha: Ext = rng.random();

        let n_iter = 10;
        // crate::test::init_tracing();

        for k in [2, 3, 4, 5, 6, 7, 12, 18] {
            println!("--- **** k: {}", k);
            for n in [1, 10, 100] {
                let vars: Vec<Ext> = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
                println!(" --- **** n: {}", n);

                // #[tracing::instrument(skip_all)]
                // fn binary_powers<F: Field>(vars: &[F], k: usize) -> RowMajorMatrix<F> {
                //     let flat_powers: Vec<F> = vars
                //         .iter()
                //         .cloned()
                //         .flat_map(|mut var| {
                //             (0..k)
                //                 .map(|_| {
                //                     let ret = var;
                //                     var = var.square();
                //                     ret
                //                 })
                //                 .collect::<Vec<_>>()
                //                 .into_iter()
                //                 .rev()
                //         })
                //         .collect::<Vec<_>>();
                //     // tracing::info_span!("transpose")
                //     //     .in_scope(|| RowMajorMatrix::new(flat_powers, k).transpose())
                //     RowMajorMatrix::new(flat_powers, k).transpose()
                // }

                // #[tracing::instrument(skip_all)]
                // fn binary_powers2<F: Field>(vars: &[F], k: usize) -> RowMajorMatrix<F> {
                //     let mut flat_powers = F::zero_vec(vars.len() * k);
                //     let n = vars.len();
                //     vars.iter().enumerate().for_each(|(i, &var)| {
                //         let mut cur = var;
                //         (0..k).for_each(|j| {
                //             flat_powers[(k - 1 - j) * n + i] = cur;
                //             cur = cur.square();
                //         })
                //     });
                //     RowMajorMatrix::new(flat_powers, n)
                // }

                // let u0 = binary_powers(&vars, k);
                // let u1 = binary_powers2(&vars, k);
                // assert_eq!(u0, u1);

                let mut out_ref = Ext::zero_vec(1 << k);
                pow_compress_ref::<F, Ext>(&mut out_ref, &vars, alpha);

                crate::test::bench("rec0", n_iter, Some(&out_ref), || {
                    let mut out = Ext::zero_vec(1 << k);

                    let alphas = alpha.powers().take(vars.len()).collect();
                    p3_multilinear_util::eq_batch::eval_pow_batch::<F, Ext, false>(
                        &vars, &mut out, &alphas,
                    );
                    out
                });

                crate::test::bench("rec0", n_iter, Some(&out_ref), || {
                    let mut out = Ext::zero_vec(1 << k);
                    pow_compress_recursive::<F, Ext>(&mut out, &vars, alpha);
                    out
                });

                crate::test::bench("rec1", n_iter, Some(&out_ref), || {
                    let mut out = Ext::zero_vec(1 << k);
                    pow_compress_recursive_x::<F, Ext>(&mut out, &vars, alpha);
                    out
                });
            }
        }
    }
}
