use crate::pcs::PowClaim;
use crate::utils::{TwoAdicSlice, VecOps};
use crate::{pcs::EqClaim, poly::Point};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use rayon::prelude::*;

pub fn compress_claims<F: Field, Ext: ExtensionField<F>>(
    weights: &mut [Ext],
    sum: &mut Ext,
    alpha: Ext,
    eq_claims: &[EqClaim<Ext, Ext>],
    pow_claims: &[PowClaim<F, Ext>],
) {
    use crate::utils::TwoAdicSlice;
    let k = weights.k();
    assert_ne!(k, 0);
    eq_claims.iter().for_each(|claim| assert_eq!(claim.k(), k));
    pow_claims.iter().for_each(|claim| assert_eq!(claim.k, k));

    *sum += eq_claims
        .iter()
        .map(EqClaim::<Ext, Ext>::eval)
        .chain(pow_claims.iter().map(PowClaim::<F, Ext>::eval))
        .horner(alpha);

    let points = eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>();
    let vars = pow_claims.iter().map(PowClaim::var).collect::<Vec<_>>();

    compress_eqs::<F, Ext>(weights, &points, alpha);
    compress_pows(weights, &vars, alpha, points.len());
}

#[tracing::instrument(skip_all, fields(points = points.len(), k = out.k()))]
pub(super) fn compress_eqs<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    points: &[Point<Ext>],
    alpha: Ext,
) {
    let k = out.k();
    assert_ne!(k, 0);
    if points.is_empty() {
        return;
    }
    points.iter().for_each(|point| assert_eq!(point.len(), k));

    let alphas = alpha.powers().take(points.len()).collect();
    let points = points
        .iter()
        .flat_map(|point| point.iter().rev().cloned().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let points = RowMajorMatrixView::new(&points, k);
    let points = points.transpose();
    eval_eq_batch::<F, Ext, true>(points.as_view(), out, &alphas);
}

#[tracing::instrument(skip_all, fields(vars = vars.len(), k = out.k()))]
pub(super) fn compress_pows<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    vars: &[F],
    alpha: Ext,
    shift: usize,
) {
    let k = out.k();
    assert_ne!(k, 0);
    if vars.is_empty() {
        return;
    }
    let n = vars.len();

    let bin_powers = vars
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
        let bin_powers = bin_powers.iter().map(|c| c[i]).collect::<Vec<_>>();

        lo.par_chunks_mut(n)
            .zip(hi.par_chunks_mut(n))
            .for_each(|(lo, hi)| {
                bin_powers
                    .iter()
                    .zip(lo.iter().zip(hi.iter_mut()))
                    .for_each(|(&combo, (&lo, hi))| *hi = lo * combo);
            });
    }

    let alphas = alpha.powers().skip(shift).take(n).collect::<Vec<_>>();
    acc.par_chunks(n)
        .zip(out.par_iter_mut())
        .for_each(|(row, out)| {
            *out += row
                .iter()
                .zip(alphas.iter())
                .fold(Ext::ZERO, |acc, (&v, &alpha)| acc + alpha * v)
        });
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            sumcheck::compress::{compress_eqs, compress_pows},
            test::{make_eq_claims_ext, make_pow_claims_base},
        },
        poly::{Point, Poly},
        transcript::test_transcript::TestWriter,
        utils::VecOps,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};

    type F = BabyBear;
    type Ext = BinomialExtensionField<F, 4>;
    use rand::Rng;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

    fn compress_naive<F: Field, Ext: ExtensionField<F>>(
        k: usize,
        alpha: Ext,
        points: &[Point<Ext>],
        vars: &[F],
    ) -> Poly<Ext> {
        let eqs = points
            .iter()
            .map(|point| point.eq(Ext::ONE))
            .collect::<Vec<_>>();

        let pows = vars
            .iter()
            .map(|&x| x.powers().take(1 << k).collect())
            .collect::<Vec<_>>();

        let mut acc = vec![Ext::ZERO; 1 << k];
        acc.par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| *acc += eqs.iter().map(|eq| &eq[i]).horner(alpha));

        let shift = alpha.exp_u64(eqs.len() as u64);
        acc.par_iter_mut()
            .enumerate()
            .for_each(|(i, acc)| *acc += pows.iter().map(|eq| &eq[i]).horner_shifted(alpha, shift));

        acc.into()
    }

    #[test]
    fn test_compress() {
        let mut rng = crate::test::rng(1);

        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        for k in 1..5 {
            let poly = Poly::<Ext>::rand(&mut rng, k);
            for _ in 0..1000 {
                let alpha: Ext = rng.random();

                let n_eqs = rng.random_range(0..3);
                let n_pows = rng.random_range(0..3);
                let eq_claims =
                    make_eq_claims_ext::<_, F, Ext, Ext>(&mut transcript, n_eqs, &poly).unwrap();
                let pow_claims = make_pow_claims_base(&mut transcript, n_pows, &poly).unwrap();

                let points = eq_claims
                    .iter()
                    .map(|c| c.point().clone())
                    .collect::<Vec<_>>();
                let vars = pow_claims.iter().map(|c| c.var()).collect::<Vec<_>>();

                let acc0 = compress_naive(k, alpha, &points, &vars);
                let mut acc1 = Poly::<Ext>::zero(k);
                compress_eqs::<F, _>(&mut acc1, &points, alpha);
                compress_pows(&mut acc1, &vars, alpha, points.len());
                assert_eq!(acc0, acc1);
            }
        }
    }
}
