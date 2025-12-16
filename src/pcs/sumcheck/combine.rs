use crate::{
    p3_field_prelude::*,
    pcs::{EqClaim, PowClaim},
    utils::VecOps,
};
use p3_util::log2_strict_usize;

#[tracing::instrument(skip_all)]
pub(crate) fn combine_claims_packed<F: Field, Ext: ExtensionField<F>>(
    weights: &mut [Ext::ExtensionPacking],
    sum: &mut Ext,
    alpha: Ext,
    eq_claims: &[EqClaim<Ext, Ext>],
    pow_claims: &[PowClaim<F, Ext>],
) {
    use crate::utils::TwoAdicSlice;
    let k = weights.k() + log2_strict_usize(F::Packing::WIDTH);
    eq_claims.iter().for_each(|claim| assert_eq!(claim.k(), k));
    pow_claims.iter().for_each(|claim| assert_eq!(claim.k, k));

    *sum += eq_claims
        .iter()
        .map(EqClaim::<Ext, Ext>::eval)
        .chain(pow_claims.iter().map(PowClaim::<F, Ext>::eval))
        .horner(alpha);

    let points = eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>();
    let vars = pow_claims.iter().map(PowClaim::var).collect::<Vec<_>>();

    eq::combine_eqs_packed::<F, Ext>(weights, &points, alpha);
    pow::combine_pows_packed::<F, Ext>(weights, &vars, alpha, points.len());
}

#[tracing::instrument(skip_all)]
pub(crate) fn combine_claims<F: Field, Ext: ExtensionField<F>>(
    weights: &mut [Ext],
    sum: &mut Ext,
    alpha: Ext,
    eq_claims: &[EqClaim<Ext, Ext>],
    pow_claims: &[PowClaim<F, Ext>],
) {
    use crate::utils::TwoAdicSlice;
    let k = weights.k();
    eq_claims.iter().for_each(|claim| assert_eq!(claim.k(), k));
    pow_claims.iter().for_each(|claim| assert_eq!(claim.k, k));

    *sum += eq_claims
        .iter()
        .map(EqClaim::<Ext, Ext>::eval)
        .chain(pow_claims.iter().map(PowClaim::<F, Ext>::eval))
        .horner(alpha);

    let points = eq_claims.iter().map(EqClaim::point).collect::<Vec<_>>();
    let vars = pow_claims.iter().map(PowClaim::var).collect::<Vec<_>>();

    eq::combine_eqs::<F, Ext>(weights, &points, alpha);
    pow::combine_pows::<F, Ext>(weights, &vars, alpha, points.len());
}

pub(crate) mod eq {
    use crate::p3_field_prelude::*;
    use crate::{poly::Point, utils::TwoAdicSlice};
    use itertools::Itertools;
    use p3_matrix::Matrix;
    use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
    use p3_util::log2_strict_usize;
    use rayon::prelude::*;

    fn batch_eqs<F: Field>(points: RowMajorMatrixView<F>, alpha: F) -> RowMajorMatrix<F> {
        let k = points.height();
        let n = points.width();
        assert_ne!(n, 0);

        let mut mat = RowMajorMatrix::new(F::zero_vec(n * (1 << k)), n);
        mat.row_mut(0).copy_from_slice(&alpha.powers().collect_n(n));
        points.row_slices().enumerate().for_each(|(i, vars)| {
            let (mut lo, mut hi) = mat.split_rows_mut(1 << i);
            lo.rows_mut().zip(hi.rows_mut()).for_each(|(lo, hi)| {
                vars.iter()
                    .zip(lo.iter_mut().zip(hi.iter_mut()))
                    .for_each(|(&var, (lo, hi))| {
                        *hi = *lo * var;
                        *lo -= *hi;
                    });
            });
        });
        mat
    }

    fn packed_batch_eqs<F: Field, Ext: ExtensionField<F>>(
        points: RowMajorMatrixView<Ext>,
    ) -> RowMajorMatrix<Ext::ExtensionPacking> {
        let k = points.height();
        let n = points.width();
        assert_ne!(n, 0);
        let k_pack = log2_strict_usize(F::Packing::WIDTH);

        let (init_vars, rest_vars) = points.split_rows(k_pack);
        let mut mat =
            RowMajorMatrix::new(Ext::ExtensionPacking::zero_vec(n * (1 << (k - k_pack))), n);
        if k_pack > 0 {
            init_vars
                .transpose()
                .row_slices()
                .zip(mat.values.iter_mut())
                .for_each(|(vars, packed)| {
                    let eq = Point::<Ext>::new(vars.to_vec()).eq(Ext::ONE);
                    *packed = Ext::ExtensionPacking::from_ext_slice(&eq);
                });
        } else {
            mat.row_mut(0).fill(Ext::ExtensionPacking::ONE);
        }

        rest_vars.row_slices().enumerate().for_each(|(i, vars)| {
            let (mut lo, mut hi) = mat.split_rows_mut(1 << i);
            lo.rows_mut().zip(hi.rows_mut()).for_each(|(lo, hi)| {
                vars.iter()
                    .zip(lo.iter_mut().zip(hi.iter_mut()))
                    .for_each(|(&var, (lo, hi))| {
                        *hi = *lo * var;
                        *lo -= *hi;
                    });
            });
        });

        mat
    }

    #[tracing::instrument(skip_all, fields(n = points.len(), k = out.k() + log2_strict_usize(F::Packing::WIDTH)))]
    pub(crate) fn combine_eqs_packed<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext::ExtensionPacking],
        points: &[Point<Ext>],
        alpha: Ext,
    ) {
        if points.is_empty() {
            return;
        }

        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = points.iter().map(Point::k).all_equal_value().unwrap();
        assert_eq!(out.k() + k_pack, k);
        assert!(k >= k_pack);

        if k_pack * 2 > k {
            points
                .iter()
                .zip(alpha.powers())
                .for_each(|(point, alpha)| {
                    let eq = point.eq(alpha);
                    out.iter_mut()
                        .zip_eq(eq.chunks(F::Packing::WIDTH))
                        .for_each(|(out, chunk)| {
                            *out += Ext::ExtensionPacking::from_ext_slice(chunk)
                        });
                });
        } else {
            let points = Point::transpose(points);
            let (left, right) = points.split_rows(k / 2);
            let left = packed_batch_eqs::<F, _>(left);
            let right = batch_eqs(right, alpha);
            out.par_chunks_mut(left.height())
                .zip_eq(right.par_row_slices())
                .for_each(|(out, right)| {
                    out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                        *out +=
                            dot_product::<Ext::ExtensionPacking, _, _>(left, right.iter().copied());
                    });
                });
        }
    }

    #[tracing::instrument(skip_all, fields(n = points.len(), k = out.k()))]
    pub(crate) fn combine_eqs<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext],
        points: &[Point<Ext>],
        alpha: Ext,
    ) {
        if points.is_empty() {
            return;
        }

        let k = points.iter().map(Point::k).all_equal_value().unwrap();
        assert_eq!(out.k(), k);

        let points = Point::transpose(points);
        let (left, right) = points.split_rows(k / 2);
        let left = batch_eqs(left, Ext::ONE);
        let right = batch_eqs(right, alpha);

        out.par_chunks_mut(left.height())
            .zip_eq(right.par_row_slices())
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                    *out += dot_product::<Ext, _, _>(left, right.iter().copied());
                });
            });
    }

    #[cfg(test)]
    mod test {
        use crate::poly::Point;
        use crate::utils::{TwoAdicSlice, unpack};
        use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
        use p3_field::{PackedValue, extension::BinomialExtensionField};
        use p3_util::log2_strict_usize;
        use rand::Rng;
        use rayon::prelude::*;

        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

        #[tracing::instrument(skip_all, fields(points = points.len()))]
        fn combine_eq_ref<F: Field, Ext: ExtensionField<F>>(
            out: &mut [Ext],
            points: &[Point<Ext>],
            alpha: Ext,
        ) {
            if points.is_empty() {
                return;
            }
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
                    .for_each(|(lo, hi)| {
                        vars.iter().zip(lo.iter_mut().zip(hi.iter_mut())).for_each(
                            |(&z, (lo, hi))| {
                                *hi = *lo * z;
                                *lo -= *hi;
                            },
                        );
                    });
            }
            acc.par_chunks(n)
                .zip(out.par_iter_mut())
                .for_each(|(row, out)| *out += row.iter().cloned().sum::<Ext>());
        }

        #[test]
        fn test_combine_eqs() {
            let mut rng = crate::test::rng(1);
            let alpha: Ext = rng.random();
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
            for k in k_pack..10 {
                for n in [1, 2, 10, 11] {
                    let points = (0..n)
                        .map(|_| Point::<Ext>::rand(&mut rng, k))
                        .collect::<Vec<_>>();
                    let mut out0 = Ext::zero_vec(1 << k);
                    combine_eq_ref::<F, Ext>(&mut out0, &points, alpha);
                    let mut out1 = Ext::zero_vec(1 << k);
                    super::combine_eqs::<F, Ext>(&mut out1, &points, alpha);
                    assert_eq!(out0, out1);
                    let mut out_packed = PackedExt::zero_vec(1 << (k - k_pack));
                    super::combine_eqs_packed::<F, Ext>(&mut out_packed, &points, alpha);
                    assert_eq!(out0, unpack::<F, Ext>(&out_packed));
                }
            }
        }
    }
}

pub(crate) mod pow {
    use crate::p3_field_prelude::*;
    use crate::utils::TwoAdicSlice;
    use itertools::Itertools;
    use p3_matrix::Matrix;
    use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
    use p3_util::log2_strict_usize;
    use rayon::prelude::*;

    fn binary_powers<F: Field>(vars: &[F], k: usize) -> RowMajorMatrix<F> {
        let n = vars.len();
        let mut points = F::zero_vec(k * n);
        let n = vars.len();
        vars.iter().enumerate().for_each(|(i, &var)| {
            let mut cur = var;
            (0..k).for_each(|j| {
                points[j * n + i] = cur;
                cur = cur.square();
            });
        });
        RowMajorMatrix::new(points, n)
    }

    fn batch_pows<F: Field>(points: RowMajorMatrixView<F>) -> RowMajorMatrix<F> {
        let k = points.height();
        let n = points.width();

        let mut mat = RowMajorMatrix::new(F::zero_vec(n * (1 << k)), n);
        mat.row_mut(0).fill(F::ONE);

        points.row_slices().enumerate().for_each(|(i, vars)| {
            let (lo, mut hi) = mat.split_rows_mut(1 << i);
            lo.rows().zip(hi.rows_mut()).for_each(|(lo, hi)| {
                vars.iter()
                    .zip(lo.zip(hi.iter_mut()))
                    .for_each(|(&var, (lo, hi))| *hi = lo * var);
            });
        });
        mat
    }

    fn packed_batch_pows<F: Field>(points: RowMajorMatrixView<F>) -> RowMajorMatrix<F::Packing> {
        let k = points.height();
        let n = points.width();
        assert_ne!(n, 0);
        let k_pack = log2_strict_usize(F::Packing::WIDTH);

        let (init_vars, rest_vars) = points.split_rows(k_pack);
        let mut mat = RowMajorMatrix::new(F::Packing::zero_vec(n * (1 << (k - k_pack))), n);
        if k_pack > 0 {
            init_vars
                .transpose()
                .row_slices()
                .zip(mat.values.iter_mut())
                .for_each(|(vars, packed)| {
                    let point = RowMajorMatrixView::new(vars, 1);
                    *packed = *F::Packing::from_slice(&batch_pows(point).values)
                });
        } else {
            mat.row_mut(0).fill(F::Packing::ONE);
        }

        for (i, vars) in rest_vars.row_slices().enumerate() {
            let (lo, mut hi) = mat.split_rows_mut(1 << i);
            lo.rows().zip(hi.rows_mut()).for_each(|(lo, hi)| {
                vars.iter()
                    .zip(lo.zip(hi.iter_mut()))
                    .for_each(|(&var, (lo, hi))| *hi = lo * var);
            });
        }
        mat
    }

    #[tracing::instrument(skip_all, fields(vars = vars.len(), k = out.k() + log2_strict_usize(F::Packing::WIDTH)))]
    pub(crate) fn combine_pows_packed<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext::ExtensionPacking],
        vars: &[F],
        alpha: Ext,
        shift: usize,
    ) {
        if vars.is_empty() {
            return;
        }

        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = out.k() + k_pack;
        let n = vars.len();

        if k_pack * 2 > k {
            vars.iter()
                .zip(alpha.powers().skip(shift))
                .for_each(|(&var, challenge)| {
                    let pow = Ext::from(var).shifted_powers(challenge).collect_n(1 << k);
                    out.iter_mut()
                        .zip_eq(pow.chunks(F::Packing::WIDTH))
                        .for_each(|(out, chunk)| {
                            *out += Ext::ExtensionPacking::from_ext_slice(chunk)
                        });
                });
        } else {
            let points = binary_powers(vars, k);
            let (left, right) = points.split_rows(k / 2);
            let left = packed_batch_pows(left);
            let right = batch_pows(right);

            let alphas = alpha
                .powers()
                .skip(shift)
                .take(n)
                .map(Ext::ExtensionPacking::from)
                .collect::<Vec<_>>();

            out.par_chunks_mut(left.height())
                .zip(right.par_row_slices())
                .for_each(|(out, right)| {
                    out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                        *out += left
                            .zip(right.iter())
                            .zip(alphas.iter())
                            .map(|((left, &right), &alpha)| alpha * (left * right))
                            .sum::<Ext::ExtensionPacking>();
                    });
                });
        }
    }

    #[tracing::instrument(skip_all, fields(vars = vars.len(), k = out.k()))]
    pub(crate) fn combine_pows<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext],
        vars: &[F],
        alpha: Ext,
        shift: usize,
    ) {
        if vars.is_empty() {
            return;
        }

        let k = out.k();
        let n = vars.len();

        let points = binary_powers(vars, k);
        let (left, right) = points.split_rows(k / 2);
        let left = batch_pows(left);
        let right = batch_pows(right);

        let alphas = alpha.powers().skip(shift).take(n).collect::<Vec<_>>();
        out.par_chunks_mut(left.height())
            .zip_eq(right.par_row_slices())
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                    *out += left
                        .zip(right.iter())
                        .zip(alphas.iter())
                        .map(|((left, &right), &alpha)| alpha * (left * right))
                        .sum::<Ext>();
                });
            });
    }

    #[cfg(test)]
    mod test {
        use crate::utils::{TwoAdicSlice, unpack};
        use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
        use p3_field::{PackedValue, extension::BinomialExtensionField};
        use p3_util::log2_strict_usize;
        use rand::Rng;
        use rayon::prelude::*;

        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

        #[tracing::instrument(skip_all)]
        fn combine_pow_ref<F: Field, Ext: ExtensionField<F>>(
            out: &mut [Ext],
            vars: &[F],
            alpha: Ext,
            shift: usize,
        ) {
            let k = out.k();
            let alphas = alpha.powers().skip(shift).take(vars.len());
            for (&var, alpha) in vars.iter().zip(alphas) {
                let pows = var.powers().take(1 << k).collect();
                out.par_iter_mut()
                    .zip(pows.par_iter())
                    .with_max_len(1 << 14)
                    .for_each(|(acc, &el)| *acc += alpha * el);
            }
        }

        #[test]
        fn test_combine_pows() {
            let mut rng = crate::test::rng(1);
            let alpha: Ext = rng.random();
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);

            let shift = 11;
            for k in k_pack..10 {
                for n in [1, 2, 10, 11] {
                    let vars: Vec<F> = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
                    let mut out0 = Ext::zero_vec(1 << k);
                    combine_pow_ref::<F, Ext>(&mut out0, &vars, alpha, shift);
                    let mut out1 = Ext::zero_vec(1 << k);
                    super::combine_pows(&mut out1, &vars, alpha, shift);
                    assert_eq!(out0, out1);
                    let mut out_packed = PackedExt::zero_vec(1 << (k - k_pack));
                    super::combine_pows_packed(&mut out_packed, &vars, alpha, shift);
                    assert_eq!(out0, unpack::<F, Ext>(&out_packed));
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::p3_field_prelude::*;

    use crate::pcs::test::{make_eq_claims, make_pow_claims};
    use crate::transcript::test_transcript::TestWriter;
    use crate::utils::{VecOps, unpack};
    use crate::{
        pcs::sumcheck::{combine_claims, combine_claims_packed},
        poly::{Point, Poly},
    };
    use p3_util::log2_strict_usize;
    use rand::Rng;

    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;
    // type F = p3_goldilocks::Goldilocks;
    // type Ext = BinomialExtensionField<F, 2>;
    type PackedF = <F as Field>::Packing;
    type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

    fn combine_naive<F: Field, Ext: ExtensionField<F>>(
        k: usize,
        alpha: Ext,
        points: &[Point<Ext>],
        vars: &[F],
    ) -> Vec<Ext> {
        let eqs = points
            .iter()
            .map(|point| point.eq(Ext::ONE))
            .collect::<Vec<_>>();

        let pows = vars
            .iter()
            .map(|&x| x.powers().take(1 << k).collect())
            .collect::<Vec<_>>();

        let mut acc = Ext::zero_vec(1 << k);
        acc.iter_mut()
            .enumerate()
            .for_each(|(i, acc)| *acc += eqs.iter().map(|eq| &eq[i]).horner(alpha));

        let shift = alpha.exp_u64(points.len() as u64);
        acc.iter_mut().enumerate().for_each(|(i, acc)| {
            *acc += pows.iter().map(|pow| &pow[i]).horner_shifted(alpha, shift)
        });

        acc
    }

    #[test]
    fn test_combine_packed() {
        let mut rng = crate::test::rng(1);
        let alpha: Ext = rng.random();

        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        let k_pack = log2_strict_usize(PackedF::WIDTH);
        for k in k_pack..10 {
            let poly = Poly::<Ext>::rand(&mut rng, k).pack::<F>();
            for _ in 0..100 {
                let n_eqs = rng.random_range(0..3);
                let n_pows = rng.random_range(0..3);
                let unpacked = poly.unpack::<F, Ext>();
                let eq_claims =
                    make_eq_claims::<_, F, Ext>(&mut transcript, n_eqs, &unpacked).unwrap();
                let pow_claims =
                    make_pow_claims::<_, F, Ext>(&mut transcript, n_pows, &unpacked).unwrap();

                let points = eq_claims
                    .iter()
                    .map(|c| c.point().clone())
                    .collect::<Vec<_>>();
                let vars = pow_claims.iter().map(|c| c.var()).collect::<Vec<_>>();
                let acc0 = combine_naive(k, alpha, &points, &vars);

                {
                    let mut acc1 = Ext::zero_vec(1 << k);
                    let mut sum = Ext::ZERO;
                    combine_claims::<F, Ext>(&mut acc1, &mut sum, alpha, &eq_claims, &pow_claims);
                    assert_eq!(acc0, acc1);
                }

                {
                    let mut acc1 = PackedExt::zero_vec(1 << (k - k_pack));
                    let mut sum = Ext::ZERO;
                    combine_claims_packed::<F, Ext>(
                        &mut acc1,
                        &mut sum,
                        alpha,
                        &eq_claims,
                        &pow_claims,
                    );
                    assert_eq!(acc0, unpack::<F, Ext>(&acc1));
                }
            }
        }
    }
}
