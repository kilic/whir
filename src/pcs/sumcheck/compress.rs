use crate::{
    p3_field_prelude::*,
    pcs::{EqClaim, PowClaim},
    utils::VecOps,
};
use p3_util::log2_strict_usize;

#[tracing::instrument(skip_all)]
pub(crate) fn compress_claims_packed<F: Field, Ext: ExtensionField<F>>(
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

    eq::compress_eqs_packed::<F, Ext>(weights, &points, alpha);
    pow::compress_pows_packed::<F, Ext>(weights, &vars, alpha, points.len());
}

#[tracing::instrument(skip_all)]
pub(crate) fn compress_claims<F: Field, Ext: ExtensionField<F>>(
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

    eq::compress_eqs::<F, Ext>(weights, &points, alpha);
    pow::compress_pows::<F, Ext>(weights, &vars, alpha, points.len());
}

pub(crate) mod eq {
    use crate::p3_field_prelude::*;
    use crate::{poly::Point, utils::TwoAdicSlice};
    use p3_util::log2_strict_usize;
    use rayon::prelude::*;

    fn flat_eqs<F: Field, Ext: ExtensionField<F>>(points: &[Point<Ext>], alpha: Ext) -> Vec<Ext> {
        let k = points[0].len();
        let n = points.len();

        let mut acc = Ext::zero_vec(n * (1 << k));
        acc[..n].copy_from_slice(&alpha.powers().take(n).collect());
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

    fn packed_flat_eqs<F: Field, Ext: ExtensionField<F>>(
        points: &[Point<Ext>],
        alpha: Ext,
    ) -> Vec<Ext::ExtensionPacking> {
        assert!(!points.is_empty());
        let k = points[0].len();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let n = points.len();

        let mut acc_init = Ext::zero_vec(n * (1 << k_pack));
        acc_init[..n].copy_from_slice(&alpha.powers().take(n).collect());
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
        let mut acc_packed = Ext::ExtensionPacking::zero_vec(n * (1 << (k - k_pack)));
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

    #[tracing::instrument(skip_all, fields(n = points.len(), k = out.k() + log2_strict_usize(F::Packing::WIDTH)))]
    pub(crate) fn compress_eqs_packed<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext::ExtensionPacking],
        points: &[Point<Ext>],
        alpha: Ext,
    ) {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = out.k() + k_pack;
        assert!(k_pack * 2 <= k);

        if points.is_empty() {
            return;
        }
        points.iter().for_each(|point| assert_eq!(point.len(), k));

        let mid = k / 2;
        let left = points
            .iter()
            .map(|point| point.split_at(mid).0)
            .collect::<Vec<_>>();
        let right = points
            .iter()
            .map(|point| point.split_at(mid).1)
            .collect::<Vec<_>>();
        let left = packed_flat_eqs::<F, Ext>(&left, alpha);
        let right = flat_eqs(&right, Ext::ONE);

        let n = points.len();
        out.par_chunks_mut(left.len() / n)
            .zip_eq(right.par_chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.chunks(n)).for_each(|(out, left)| {
                    *out += left
                        .iter()
                        .zip(right.iter())
                        .map(|(&left, &right)| left * right)
                        .sum::<Ext::ExtensionPacking>();
                });
            });
    }

    #[tracing::instrument(skip_all, fields(n = points.len(), k = out.k()))]
    pub(crate) fn compress_eqs<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext],
        points: &[Point<Ext>],
        alpha: Ext,
    ) {
        if points.is_empty() {
            return;
        }

        let k = out.k();
        points.iter().for_each(|point| assert_eq!(point.len(), k));

        let mid = k / 2;
        let left = points
            .iter()
            .map(|point| point.split_at(mid).0)
            .collect::<Vec<_>>();
        let left = flat_eqs::<F, Ext>(&left, alpha);
        let right = points
            .iter()
            .map(|point| point.split_at(mid).1)
            .collect::<Vec<_>>();
        let right = flat_eqs(&right, Ext::ONE);

        let n = points.len();
        out.par_chunks_mut(left.len() / n)
            .zip_eq(right.par_chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.chunks(n)).for_each(|(out, left)| {
                    *out += left
                        .iter()
                        .zip(right.iter())
                        .map(|(&left, &right)| left * right)
                        .sum::<Ext>();
                });
            });
    }

    #[cfg(test)]
    mod test {
        use crate::poly::Point;
        use crate::utils::{unpack, TwoAdicSlice};
        use p3_field::{extension::BinomialExtensionField, PackedValue};
        use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
        use p3_util::log2_strict_usize;
        use rand::Rng;
        use rayon::prelude::*;

        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

        #[tracing::instrument(skip_all, fields(points = points.len()))]
        pub fn compress_eq_ref<F: Field, Ext: ExtensionField<F>>(
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
                    .with_min_len(1 << 14)
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
                .with_min_len(1 << 14)
                .for_each(|(row, out)| *out += row.iter().cloned().sum::<Ext>());
        }

        #[test]
        fn test_compress_eqs() {
            let mut rng = crate::test::rng(1);
            let alpha: Ext = rng.random();
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);

            for k in 4..10 {
                for n in [1, 2, 10] {
                    let points = (0..n)
                        .map(|_| Point::<Ext>::rand(&mut rng, k))
                        .collect::<Vec<_>>();
                    let mut out0 = Ext::zero_vec(1 << k);
                    compress_eq_ref::<F, Ext>(&mut out0, &points, alpha);
                    let mut out1 = Ext::zero_vec(1 << k);
                    super::compress_eqs::<F, Ext>(&mut out1, &points, alpha);
                    assert_eq!(out0, out1);
                    let mut out_packed = PackedExt::zero_vec(1 << (k - k_pack));
                    super::compress_eqs_packed::<F, Ext>(&mut out_packed, &points, alpha);
                    assert_eq!(out0, unpack::<F, Ext>(&out_packed));
                }
            }
        }
    }
}

pub(crate) mod pow {
    use crate::p3_field_prelude::*;
    use crate::{poly::Point, utils::TwoAdicSlice};
    use p3_util::log2_strict_usize;
    use rayon::prelude::*;

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

    fn flat_pows<F: Field>(points: &[Point<F>]) -> Vec<F> {
        let k = points[0].len();
        let n = points.len();

        let mut acc = F::zero_vec(n * (1 << k));
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

    fn packed_flat_pows<F: Field>(points: &[Point<F>]) -> Vec<F::Packing> {
        let k = points[0].len();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let n = points.len();

        let mut acc_init: Vec<F> = F::zero_vec(n * (1 << k_pack));
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
        let mut acc_packed = F::Packing::zero_vec((1 << (k - k_pack)) * n);
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

    #[tracing::instrument(skip_all, fields(vars = vars.len(), k = out.k() + log2_strict_usize(F::Packing::WIDTH)))]
    pub(crate) fn compress_pows_packed<F: Field, Ext: ExtensionField<F>>(
        out: &mut [Ext::ExtensionPacking],
        vars: &[F],
        alpha: Ext,
        shift: usize,
    ) {
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k = out.k() + k_pack;
        assert!(k_pack * 2 <= k);

        if vars.is_empty() {
            return;
        }
        let points = binary_powers(vars, k);

        let mid = k / 2;
        let left = points
            .iter()
            .map(|point| point.split_at(mid).0)
            .collect::<Vec<_>>();
        let right = points
            .iter()
            .map(|point| point.split_at(mid).1)
            .collect::<Vec<_>>();
        let left = packed_flat_pows(&left);
        let right = flat_pows(&right);

        let n = vars.len();
        let alphas = alpha
            .powers()
            .skip(shift)
            .take(n)
            .map(|alpha| Ext::ExtensionPacking::from_ext_slice(&vec![alpha; F::Packing::WIDTH]))
            .collect::<Vec<_>>();

        out.par_chunks_mut(left.len() / n)
            .zip_eq(right.par_chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.chunks(n)).for_each(|(out, left)| {
                    *out += left
                        .iter()
                        .zip(right.iter())
                        .zip(alphas.iter())
                        .map(|((&left, &right), &alpha)| alpha * (left * right))
                        .sum::<Ext::ExtensionPacking>();
                });
            });
    }

    #[tracing::instrument(skip_all, fields(vars = vars.len(), k = out.k()))]
    pub(crate) fn compress_pows<F: Field, Ext: ExtensionField<F>>(
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

        let mid = k / 2;
        let points = binary_powers(vars, k);
        let left = points
            .iter()
            .map(|point| point.split_at(mid).0)
            .collect::<Vec<_>>();
        let left = flat_pows(&left);
        let right = points
            .iter()
            .map(|point| point.split_at(mid).1)
            .collect::<Vec<_>>();
        let right = flat_pows(&right);

        let alphas = alpha.powers().skip(shift).take(n).collect::<Vec<_>>();
        out.par_chunks_mut(left.len() / n)
            .zip_eq(right.par_chunks(n))
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.chunks(n)).for_each(|(out, left)| {
                    *out += left
                        .iter()
                        .zip(right.iter())
                        .zip(alphas.iter())
                        .map(|((&left, &right), &alpha)| alpha * (left * right))
                        .sum::<Ext>();
                });
            });
    }

    #[cfg(test)]
    mod test {
        use crate::utils::{unpack, TwoAdicSlice};
        use p3_field::{extension::BinomialExtensionField, PackedValue};
        use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
        use p3_util::log2_strict_usize;
        use rand::Rng;
        use rayon::prelude::*;

        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

        #[tracing::instrument(skip_all)]
        pub fn compress_pow_ref<F: Field, Ext: ExtensionField<F>>(
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
        fn test_compress_pows() {
            let mut rng = crate::test::rng(1);
            let alpha: Ext = rng.random();
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);

            let shift = 11;
            for k in 4..10 {
                for n in [1, 2, 10] {
                    let vars: Vec<F> = (0..n).map(|_| rng.random()).collect::<Vec<_>>();
                    let mut out0 = Ext::zero_vec(1 << k);
                    compress_pow_ref::<F, Ext>(&mut out0, &vars, alpha, shift);
                    let mut out1 = Ext::zero_vec(1 << k);
                    super::compress_pows(&mut out1, &vars, alpha, shift);
                    assert_eq!(out0, out1);
                    let mut out_packed = PackedExt::zero_vec(1 << (k - k_pack));
                    super::compress_pows_packed(&mut out_packed, &vars, alpha, shift);
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
    use crate::utils::{unpack, VecOps};
    use crate::{
        pcs::sumcheck::{compress_claims, compress_claims_packed},
        poly::{Point, Poly},
    };
    use p3_util::log2_strict_usize;
    use rand::Rng;

    type F = p3_koala_bear::KoalaBear;
    type PackedF = <F as Field>::Packing;
    type Ext = BinomialExtensionField<F, 4>;
    type PackedExt = <Ext as ExtensionField<F>>::ExtensionPacking;

    fn compress_naive<F: Field, Ext: ExtensionField<F>>(
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
    fn test_compress_packed() {
        let mut rng = crate::test::rng(1);
        let alpha: Ext = rng.random();

        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        let k_pack = log2_strict_usize(PackedF::WIDTH);
        for k in 4..10 {
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
                let acc0 = compress_naive(k, alpha, &points, &vars);

                {
                    let mut acc1 = Ext::zero_vec(1 << k);
                    let mut sum = Ext::ZERO;
                    compress_claims::<F, Ext>(&mut acc1, &mut sum, alpha, &eq_claims, &pow_claims);
                    assert_eq!(acc0, acc1);
                }

                {
                    let mut acc1 = PackedExt::zero_vec(1 << (k - k_pack));
                    let mut sum = Ext::ZERO;
                    compress_claims_packed::<F, Ext>(
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
