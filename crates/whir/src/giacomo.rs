#[cfg(test)]
mod test {
    use common::field::*;
    use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
    use p3_field::TwoAdicField;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use poly::{Point, Poly};
    use rand::RngExt;

    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;

    fn encode_no_transpose<F: TwoAdicField>(
        poly: &Poly<F>,
        rate: usize,
        folding: usize,
    ) -> Poly<F> {
        let split = 1 << (poly.k() - folding);
        let mut mat = RowMajorMatrix::new(poly.to_vec(), poly.len() / split);
        mat.pad_to_height(1 << (poly.k() - folding + rate), F::ZERO);
        Radix2DFTSmallBatch::<F>::default()
            .dft_batch(mat)
            .values
            .into()
    }

    fn combine_pows_dense<F: Field, Ext: ExtensionField<F>>(
        vars: &[F],
        alpha: Ext,
        len: usize,
    ) -> Poly<Ext> {
        let mut out = vec![Ext::ZERO; len];
        vars.iter().zip(alpha.powers()).for_each(|(&var, alpha_i)| {
            out.iter_mut()
                .zip(var.powers().take(len))
                .for_each(|(acc, pow)| *acc += alpha_i * pow);
        });
        out.into()
    }

    fn coeffs<A, B>(poly: &[A], weights: &[B]) -> (B, B)
    where
        A: Copy + Send + Sync + PrimeCharacteristicRing,
        B: Copy + Send + Sync + Algebra<A>,
    {
        let mut c0 = B::ZERO;
        let mut c2 = B::ZERO;
        poly.chunks(2)
            .zip(weights.chunks(2))
            .for_each(|(chunk_poly, chunk_weights)| {
                let (p0, p1) = (chunk_poly[0], chunk_poly[1]);
                let (e0, e1) = (chunk_weights[0], chunk_weights[1]);
                c0 += e0 * p0;
                c2 += (e1.double() - e0) * (p1.double() - p0);
            });

        (c0, c2)
    }

    #[test]
    fn first_round_reversed_pow_c0_c2_from_codeword() {
        let mut rng = common::test::rng(1);

        let k = 6usize;
        let folding = 2usize;
        let rate = 1usize;
        let rs = Point::<F>::rand(&mut rng, folding);

        let f0 = Poly::<F>::rand(&mut rng, k);
        let f1 = f0.compress_lo(&rs, F::ONE);
        let cw0 = encode_no_transpose(&f0, rate, folding);
        let g0 = cw0.compress_lo(&rs, F::ONE);
        assert_eq!(g0, encode_no_transpose(&f1, rate, 0));

        let n = f1.len();
        let n_domain = g0.len();
        let mid = n_domain / 2;
        let two_inv = F::TWO.inverse();

        let f_even = Poly::<F>::new(f1.iter().step_by(2).copied().collect());
        let f_odd = Poly::<F>::new(f1.iter().skip(1).step_by(2).copied().collect());

        let g_even_direct = encode_no_transpose(&f_even, rate, 0);
        let g_odd_direct = encode_no_transpose(&f_odd, rate, 0);
        assert_eq!(g_even_direct.len(), mid);
        assert_eq!(g_odd_direct.len(), mid);

        let omega = F::two_adic_generator(g0.k());
        for index in 0..mid {
            let z = omega.exp_u64(index as u64);
            let even_from_g0 = (g0[index] + g0[index + mid]) * two_inv;
            let odd_from_g0 = (g0[index] - g0[index + mid]) * (two_inv * z.inverse());
            assert_eq!(even_from_g0, g_even_direct[index]);
            assert_eq!(odd_from_g0, g_odd_direct[index]);
        }

        let n_stir = 5usize;
        let mut stir_indices = Vec::with_capacity(n_stir);
        while stir_indices.len() < n_stir {
            let idx = (rng.random::<u64>() as usize) % mid;
            if !stir_indices.contains(&idx) {
                stir_indices.push(idx);
            }
        }
        stir_indices.sort_unstable();

        let alpha: Ext = rng.random();
        let vars = stir_indices
            .iter()
            .map(|&index| omega.exp_u64(index as u64))
            .collect::<Vec<_>>();
        let combined_pow_weights = combine_pows_dense::<F, Ext>(&vars, alpha, n);
        let (dense_c0, dense_c2) = coeffs(&f1, &combined_pow_weights);

        let (lookup_c0, lookup_c2) = stir_indices.iter().zip(alpha.powers()).fold(
            (Ext::ZERO, Ext::ZERO),
            |(acc0, acc2), (&index, alpha_i)| {
                let z_base = omega.exp_u64(index as u64);
                let z = Ext::from(z_base);
                let even = Ext::from((g0[index] + g0[index + mid]) * two_inv);
                let odd = Ext::from((g0[index] - g0[index + mid]) * (two_inv * z_base.inverse()));
                (
                    acc0 + alpha_i * even,
                    acc2 + alpha_i * ((z.double() - Ext::ONE) * (odd.double() - even)),
                )
            },
        );

        assert_eq!(dense_c0, lookup_c0);
        assert_eq!(dense_c2, lookup_c2);
    }
}
