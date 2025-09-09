use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    extension::FlatMatrixView,
    Matrix,
};
use p3_symmetric::{
    CryptographicHasher, CryptographicPermutation, PseudoCompressionFunction, TruncatedPermutation,
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::fmt::Debug;

use crate::merkle::{Compress, Hasher};

#[derive(Debug, Clone)]
pub struct PoseidonHasher<F, H, const OUT: usize> {
    h: H,
    _marker: std::marker::PhantomData<F>,
}

impl<F, H, const OUT: usize> PoseidonHasher<F, H, OUT> {
    pub fn new(h: H) -> PoseidonHasher<F, H, OUT> {
        PoseidonHasher {
            h,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        F: Field,
        H: CryptographicHasher<F, [F; OUT]>
            + CryptographicHasher<F::Packing, [F::Packing; OUT]>
            + Sync,
        const OUT: usize,
    > Hasher<F, [F; OUT]> for PoseidonHasher<F, H, OUT>
{
    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    // adapted from p3
    fn hash_base_data(&self, data: &RowMajorMatrix<F>) -> Vec<[F; OUT]> {
        use p3_field::PackedValue;
        fn unpack_array<P: PackedValue, const N: usize>(
            packed_digest: [P; N],
        ) -> impl Iterator<Item = [P::Value; N]> {
            (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
        }

        if data.height() < F::Packing::WIDTH {
            super::hash_reference_base(data, self)
        } else {
            let mut res = vec![[F::ZERO; OUT]; data.height()];
            res.par_chunks_exact_mut(F::Packing::WIDTH)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let packed_digest: [F::Packing; OUT] = self
                        .h
                        .hash_iter(data.vertically_packed_row::<F::Packing>(i * F::Packing::WIDTH));

                    for (dst, src) in chunk.iter_mut().zip(unpack_array(packed_digest)) {
                        *dst = src;
                    }
                });

            res
        }
    }

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    // adapted from p3
    fn hash_ext_data<Ext: ExtensionField<F>>(
        &self,
        data: &FlatMatrixView<F, Ext, DenseMatrix<Ext>>,
    ) -> Vec<[F; OUT]> {
        use p3_field::PackedValue;
        fn unpack_array<P: PackedValue, const N: usize>(
            packed_digest: [P; N],
        ) -> impl Iterator<Item = [P::Value; N]> {
            (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
        }
        if data.height() < F::Packing::WIDTH {
            super::hash_reference_ext(data, self)
        } else {
            let mut res = vec![[F::ZERO; OUT]; data.height()];
            res.par_chunks_exact_mut(F::Packing::WIDTH)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let packed_digest: [F::Packing; OUT] = self
                        .h
                        .hash_iter(data.vertically_packed_row::<F::Packing>(i * F::Packing::WIDTH));

                    for (dst, src) in chunk.iter_mut().zip(unpack_array(packed_digest)) {
                        *dst = src;
                    }
                });

            res
        }
    }

    fn hash_base_iter<I>(&self, input: I) -> [F; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        self.h.hash_iter(input)
    }

    fn hash_ext_iter<I, Ext: ExtensionField<F>>(&self, input: I) -> [F; OUT]
    where
        I: IntoIterator<Item = Ext>,
    {
        let input = input
            .into_iter()
            .flat_map(|ext| ext.as_basis_coefficients_slice().to_vec());
        self.h.hash_iter(input)
    }
}

#[derive(Debug, Clone)]
pub struct PoseidonCompress<F: Field, Perm, const N: usize, const CHUNK: usize, const WIDTH: usize>
{
    poseidon: TruncatedPermutation<Perm, N, CHUNK, WIDTH>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Perm, const N: usize, const CHUNK: usize, const WIDTH: usize>
    PoseidonCompress<F, Perm, N, CHUNK, WIDTH>
{
    pub fn new(perm: Perm) -> PoseidonCompress<F, Perm, N, CHUNK, WIDTH> {
        PoseidonCompress {
            poseidon: TruncatedPermutation::new(perm),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<
        F: Field,
        Perm: CryptographicPermutation<[F; WIDTH]>,
        const N: usize,
        const CHUNK: usize,
        const WIDTH: usize,
    > Compress<[F; CHUNK], N> for PoseidonCompress<F, Perm, N, CHUNK, WIDTH>
{
    fn apply(&self, input: [[F; CHUNK]; N]) -> [F; CHUNK] {
        self.poseidon.compress(input)
    }
}

#[cfg(test)]
mod test {

    use crate::merkle::poseidon::PoseidonHasher;
    use crate::merkle::{hash_reference_base, hash_reference_ext, Hasher};
    use crate::utils::n_rand;
    use p3_baby_bear::Poseidon2BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field};
    use p3_goldilocks::Poseidon2Goldilocks;
    use p3_koala_bear::Poseidon2KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::extension::FlatMatrixView;
    use p3_symmetric::PaddingFreeSponge;
    use rand::distr::{Distribution, StandardUniform};
    use std::fmt::Debug;

    fn run_test_hash_base<F: Field, H: Hasher<F, Out>, Out: Debug + PartialEq + Send + Sync>(
        k: usize,
        folding: usize,
        hasher: &H,
    ) where
        StandardUniform: Distribution<F>,
    {
        let mut rng = crate::test::rng(0);
        let values: Vec<F> = n_rand::<F>(&mut rng, 1 << k);
        let data = RowMajorMatrix::new(values, 1 << folding);
        let l0 = hash_reference_ext::<F, F, _, _>(&data, hasher);
        let l1 = hash_reference_base::<F, _, _>(&data, hasher);
        assert_eq!(l0, l1);
        let l1 = hasher.hash_base_data(&data);
        assert_eq!(l0, l1);
    }

    fn run_test_hash_ext<
        F: Field,
        Ext: ExtensionField<F>,
        H: Hasher<F, Out>,
        Out: Debug + PartialEq + Send + Sync,
    >(
        k: usize,
        folding: usize,
        hasher: &H,
    ) where
        StandardUniform: Distribution<Ext>,
    {
        let mut rng = crate::test::rng(0);
        let values: Vec<Ext> = n_rand::<Ext>(&mut rng, 1 << k);
        let data = RowMajorMatrix::new(values, 1 << folding);
        let l0 = hash_reference_ext::<F, Ext, _, _>(&data, hasher);
        let data = FlatMatrixView::<F, Ext, _>::new(data);
        let l1 = hasher.hash_ext_data(&data);
        assert_eq!(l0, l1);
    }

    fn run_test_hash_poseidon<
        F: Field,
        Ext: ExtensionField<F>,
        H: Hasher<F, Out>,
        Out: Debug + PartialEq + Send + Sync,
    >(
        hasher: &H,
    ) where
        StandardUniform: Distribution<F>,
        StandardUniform: Distribution<Ext>,
    {
        for k in 0..10 {
            for folding in 0..=k {
                run_test_hash_base(k, folding, hasher);
            }
        }

        for k in 0..10 {
            for folding in 0..=k {
                run_test_hash_ext::<F, F, H, _>(k, folding, hasher);
                run_test_hash_ext::<F, Ext, H, _>(k, folding, hasher);
            }
        }
    }
    #[test]
    fn test_hash_poseidon() {
        {
            type F = p3_koala_bear::KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;

            let perm = Poseidon2KoalaBear::<16>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 16, 8, 8> = PaddingFreeSponge::new(perm);
            run_test_hash_poseidon::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));

            let perm = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 24, 16, 8> = PaddingFreeSponge::new(perm);
            run_test_hash_poseidon::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));
        }

        {
            type F = p3_baby_bear::BabyBear;
            type Ext = BinomialExtensionField<F, 4>;

            let perm = Poseidon2BabyBear::<16>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 16, 8, 8> = PaddingFreeSponge::new(perm);
            run_test_hash_poseidon::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));

            let perm = Poseidon2BabyBear::<24>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 24, 16, 8> = PaddingFreeSponge::new(perm);
            run_test_hash_poseidon::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));
        }

        {
            type F = p3_goldilocks::Goldilocks;
            type Ext = BinomialExtensionField<F, 2>;

            let perm = Poseidon2Goldilocks::<8>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 8, 4, 4> = PaddingFreeSponge::new(perm);
            run_test_hash_poseidon::<F, Ext, _, _>(&PoseidonHasher::<F, _, 4>::new(hasher));
        }
    }
}
