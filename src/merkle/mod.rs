use crate::Error;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    extension::FlatMatrixView,
};
use rayon::iter::ParallelIterator;
use std::fmt::Debug;

pub mod comm;
pub mod poseidon;
pub mod rust_crypto;

pub trait Compress<T, const N: usize>: Sync {
    fn apply(&self, input: [T; N]) -> T;
}

pub trait Hasher<F: Field, Out>: Sync {
    fn hash_base_data(&self, data: &RowMajorMatrix<F>) -> Vec<Out>;
    fn hash_ext_data<Ext: ExtensionField<F>>(
        &self,
        data: &FlatMatrixView<F, Ext, DenseMatrix<Ext>>,
    ) -> Vec<Out>;
    fn hash_base_iter<I>(&self, row: I) -> Out
    where
        I: IntoIterator<Item = F>;
    fn hash_ext_iter<I, Ext: ExtensionField<F>>(&self, row: I) -> Out
    where
        I: IntoIterator<Item = Ext>;
}

pub(super) fn hash_reference_base<F: Field, Out: Send + Sync, H: Hasher<F, Out> + Sync>(
    data: &RowMajorMatrix<F>,
    hasher: &H,
) -> Vec<Out> {
    data.par_row_slices()
        .map(|els| hasher.hash_base_iter(els.to_vec()))
        .collect::<Vec<_>>()
}

pub(super) fn hash_reference_ext<
    F: Field,
    Ext: ExtensionField<F>,
    Out: Send + Sync,
    H: Hasher<F, Out> + Sync,
>(
    data: &RowMajorMatrix<Ext>,
    hasher: &H,
) -> Vec<Out> {
    data.par_row_slices()
        .map(|ext| {
            let els = Ext::flatten_to_base(ext.to_vec());
            hasher.hash_base_iter(els)
        })
        .collect::<Vec<_>>()
}

pub fn verify_merkle_proof<C, Node>(
    c: &C,
    claim: Node,
    mut index: usize,
    leaf: Node,
    witness: &[Node],
) -> Result<(), Error>
where
    C: Compress<Node, 2>,
    Node: Copy + Clone + Sync + Debug + Eq + PartialEq,
{
    assert!(index < 1 << witness.len());
    let found = witness.iter().fold(leaf, |acc, &w| {
        let acc = c.apply(if index & 1 == 1 { [w, acc] } else { [acc, w] });
        index >>= 1;
        acc
    });
    (claim == found).then_some(()).ok_or(Error::Verify)
}

pub struct MerkleTree<F: Field, Digest, H: Hasher<F, Digest>, C: Compress<Digest, 2>> {
    pub(crate) hasher: H,
    pub(crate) compress: C,
    pub(crate) _phantom: std::marker::PhantomData<(F, Digest)>,
}

impl<F, Digest, H, C> MerkleTree<F, Digest, H, C>
where
    F: Field,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    pub fn new(hasher: H, compress: C) -> Self {
        Self {
            hasher,
            compress,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod test {

    use crate::merkle::poseidon::PoseidonHasher;
    use crate::merkle::rust_crypto::RustCryptoHasher;
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

    fn run_test_hash<
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
            run_test_hash::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));

            let perm = Poseidon2KoalaBear::<24>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 24, 16, 8> = PaddingFreeSponge::new(perm);
            run_test_hash::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));
        }

        {
            type F = p3_baby_bear::BabyBear;
            type Ext = BinomialExtensionField<F, 4>;

            let perm = Poseidon2BabyBear::<16>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 16, 8, 8> = PaddingFreeSponge::new(perm);
            run_test_hash::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));

            let perm = Poseidon2BabyBear::<24>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 24, 16, 8> = PaddingFreeSponge::new(perm);
            run_test_hash::<F, Ext, _, _>(&PoseidonHasher::<F, _, 8>::new(hasher));
        }

        {
            type F = p3_goldilocks::Goldilocks;
            type Ext = BinomialExtensionField<F, 2>;

            let perm = Poseidon2Goldilocks::<8>::new_from_rng_128(&mut crate::test::rng(0));
            let hasher: PaddingFreeSponge<_, 8, 4, 4> = PaddingFreeSponge::new(perm);
            run_test_hash::<F, Ext, _, _>(&PoseidonHasher::<F, _, 4>::new(hasher));
        }
    }

    #[test]
    fn test_hash_rust_crypto() {
        {
            type F = p3_koala_bear::KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;
            let hasher = RustCryptoHasher::<sha2::Sha256>::default();
            run_test_hash::<F, Ext, _, _>(&hasher);
        }

        {
            type F = p3_baby_bear::BabyBear;
            type Ext = BinomialExtensionField<F, 4>;
            let hasher = RustCryptoHasher::<sha2::Sha256>::default();
            run_test_hash::<F, Ext, _, _>(&hasher);
        }

        {
            type F = p3_koala_bear::KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;
            let hasher = RustCryptoHasher::<sha2::Sha256>::default();
            run_test_hash::<F, Ext, _, _>(&hasher);
        }
    }
}
