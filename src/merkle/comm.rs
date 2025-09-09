use crate::{
    merkle::{verify_merkle_proof, Compress, Hasher, MerkleTree},
    transcript::{Reader, Writer},
    utils::log2_strict,
    Error,
};
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    extension::FlatMatrixView,
    Matrix,
};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::fmt::Debug;

#[derive(Debug)]
pub struct CommitmentData<F: Field, Ext: ExtensionField<F>, Digest> {
    pub(crate) data: FlatMatrixView<F, Ext, DenseMatrix<Ext>>,
    pub(crate) layers: Vec<Vec<Digest>>,
}

impl<F: Field, Ext: ExtensionField<F>, Digest: Clone> CommitmentData<F, Ext, Digest> {
    pub fn k(&self) -> usize {
        log2_strict(self.data.height())
    }

    pub fn root(&self) -> Digest {
        self.layers.last().unwrap().first().unwrap().clone()
    }
}

pub trait Commitment<F: Field> {
    type Digest: Copy + Debug;

    fn commit_base<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<CommitmentData<F, F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>;

    fn commit_ext<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<CommitmentData<F, Ext, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>;

    fn query<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &CommitmentData<F, Ext, Self::Digest>,
    ) -> Result<Vec<Ext>, Error>
    where
        F: Copy,
        Transcript: Writer<F> + Writer<Self::Digest>;

    fn verify<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Reader<Ext> + Reader<Self::Digest>;
}

impl<F: Field, Digest, H, C> Commitment<F> for MerkleTree<F, Digest, H, C>
where
    Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    type Digest = Digest;
    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit_base<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<CommitmentData<F, F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>,
    {
        let mut layers = vec![self.hasher.hash_base_data(&data)];
        tracing::info_span!("ascend tree").in_scope(|| {
            for _ in 0..log2_strict(data.height()) {
                let next_layer = layers
                    .last()
                    .unwrap()
                    .par_chunks(2)
                    .map(|chunk| self.compress.apply(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                layers.push(next_layer);
            }
        });

        let top = layers.last().unwrap();
        debug_assert_eq!(top.len(), 1);
        transcript.write(top[0])?;
        Ok(CommitmentData {
            data: FlatMatrixView::new(data),
            layers,
        })
    }

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit_ext<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<CommitmentData<F, Ext, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>,
    {
        let data = FlatMatrixView::new(data);
        let mut layers = vec![self.hasher.hash_ext_data(&data)];
        tracing::info_span!("ascend tree").in_scope(|| {
            for _ in 0..log2_strict(data.height()) {
                let next_layer = layers
                    .last()
                    .unwrap()
                    .par_chunks(2)
                    .map(|chunk| self.compress.apply(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                layers.push(next_layer);
            }
        });

        let top = layers.last().unwrap();
        debug_assert_eq!(top.len(), 1);
        transcript.write(top[0])?;
        Ok(CommitmentData { data, layers })
    }

    fn query<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &CommitmentData<F, Ext, Self::Digest>,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Writer<F> + Writer<Digest>,
    {
        let k = comm.k();
        let leaf = comm
            .data
            .row(index)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        transcript.write_hint_many(&leaf)?;

        let mut witness = vec![];
        let mut index_asc = index;

        comm.layers
            .iter()
            .take(k)
            .enumerate()
            .try_for_each(|(_i, layer)| {
                let node = layer[index_asc ^ 1];
                #[cfg(debug_assertions)]
                if _i == 0 {
                    let sibling = comm.data.row(index ^ 1).unwrap().into_iter();
                    debug_assert_eq!(self.hasher.hash_base_iter(sibling.into_iter()), node);
                }
                witness.push(node);
                index_asc >>= 1;
                transcript.write_hint(node)
            })?;

        #[cfg(debug_assertions)]
        {
            let leaf = self.hasher.hash_base_iter(leaf.iter().cloned());
            verify_merkle_proof(&self.compress, comm.root(), index, leaf, &witness).unwrap();
        }

        Ok(leaf
            .chunks(Ext::DIMENSION)
            .map(|e| Ext::from_basis_coefficients_slice(e).unwrap())
            .collect::<Vec<_>>())
    }

    fn verify<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Reader<Ext> + Reader<Digest>,
    {
        let row: Vec<Ext> = transcript.read_hint_many(width)?;
        let leaf = self.hasher.hash_ext_iter(row.iter().cloned());
        let witness: Vec<Digest> = transcript.read_hint_many(k)?;
        verify_merkle_proof(&self.compress, comm, index, leaf, &witness)?;
        Ok(row)
    }
}

#[cfg(test)]
mod test {
    use std::{fmt::Debug, ops::Range};

    use crate::{
        merkle::{
            comm::Commitment,
            poseidon::{PoseidonCompress, PoseidonHasher},
            rust_crypto::{RustCryptoCompress, RustCryptoHasher},
            Compress, Hasher, MerkleTree,
        },
        transcript::{
            poseidon::{PoseidonReader, PoseidonWriter},
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            Reader, Writer,
        },
        utils::n_rand,
    };
    use p3_challenger::DuplexChallenger;
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::Poseidon2KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::PaddingFreeSponge;
    use rand::distr::{Distribution, StandardUniform};

    fn prover_base_commit<F: Field, Transcript, Digest, H, C>(
        mut rng: impl rand::RngCore,
        transcript: &mut Transcript,
        hasher: H,
        compress: C,
        width: Range<usize>,
        k: usize,
    ) where
        Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
        H: Hasher<F, Digest>,
        C: Compress<Digest, 2>,
        StandardUniform: Distribution<F>,
        Transcript: Writer<Digest> + Writer<F> + Clone,
    {
        let merkle_tree = MerkleTree::<F, Digest, _, _>::new(hasher, compress);
        for width in width {
            let coeffs = n_rand(&mut rng, (1 << k) * width);
            let data = RowMajorMatrix::new(coeffs, width);
            let comm0 = merkle_tree
                .commit_base::<_>(&mut transcript.clone(), data.clone())
                .unwrap();
            let comm1 = merkle_tree.commit_ext::<_, F>(transcript, data).unwrap();
            assert_eq!(comm0.root(), comm1.root());

            (0..1 << k).for_each(|index| {
                merkle_tree.query(transcript, index, &comm0).unwrap();
            });
        }
    }

    fn prover_ext_commit<F: Field, Ext: ExtensionField<F>, Transcript, Digest, H, C>(
        mut rng: impl rand::RngCore,
        transcript: &mut Transcript,
        hasher: H,
        compress: C,
        width: Range<usize>,
        k: usize,
    ) where
        Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
        H: Hasher<F, Digest>,
        C: Compress<Digest, 2>,
        StandardUniform: Distribution<Ext>,
        Transcript: Writer<Digest> + Writer<F>,
    {
        let comm = MerkleTree::<F, Digest, _, _>::new(hasher, compress);
        for width in width {
            let coeffs = n_rand(&mut rng, (1 << k) * width);
            let data = RowMajorMatrix::new(coeffs, width);
            let comm_data = comm.commit_ext::<_, Ext>(transcript, data).unwrap();
            (0..1 << k).for_each(|index| {
                comm.query(transcript, index, &comm_data).unwrap();
            });
        }
    }

    fn verifier<F: Field, Ext: ExtensionField<F>, Transcript, Digest, H, C>(
        transcript: &mut Transcript,
        hasher: H,
        compress: C,
        width: Range<usize>,
        k: usize,
    ) where
        Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
        H: Hasher<F, Digest>,
        C: Compress<Digest, 2>,
        Transcript: Reader<Digest> + Reader<Ext>,
    {
        let merkle_tree = MerkleTree::<F, Digest, _, _>::new(hasher, compress);
        for width in width {
            let comm: Digest = transcript.read().unwrap();
            (0..1 << k).for_each(|index| {
                merkle_tree
                    .verify::<_, Ext>(transcript, comm, index, width, k)
                    .unwrap();
            });
        }
    }

    #[test]
    fn test_comm_rust_crypto() {
        for k in 1..8 {
            {
                type F = Goldilocks;
                type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
                type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
                let mut transcript = Writer::init("test");
                prover_base_commit::<F, _, _, _, _>(
                    &mut crate::test::rng(0),
                    &mut transcript,
                    RustCryptoHasher::<sha3::Keccak256>::default(),
                    RustCryptoCompress::<sha3::Keccak256>::default(),
                    1..8,
                    k,
                );
                let proof = transcript.finalize();
                let mut transcript = Reader::init(&proof, "test");
                verifier::<F, F, _, _, _, _>(
                    &mut transcript,
                    RustCryptoHasher::<sha3::Keccak256>::default(),
                    RustCryptoCompress::<sha3::Keccak256>::default(),
                    1..8,
                    k,
                );
            }

            {
                type F = Goldilocks;
                type Ext = BinomialExtensionField<F, 2>;
                type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
                type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
                let mut transcript = Writer::init("test");
                prover_ext_commit::<F, Ext, _, _, _, _>(
                    &mut crate::test::rng(0),
                    &mut transcript,
                    RustCryptoHasher::<sha3::Keccak256>::default(),
                    RustCryptoCompress::<sha3::Keccak256>::default(),
                    1..8,
                    k,
                );
                let proof = transcript.finalize();
                let mut transcript = Reader::init(&proof, "test");
                verifier::<F, Ext, _, _, _, _>(
                    &mut transcript,
                    RustCryptoHasher::<sha3::Keccak256>::default(),
                    RustCryptoCompress::<sha3::Keccak256>::default(),
                    1..8,
                    k,
                );
            }
        }
    }

    #[test]
    fn test_comm_poseidon_koala_bear() {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = PoseidonCompress<F, Poseidon16, 2, 8, 16>;
        type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;

        type Writer = PoseidonWriter<Vec<u8>, F, Challenger>;
        type Reader<'a> = PoseidonReader<&'a [u8], F, Challenger>;

        let perm16 = Poseidon16::new_from_rng_128(&mut crate::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut crate::test::rng(1000));

        let hasher = PoseidonHasher::<F, _, 8>::new(Hasher::new(perm24));
        let compress = Compress::new(perm16.clone());
        let challenger = Challenger::new(perm16);

        let mut rng = crate::test::rng(0);
        for k in 1..8 {
            {
                let mut transcript = Writer::init(challenger.clone());
                prover_base_commit::<F, _, _, _, _>(
                    &mut rng,
                    &mut transcript,
                    hasher.clone(),
                    compress.clone(),
                    1..8,
                    k,
                );

                let proof = transcript.finalize();
                let mut transcript = Reader::init(&proof, challenger.clone());
                verifier::<F, F, _, _, _, _>(
                    &mut transcript,
                    hasher.clone(),
                    compress.clone(),
                    1..8,
                    k,
                );
            }

            {
                let mut transcript = Writer::init(challenger.clone());
                prover_ext_commit::<F, Ext, _, _, _, _>(
                    &mut rng,
                    &mut transcript,
                    hasher.clone(),
                    compress.clone(),
                    1..8,
                    k,
                );

                let proof = transcript.finalize();
                let mut transcript = Reader::init(&proof, challenger.clone());
                verifier::<F, Ext, _, _, _, _>(
                    &mut transcript,
                    hasher.clone(),
                    compress.clone(),
                    1..8,
                    k,
                );
            }
        }
    }
}
