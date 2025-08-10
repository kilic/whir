use crate::{
    merkle::{to_interleaved_index, verify_merkle_proof, Compress, Hasher},
    transcript::{Reader, Writer},
    utils::log2_strict,
    Error,
};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::fmt::Debug;

use super::MerkleTree;

#[derive(Debug)]
pub struct MatrixCommitmentData<F: Field, Digest> {
    pub(crate) data: RowMajorMatrix<F>,
    pub(crate) layers: Vec<Vec<Digest>>,
}

impl<F: Field, Digest: Clone> MatrixCommitmentData<F, Digest> {
    pub fn k(&self) -> usize {
        log2_strict(self.data.height())
    }

    pub fn root(&self) -> Digest {
        self.layers.last().unwrap().first().unwrap().clone()
    }
}

pub trait MatrixCommitment<F: Field> {
    type Digest: Copy + Debug;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<MatrixCommitmentData<F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>;

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &MatrixCommitmentData<F, Self::Digest>,
    ) -> Result<Vec<F>, Error>
    where
        F: Copy,
        Transcript: Writer<F> + Writer<Self::Digest>;

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Reader<F> + Reader<Self::Digest>;
}

impl<F: Field, Digest, H, C> MatrixCommitment<F> for MerkleTree<F, Digest, H, C>
where
    F: Copy + Clone + Debug + Send + Sync,
    Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    type Digest = Digest;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<MatrixCommitmentData<F, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>,
    {
        let (l, r) = data.split_rows(data.height() / 2);
        let layer0 = l
            .par_row_slices()
            .zip(r.par_row_slices())
            .flat_map(|(l, r)| [self.h.hash_iter(l), self.h.hash_iter(r)])
            .collect::<Vec<_>>();

        let mut layers = vec![layer0];
        for _ in 0..log2_strict(data.height()) {
            let next_layer = layers
                .last()
                .unwrap()
                .par_chunks(2)
                .map(|chunk| self.c.compress(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            layers.push(next_layer);
        }

        let top = layers.last().unwrap();
        debug_assert_eq!(top.len(), 1);
        transcript.write(top[0])?;
        Ok(MatrixCommitmentData { data, layers })
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &MatrixCommitmentData<F, Self::Digest>,
    ) -> Result<Vec<F>, Error>
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

        let mid = 1 << (k - 1);
        let _sibling = comm
            .data
            .row(index ^ mid)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        let index = to_interleaved_index(k, index);

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
                    assert_eq!(self.h.hash_iter(&_sibling), node);
                }
                witness.push(node);
                index_asc >>= 1;
                transcript.write_hint(node)
            })?;

        #[cfg(debug_assertions)]
        {
            let leaf = self.h.hash_iter(&leaf);
            verify_merkle_proof(&self.c, comm.root(), index, leaf, &witness).unwrap();
        }

        Ok(leaf)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: Self::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Reader<F> + Reader<Digest>,
    {
        let row: Vec<F> = transcript.read_hint_many(width)?;
        let leaf = self.h.hash_iter(&row);
        let index = to_interleaved_index(k, index);
        let witness: Vec<Digest> = transcript.read_hint_many(k)?;
        verify_merkle_proof(&self.c, comm, index, leaf, &witness)?;
        Ok(row)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        merkle::{matrix::MatrixCommitment, MerkleTree, RustCrypto},
        transcript::{
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            Reader,
        },
        utils::n_rand,
    };
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn test_mat_com() {
        type F = Goldilocks;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
        type Hasher = RustCrypto<sha3::Keccak256>;
        type Compress = RustCrypto<sha3::Keccak256>;
        let hasher = Hasher::default();
        let compress = Compress::default();

        let mat_comm = MerkleTree::<F, [u8; 32], _, _>::new(hasher, compress);
        let mut transcript = Writer::init("test");

        let k = 3;
        let width = 1;
        let mut rng = crate::test::rng(1);
        let coeffs = n_rand(&mut rng, 1 << k);
        let data = RowMajorMatrix::new(coeffs, width);
        let comm_data = mat_comm.commit(&mut transcript, data).unwrap();
        (0..1 << k).for_each(|index| {
            mat_comm.query(&mut transcript, index, &comm_data).unwrap();
        });

        let proof = transcript.finalize();
        let mut transcript = Reader::init(&proof, "test");
        let comm: [u8; 32] = transcript.read().unwrap();

        (0..1 << k).for_each(|index| {
            mat_comm
                .verify(&mut transcript, comm, index, width, k)
                .unwrap();
        });
    }
}
