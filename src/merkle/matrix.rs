use crate::{
    merkle::{to_interleaved_index, verify_merkle_proof, Compress, Hasher},
    transcript::{Reader, Writer},
    utils::log2_strict,
    Error,
};
use p3_field::{ExtensionField, Field};
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

pub trait MatrixCommitment<F: Field, Ext: ExtensionField<F>> {
    type Digest: Copy + Debug;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<MatrixCommitmentData<Ext, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>;

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        comm: &MatrixCommitmentData<Ext, Self::Digest>,
    ) -> Result<Vec<Ext>, Error>
    where
        F: Copy,
        Transcript: Writer<Ext> + Writer<Self::Digest>;

    fn verify<Transcript>(
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

impl<F: Field, Ext: ExtensionField<F>, Digest, H, C> MatrixCommitment<F, Ext>
    for MerkleTree<F, Ext, Digest, H, C>
where
    Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq,
    H: Hasher<F, Digest>,
    C: Compress<Digest, 2>,
{
    type Digest = Digest;
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<MatrixCommitmentData<Ext, Self::Digest>, Error>
    where
        Transcript: Writer<Self::Digest>,
    {
        let (l, r) = data.split_rows(data.height() / 2);
        let layer0 = l
            .par_row_slices()
            .zip(r.par_row_slices())
            .flat_map(|(l, r)| {
                let l = l.iter().flat_map(|e| e.as_basis_coefficients_slice());
                let r = r.iter().flat_map(|e| e.as_basis_coefficients_slice());
                [self.h.hash_iter(l), self.h.hash_iter(r)]
            })
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
        comm: &MatrixCommitmentData<Ext, Self::Digest>,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Writer<Ext> + Writer<Digest>,
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
                    let sibling = _sibling
                        .iter()
                        .flat_map(|e| e.as_basis_coefficients_slice());
                    assert_eq!(self.h.hash_iter(sibling), node);
                }
                witness.push(node);
                index_asc >>= 1;
                transcript.write_hint(node)
            })?;

        #[cfg(debug_assertions)]
        {
            let leaf = leaf.iter().flat_map(|e| e.as_basis_coefficients_slice());
            let leaf = self.h.hash_iter(leaf);
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
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Reader<Ext> + Reader<Digest>,
    {
        let row: Vec<Ext> = transcript.read_hint_many(width)?;
        let leaf = self
            .h
            .hash_iter(row.iter().flat_map(|e| e.as_basis_coefficients_slice()));
        let index = to_interleaved_index(k, index);
        let witness: Vec<Digest> = transcript.read_hint_many(k)?;
        verify_merkle_proof(&self.c, comm, index, leaf, &witness)?;
        Ok(row)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        merkle::{matrix::MatrixCommitment, MerkleTree, RustCryptoCompress, RustCryptoHasher},
        transcript::{
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            Reader,
        },
        utils::n_rand,
    };
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;

    #[test]
    fn test_mat_com() {
        type F = Goldilocks;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
        type Hasher = RustCryptoHasher<sha3::Keccak256>;
        type Compress = RustCryptoCompress<sha3::Keccak256>;
        let hasher = Hasher::default();
        let compress = Compress::default();

        let mat_comm = MerkleTree::<F, F, [u8; 32], _, _>::new(hasher, compress);
        for width in 1..5 {
            let mut transcript = Writer::init("test");
            let k = 5;
            let mut rng = crate::test::rng(1);
            let coeffs = n_rand(&mut rng, (1 << k) * width);
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

    #[test]
    fn test_mat_com_ext() {
        type F = Goldilocks;
        type Ext = BinomialExtensionField<F, 2>;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;
        type Hasher = RustCryptoHasher<sha3::Keccak256>;
        type Compress = RustCryptoCompress<sha3::Keccak256>;
        let hasher = Hasher::default();
        let compress = Compress::default();

        let mat_comm = MerkleTree::<F, Ext, [u8; 32], _, _>::new(hasher, compress);
        for width in 1..5 {
            let mut transcript = Writer::init("test");
            let k = 5;
            let mut rng = crate::test::rng(1);
            let coeffs = n_rand(&mut rng, (1 << k) * width);
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
}
