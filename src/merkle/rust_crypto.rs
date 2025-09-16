use digest::Digest;
use p3_field::{ExtensionField, Field};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use std::fmt::Debug;

use crate::{
    merkle::{MerkleData, MerkleTree, MerkleTreeExt},
    transcript::{Reader, Writer},
    utils::log2_strict,
};

pub fn verify_merkle_proof<D: Digest>(
    comm: [u8; 32],
    mut index: usize,
    leaf: [u8; 32],
    witness: &[[u8; 32]],
) -> Result<(), crate::Error> {
    assert!(index < 1 << witness.len());
    let found = witness.iter().fold(leaf, |acc, &w| {
        let mut h = D::new();

        let pair = if index & 1 == 1 { [w, acc] } else { [acc, w] };
        Digest::update(&mut h, pair[0]);
        Digest::update(&mut h, pair[1]);
        let acc = Digest::finalize(h).to_vec().try_into().unwrap();
        index >>= 1;
        acc
    });
    (comm == found).then_some(()).ok_or(crate::Error::Verify)
}

#[derive(Debug)]
pub struct CommitmentData<F: Field> {
    pub(crate) data: RowMajorMatrix<F>,
    pub(crate) layers: Vec<Vec<[u8; 32]>>,
}

impl<F: Field> MerkleData for CommitmentData<F> {
    type Digest = [u8; 32];
    fn commitment(&self) -> Self::Digest {
        *self.layers.last().unwrap().last().unwrap()
    }

    fn k(&self) -> usize {
        let k = log2_strict(self.data.height());
        assert_eq!(self.layers.len(), k + 1);
        k
    }
}

#[derive(Debug, Clone, Default)]
pub struct RustCryptoMerkleTree<D: Digest> {
    pub(crate) _phantom: std::marker::PhantomData<D>,
}

impl<D: Digest> RustCryptoMerkleTree<D> {
    fn commit<Transcript, F: Field, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<CommitmentData<Ext>, crate::Error>
    where
        Transcript: Writer<[u8; 32]>,
    {
        let layer0: Vec<[u8; 32]> = data
            .par_row_slices()
            .map(|rows| {
                let mut h = D::new();
                rows.iter()
                    .flat_map(Ext::as_basis_coefficients_slice)
                    .for_each(|field| Digest::update(&mut h, bincode::serialize(&field).unwrap()));
                Digest::finalize(h).to_vec().try_into().unwrap()
            })
            .collect::<Vec<_>>();

        let mut layers = vec![layer0];
        for _ in 0..log2_strict(data.height()) {
            let next_layer = layers
                .last()
                .unwrap()
                .par_chunks_exact(2)
                .map(|chunk| {
                    let mut h = D::new();
                    Digest::update(&mut h, chunk[0]);
                    Digest::update(&mut h, chunk[1]);
                    Digest::finalize(h).to_vec().try_into().unwrap()
                })
                .collect::<Vec<_>>();
            layers.push(next_layer);
        }

        {
            let top = layers.last().unwrap();
            assert_eq!(top.len(), 1);
            transcript.write(top[0])?;
        }

        Ok(CommitmentData { data, layers })
    }

    fn query<Transcript, F: Field, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &CommitmentData<Ext>,
    ) -> Result<Vec<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<[u8; 32]>,
    {
        let k = data.k();
        let row = data
            .data
            .row(index)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        transcript.write_hint_many(&Ext::flatten_to_base(row.clone()))?;

        let mut witness = vec![];
        let mut index_asc = index;

        data.layers
            .iter()
            .take(k)
            .enumerate()
            .try_for_each(|(_i, layer)| {
                let node = layer[index_asc ^ 1];
                #[cfg(debug_assertions)]
                if _i == 0 {
                    let sibling = data.data.row(index ^ 1).unwrap();
                    let mut h = D::new();
                    sibling.into_iter().for_each(|field| {
                        Digest::update(&mut h, bincode::serialize(&field).unwrap())
                    });
                    let _node: [u8; 32] = Digest::finalize(h).to_vec().try_into().unwrap();
                    assert_eq!(_node, node);
                }
                witness.push(node);
                index_asc >>= 1;
                transcript.write_hint(node)
            })?;

        #[cfg(debug_assertions)]
        {
            let mut h = D::new();
            row.iter().for_each(|ext| {
                ext.as_basis_coefficients_slice()
                    .iter()
                    .for_each(|e| Digest::update(&mut h, bincode::serialize(&e).unwrap()));
            });
            let leaf = Digest::finalize(h).to_vec().try_into().unwrap();
            verify_merkle_proof::<D>(data.commitment(), index, leaf, &witness).unwrap();
        }

        Ok(row)
    }

    fn verify<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        comm: [u8; 32],
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, crate::Error>
    where
        Transcript: Reader<F> + Reader<[u8; 32]>,
    {
        let row: Vec<Vec<F>> = (0..width)
            .map(|_| transcript.read_hint_many(Ext::DIMENSION))
            .collect::<Result<Vec<_>, _>>()?;
        let mut h = D::new();
        row.iter().for_each(|ext| {
            ext.iter()
                .for_each(|field| Digest::update(&mut h, bincode::serialize(field).unwrap()));
        });
        let leaf = Digest::finalize(h).to_vec().try_into().unwrap();
        let witness: Vec<[u8; 32]> = transcript.read_hint_many(k)?;
        verify_merkle_proof::<D>(comm, index, leaf, &witness)?;
        Ok(row
            .iter()
            .map(|e| Ext::from_basis_coefficients_slice(e).unwrap())
            .collect())
    }
}

impl<F: Field, D: Digest> MerkleTree<F> for RustCryptoMerkleTree<D> {
    type MerkleData = CommitmentData<F>;

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<Self::MerkleData, crate::Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData>::Digest>,
    {
        self.commit::<_, F, F>(transcript, data)
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &Self::MerkleData,
    ) -> Result<Vec<F>, crate::Error>
    where
        Transcript: Writer<F> + Writer<<Self::MerkleData as MerkleData>::Digest>,
    {
        self.query::<_, F, F>(transcript, index, data)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: <Self::MerkleData as MerkleData>::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<F>, crate::Error>
    where
        Transcript: Reader<F> + Reader<<Self::MerkleData as MerkleData>::Digest>,
    {
        RustCryptoMerkleTree::<D>::verify::<_, F, F>(transcript, comm, index, width, k)
    }
}

impl<F: Field, Ext: ExtensionField<F>, D: Digest> MerkleTreeExt<F, Ext>
    for RustCryptoMerkleTree<D>
{
    type MerkleData = CommitmentData<Ext>;

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<Self::MerkleData, crate::Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData>::Digest>,
    {
        self.commit::<_, F, Ext>(transcript, data)
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &Self::MerkleData,
    ) -> Result<Vec<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<<Self::MerkleData as MerkleData>::Digest>,
    {
        self.query::<_, F, Ext>(transcript, index, data)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: <Self::MerkleData as MerkleData>::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, crate::Error>
    where
        Transcript: Reader<F> + Reader<<Self::MerkleData as MerkleData>::Digest>,
    {
        RustCryptoMerkleTree::<D>::verify::<_, F, Ext>(transcript, comm, index, width, k)
    }
}
