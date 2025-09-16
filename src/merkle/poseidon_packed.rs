use crate::merkle::poseidon::{
    commit_ext_reference_impl, commit_reference_impl, compress_reference_impl,
};
use crate::merkle::{MerkleData, MerkleTree, MerkleTreeExt};
use crate::transcript::{Reader, Writer};
use crate::utils::log2_strict;
use crate::Error;
use p3_field::PackedValue;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::cmp;
use std::fmt::Debug;

const TOP_LEVEL_TRESHOLD: usize = 3;

fn unpack_array<P: PackedValue, const OUT: usize>(
    packed_digest: [P; OUT],
) -> impl Iterator<Item = [P::Value; OUT]> {
    (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
}

pub(super) fn verify_merkle_proof<C: PseudoCompressionFunction<Digest, 2> + Sync, Digest>(
    c: &C,
    comm: Digest,
    mut index: usize,
    leaf: Digest,
    witness: &[Digest],
) -> Result<(), Error>
where
    Digest: Copy + Clone + Sync + Debug + Eq + PartialEq,
{
    let k = witness.len();
    assert!(index < 1 << witness.len());
    let found = witness.iter().enumerate().fold(leaf, |acc, (i, &w)| {
        let mid = 1 << (k - i - 1);
        let acc = c.compress(if index >= mid { [w, acc] } else { [acc, w] });
        index &= mid - 1;
        acc
    });
    (comm == found).then_some(()).ok_or(Error::Verify)
}

#[derive(Debug)]
pub struct PoseidonCommitmentDataPacked<F: Field, Ext: Field, const OUT: usize> {
    pub(crate) data: RowMajorMatrix<Ext>,
    pub(crate) layers_bottom: Vec<Vec<[F::Packing; OUT]>>,
    pub(crate) layers_top: Vec<Vec<[F; OUT]>>,
}

impl<F: Field, Ext: ExtensionField<F>, const OUT: usize> MerkleData
    for PoseidonCommitmentDataPacked<F, Ext, OUT>
{
    type Digest = [F; OUT];
    fn commitment(&self) -> [F; OUT] {
        *self.layers_top.last().unwrap().last().unwrap()
    }

    fn k(&self) -> usize {
        let k = log2_strict(self.data.height());
        assert_eq!(self.layers_top.len() + self.layers_bottom.len(), k + 1);
        k
    }
}

pub struct PackedPoseidonMerkleTree<F: Field, H, C, const OUT: usize> {
    pub(crate) hasher: H,
    pub(crate) compress: C,
    pub(crate) _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, H, C, const OUT: usize> PackedPoseidonMerkleTree<F, H, C, OUT> {
    pub fn new(hasher: H, compress: C) -> Self {
        Self {
            hasher,
            compress,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Field, H, C, const OUT: usize> PackedPoseidonMerkleTree<F, H, C, OUT>
where
    H: CryptographicHasher<F, [F; OUT]> + CryptographicHasher<F::Packing, [F::Packing; OUT]> + Sync,
    C: PseudoCompressionFunction<[F; OUT], 2>
        + PseudoCompressionFunction<[F::Packing; OUT], 2>
        + Sync,
{
    #[tracing::instrument(skip_all)]
    #[allow(clippy::type_complexity)]
    fn ascend_tree(
        &self,
        transcript: &mut impl Writer<[F; OUT]>,
        layer0: Vec<[F::Packing; OUT]>,
    ) -> Result<(Vec<Vec<[F::Packing; OUT]>>, Vec<Vec<[F; OUT]>>), Error> {
        let mut layers_bottom = vec![layer0];
        let mut layers_top: Vec<Vec<[F; OUT]>> = vec![];

        loop {
            let layer0 = layers_bottom.last().unwrap();
            let h = layer0.len() * F::Packing::WIDTH;
            let k = log2_strict(h);
            if k == TOP_LEVEL_TRESHOLD {
                break;
            }

            let (lo, hi) = layer0.split_at(layer0.len() / 2);
            let layer1 = lo
                .par_iter()
                .zip(hi.par_iter())
                .map(|(l, h)| self.compress.compress([*l, *h]))
                .collect::<Vec<_>>();

            layers_bottom.push(layer1);
        }

        let layer = layers_bottom.last().unwrap();
        let layer = layer
            .par_iter()
            .flat_map(|&e| unpack_array::<F::Packing, OUT>(e).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for _ in 0..TOP_LEVEL_TRESHOLD {
            let layer =
                compress_reference_impl(layers_top.last().unwrap_or(&layer), &self.compress);
            layers_top.push(layer);
        }

        {
            let top = layers_top.last().unwrap();
            assert_eq!(top.len(), 1);
            transcript.write(top[0])?;
        }

        Ok((layers_bottom, layers_top))
    }

    fn query<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &PoseidonCommitmentDataPacked<F, Ext, OUT>,
    ) -> Result<Vec<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<[F; OUT]>,
    {
        let k = data.k();
        let leaf = data
            .data
            .row(index)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        transcript.write_hint_many(&Ext::flatten_to_base(leaf.clone()))?;

        #[cfg(debug_assertions)]
        let mut witness = vec![];

        let mut index_asc = index;
        data.layers_bottom
            .iter()
            .enumerate()
            .try_for_each(|(i, layer)| {
                let mid = 1 << (k - i - 1);
                let sibling_index = index_asc ^ mid;

                let packed = layer[sibling_index / F::Packing::WIDTH];
                let unpacked = unpack_array(packed).collect::<Vec<_>>();
                let node = unpacked[sibling_index % F::Packing::WIDTH];

                #[cfg(debug_assertions)]
                if i == 0 {
                    let sibling = data
                        .data
                        .row(sibling_index)
                        .unwrap()
                        .into_iter()
                        .collect::<Vec<_>>();
                    assert_eq!(self.hasher.hash_iter(Ext::flatten_to_base(sibling)), node);
                    witness.push(node);
                }

                index_asc &= mid - 1;
                transcript.write_hint(node)
            })?;

        let k = cmp::min(TOP_LEVEL_TRESHOLD - 1, k);
        data.layers_top
            .iter()
            .take(k) // skip the root
            .enumerate()
            .try_for_each(|(i, layer)| {
                let mid = 1 << (k - i - 1);
                let sibling_index = index_asc ^ mid;
                let node = layer[sibling_index];

                #[cfg(debug_assertions)]
                {
                    if i == 0 && data.layers_bottom.is_empty() {
                        let sibling = data
                            .data
                            .row(sibling_index)
                            .unwrap()
                            .into_iter()
                            .collect::<Vec<_>>();
                        assert_eq!(self.hasher.hash_iter(Ext::flatten_to_base(sibling)), node);
                    }
                    witness.push(node);
                }

                index_asc &= mid - 1;
                transcript.write_hint(node)
            })?;

        #[cfg(debug_assertions)]
        {
            let leaf = self.hasher.hash_iter(Ext::flatten_to_base(leaf.clone()));
            verify_merkle_proof(&self.compress, data.commitment(), index, leaf, &witness).unwrap();
        }

        Ok(leaf)
    }

    fn verify<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        comm: [F; OUT],
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, crate::Error>
    where
        Transcript: Reader<F> + Reader<[F; OUT]>,
    {
        let row: Vec<Vec<F>> = (0..width)
            .map(|_| transcript.read_hint_many(Ext::DIMENSION))
            .collect::<Result<Vec<_>, _>>()?;
        let leaf = self.hasher.hash_iter(row.iter().flatten().cloned());
        let witness: Vec<[F; OUT]> = transcript.read_hint_many(k)?;
        verify_merkle_proof::<C, [F; OUT]>(&self.compress, comm, index, leaf, &witness)?;
        Ok(row
            .iter()
            .map(|e| Ext::from_basis_coefficients_slice(e).unwrap())
            .collect())
    }
}

impl<
        F: Field,
        H: CryptographicHasher<F, [F; OUT]>
            + CryptographicHasher<F::Packing, [F::Packing; OUT]>
            + Sync,
        C: PseudoCompressionFunction<[F; OUT], 2>
            + PseudoCompressionFunction<[F::Packing; OUT], 2>
            + Sync,
        const OUT: usize,
    > MerkleTree<F> for PackedPoseidonMerkleTree<F, H, C, OUT>
{
    type MerkleData = PoseidonCommitmentDataPacked<F, F, OUT>;

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<Self::MerkleData, crate::Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData>::Digest>,
    {
        if data.height() < 1 << TOP_LEVEL_TRESHOLD {
            let comm = commit_reference_impl(transcript, &self.hasher, &self.compress, data)?;
            Ok(PoseidonCommitmentDataPacked {
                data: comm.data,
                layers_top: comm.layers,
                layers_bottom: vec![],
            })
        } else {
            let layer0 = tracing::info_span!("leaf_hash").in_scope(|| {
                data.par_row_chunks_exact(F::Packing::WIDTH)
                    .map(|chunk| {
                        let chunk = chunk.transpose();
                        assert_eq!(chunk.width(), F::Packing::WIDTH);
                        let slices = F::Packing::pack_slice(&chunk.values);
                        self.hasher.hash_slice(slices)
                    })
                    .collect::<Vec<_>>()
            });

            let (layers_bottom, layers_top) = self.ascend_tree(transcript, layer0)?;

            Ok(PoseidonCommitmentDataPacked {
                data,
                layers_bottom,
                layers_top,
            })
        }
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
        self.query::<_, F>(transcript, index, data)
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
        self.verify(transcript, comm, index, width, k)
    }
}

impl<
        F: Field,
        Ext: ExtensionField<F>,
        H: CryptographicHasher<F, [F; OUT]>
            + CryptographicHasher<F::Packing, [F::Packing; OUT]>
            + Sync,
        C: PseudoCompressionFunction<[F; OUT], 2>
            + PseudoCompressionFunction<[F::Packing; OUT], 2>
            + Sync,
        const OUT: usize,
    > MerkleTreeExt<F, Ext> for PackedPoseidonMerkleTree<F, H, C, OUT>
{
    type MerkleData = PoseidonCommitmentDataPacked<F, Ext, OUT>;

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<PoseidonCommitmentDataPacked<F, Ext, OUT>, crate::Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData>::Digest>,
    {
        if data.height() < 1 << TOP_LEVEL_TRESHOLD {
            let comm = commit_ext_reference_impl(transcript, &self.hasher, &self.compress, data)?;
            Ok(PoseidonCommitmentDataPacked {
                data: comm.data,
                layers_top: comm.layers,
                layers_bottom: vec![],
            })
        } else {
            let layer0 = tracing::info_span!("leaf_hash").in_scope(|| {
                data.par_row_chunks_exact(F::Packing::WIDTH)
                    .map(|chunk| {
                        let flat = Ext::flatten_to_base(chunk.values.to_vec()); // TODO: consider going unsafe transmute
                        let width = flat.len() / F::Packing::WIDTH;
                        let chunk = RowMajorMatrix::new(flat, width);
                        let chunk = chunk.transpose();
                        assert_eq!(chunk.width(), F::Packing::WIDTH);
                        let slices = F::Packing::pack_slice(&chunk.values);
                        self.hasher.hash_slice(slices)
                    })
                    .collect::<Vec<_>>()
            });

            let (layers_bottom, layers_top) = self.ascend_tree(transcript, layer0)?;

            Ok(PoseidonCommitmentDataPacked {
                data,
                layers_bottom,
                layers_top,
            })
        }
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
        self.query::<_, Ext>(transcript, index, data)
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
        self.verify(transcript, comm, index, width, k)
    }
}
