use crate::{MerkleData, MerkleTree, MerkleTreeExt};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, dense::RowMajorMatrixView};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fmt::Debug;
use transcript::{Error, Reader, Writer};

pub(crate) fn verify_merkle_proof<C: PseudoCompressionFunction<Digest, 2> + Sync, Digest>(
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

#[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
pub(crate) fn hash_base_reference_impl<
    F: Field,
    H: CryptographicHasher<F, [F; OUT]> + Sync + Sync,
    const OUT: usize,
>(
    data: &RowMajorMatrix<F>,
    hasher: &H,
) -> Vec<[F; OUT]> {
    data.par_row_slices()
        .map(|els| hasher.hash_iter(els.to_vec()))
        .collect::<Vec<_>>()
}

#[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
pub(crate) fn hash_ext_reference_impl<
    F: Field,
    Ext: ExtensionField<F>,
    H: CryptographicHasher<F, [F; OUT]> + Sync + Sync,
    const OUT: usize,
>(
    data: &RowMajorMatrix<Ext>,
    hasher: &H,
) -> Vec<[F; OUT]> {
    data.par_row_slices()
        .map(|ext| hasher.hash_iter(Ext::flatten_to_base(ext.to_vec())))
        .collect::<Vec<_>>()
}

pub(crate) fn compress_reference_impl<
    F: Field,
    C: PseudoCompressionFunction<[F; OUT], 2> + Sync,
    const OUT: usize,
>(
    layer0: &[[F; OUT]],
    compress: &C,
) -> Vec<[F; OUT]> {
    let (lo, hi) = layer0.split_at(layer0.len() / 2);
    lo.par_iter()
        .zip(hi.par_iter())
        .map(|(l, h)| compress.compress([*l, *h]))
        .collect::<Vec<_>>()
}

pub(crate) fn commit_reference_impl<
    Transcript,
    F: Field,
    H: CryptographicHasher<F, [F; OUT]> + Sync + Sync,
    C: PseudoCompressionFunction<[F; OUT], 2> + Sync,
    const OUT: usize,
>(
    transcript: &mut Transcript,
    hasher: &H,
    compress: &C,
    data: RowMajorMatrix<F>,
) -> Result<PoseidonCommitmentData<F, F, OUT>, Error>
where
    Transcript: Writer<[F; OUT]>,
{
    let mut layers = vec![hash_base_reference_impl(&data, hasher)];

    for _ in 0..log2_strict_usize(data.height()) {
        let layer = compress_reference_impl(layers.last().unwrap(), compress);
        layers.push(layer);
    }

    {
        let top = layers.last().unwrap();
        assert_eq!(top.len(), 1);
        transcript.write(top[0])?;
    }

    Ok(PoseidonCommitmentData { data, layers })
}

pub(crate) fn commit_ext_reference_impl<
    Transcript,
    F: Field,
    Ext: ExtensionField<F>,
    H: CryptographicHasher<F, [F; OUT]> + Sync + Sync,
    C: PseudoCompressionFunction<[F; OUT], 2> + Sync,
    const OUT: usize,
>(
    transcript: &mut Transcript,
    hasher: &H,
    compress: &C,
    data: RowMajorMatrix<Ext>,
) -> Result<PoseidonCommitmentData<F, Ext, OUT>, Error>
where
    Transcript: Writer<[F; OUT]>,
{
    let mut layers = vec![hash_ext_reference_impl(&data, hasher)];

    for _ in 0..log2_strict_usize(data.height()) {
        let layer = compress_reference_impl(layers.last().unwrap(), compress);
        layers.push(layer);
    }

    {
        let top = layers.last().unwrap();
        assert_eq!(top.len(), 1);
        transcript.write(top[0])?;
    }

    Ok(PoseidonCommitmentData { data, layers })
}

#[derive(Debug)]
pub struct PoseidonCommitmentData<F: Field, Ext: ExtensionField<F>, const OUT: usize> {
    pub(crate) data: RowMajorMatrix<Ext>,
    pub(crate) layers: Vec<Vec<[F; OUT]>>,
}

impl<F: Field, Ext: ExtensionField<F>, const OUT: usize> MerkleData<Ext>
    for PoseidonCommitmentData<F, Ext, OUT>
{
    type Digest = [F; OUT];
    fn commitment(&self) -> [F; OUT] {
        *self.layers.last().unwrap().last().unwrap()
    }

    fn get(&self, index: usize) -> Vec<Ext> {
        self.data.row_slice(index).unwrap().to_vec()
    }

    fn k(&self) -> usize {
        let k = log2_strict_usize(self.data.height());
        assert_eq!(self.layers.len(), k + 1);
        k
    }

    fn data(&self) -> RowMajorMatrixView<'_, Ext> {
        self.data.as_view()
    }
}

pub struct PoseidonMerkleTree<F: Field, H, C, const OUT: usize> {
    pub(crate) hasher: H,
    pub(crate) compress: C,
    pub(crate) _phantom: std::marker::PhantomData<F>,
}

impl<F: Field, H, C, const OUT: usize> PoseidonMerkleTree<F, H, C, OUT> {
    pub fn new(hasher: H, compress: C) -> Self {
        Self {
            hasher,
            compress,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<
    F: Field,
    H: CryptographicHasher<F, [F; OUT]> + Sync,
    C: PseudoCompressionFunction<[F; OUT], 2> + Sync,
    const OUT: usize,
> PoseidonMerkleTree<F, H, C, OUT>
{
    fn query<Transcript, Ext: ExtensionField<F>>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &PoseidonCommitmentData<F, Ext, OUT>,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Writer<F> + Writer<[F; OUT]>,
    {
        let k = data.k();
        let leaf = data.data.row_slice(index).unwrap().to_vec();
        transcript.write_hint_many(&Ext::flatten_to_base(leaf.clone()))?;

        #[cfg(debug_assertions)]
        let mut witness = vec![];

        let mut index_asc = index;
        data.layers
            .iter()
            .take(k) // skip the root
            .enumerate()
            .try_for_each(|(i, layer)| {
                let mid = 1 << (k - i - 1);
                let sibling_index = index_asc ^ mid;

                let node = layer[sibling_index];
                #[cfg(debug_assertions)]
                {
                    if i == 0 {
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
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Reader<F> + Reader<[F; OUT]>,
    {
        let row = (0..width)
            .map(|_| transcript.read_hint_many(Ext::DIMENSION))
            .collect::<Result<Vec<_>, _>>()?;
        let leaf = self.hasher.hash_iter(row.iter().flatten().cloned());
        let witness: Vec<[F; OUT]> = transcript.read_hint_many(k)?;
        verify_merkle_proof(&self.compress, comm, index, leaf, &witness)?;
        Ok(row
            .iter()
            .map(|e| Ext::from_basis_coefficients_slice(e).unwrap())
            .collect())
    }
}

impl<
    F: Field,
    H: CryptographicHasher<F, [F; OUT]> + Sync,
    C: PseudoCompressionFunction<[F; OUT], 2> + Sync,
    const OUT: usize,
> MerkleTree<F> for PoseidonMerkleTree<F, H, C, OUT>
{
    type MerkleData = PoseidonCommitmentData<F, F, OUT>;

    #[tracing::instrument(skip_all, fields(h = data.height(), w = data.width()))]
    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<Self::MerkleData, Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData<F>>::Digest>,
    {
        commit_reference_impl(transcript, &self.hasher, &self.compress, data)
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &Self::MerkleData,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Writer<F> + Writer<<Self::MerkleData as MerkleData<F>>::Digest>,
    {
        self.query::<_, F>(transcript, index, data)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: <Self::MerkleData as MerkleData<F>>::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<F>, Error>
    where
        Transcript: Reader<F> + Reader<<Self::MerkleData as MerkleData<F>>::Digest>,
    {
        self.verify::<_, F>(transcript, comm, index, width, k)
    }
}

impl<
    F: Field,
    Ext: ExtensionField<F>,
    H: CryptographicHasher<F, [F; OUT]> + Sync,
    C: PseudoCompressionFunction<[F; OUT], 2> + Sync,
    const OUT: usize,
> MerkleTreeExt<F, Ext> for PoseidonMerkleTree<F, H, C, OUT>
{
    type MerkleData = PoseidonCommitmentData<F, Ext, OUT>;

    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<PoseidonCommitmentData<F, Ext, OUT>, Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData<Ext>>::Digest>,
    {
        let mut layers = vec![hash_ext_reference_impl(&data, &self.hasher)];

        for _ in 0..log2_strict_usize(data.height()) {
            let layer = compress_reference_impl(layers.last().unwrap(), &self.compress);
            layers.push(layer);
        }

        {
            let top = layers.last().unwrap();
            assert_eq!(top.len(), 1);
            transcript.write(top[0])?;
        }

        Ok(PoseidonCommitmentData { data, layers })
    }

    fn get(&self, index: usize, data: &Self::MerkleData) -> Vec<Ext> {
        data.data.row_slice(index).unwrap().to_vec()
    }

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &Self::MerkleData,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Writer<F> + Writer<<Self::MerkleData as MerkleData<Ext>>::Digest>,
    {
        self.query::<_, Ext>(transcript, index, data)
    }

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: <Self::MerkleData as MerkleData<Ext>>::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, Error>
    where
        Transcript: Reader<F> + Reader<<Self::MerkleData as MerkleData<Ext>>::Digest>,
    {
        self.verify::<_, Ext>(transcript, comm, index, width, k)
    }
}
