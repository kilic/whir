use common::Error;
use common::utils::TwoAdicSlice;
use merkle::{MerkleData, MerkleTree, MerkleTreeExt};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixView},
};
use p3_util::log2_ceil_usize;
use transcript::Writer;

#[tracing::instrument(skip_all)]
pub fn commit_base<Transcript, F: TwoAdicField, Dft: TwoAdicSubgroupDft<F>, MT: MerkleTree<F>>(
    dft: &Dft,
    transcript: &mut Transcript,
    poly: &[F],
    rate: usize,
    folding: usize,
    mt: &MT,
) -> Result<MT::MerkleData, Error>
where
    Transcript: Writer<<MT::MerkleData as MerkleData<F>>::Digest>,
{
    let mut mat = RowMajorMatrix::new(poly.to_vec(), 1 << folding);
    mat.pad_to_height(1 << (poly.k() + rate - folding), F::ZERO);
    let codeword = tracing::info_span!("dft-base", height = mat.height(), width = mat.width())
        .in_scope(|| dft.dft_batch(mat).to_row_major_matrix());
    mt.commit(transcript, codeword)
}

#[tracing::instrument(skip_all)]
pub fn commit_ext<
    Transcript,
    F: TwoAdicField,
    Ext: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
    MT: MerkleTreeExt<F, Ext>,
>(
    dft: &Dft,
    transcript: &mut Transcript,
    poly: &[Ext],
    rate: usize,
    folding: usize,
    mt: &MT,
) -> Result<MT::MerkleData, Error>
where
    Transcript: Writer<<MT::MerkleData as MerkleData<Ext>>::Digest>,
{
    let mut mat = RowMajorMatrix::new(poly.to_vec(), 1 << folding);
    mat.pad_to_height(1 << (poly.k() + rate - folding), Ext::ZERO);
    let codeword = tracing::info_span!("dft-ext", h = mat.height(), w = mat.width())
        .in_scope(|| dft.dft_algebra_batch(mat));
    mt.commit(transcript, codeword)
}

#[tracing::instrument(skip_all)]
pub fn commit_base_interleaved<
    Transcript,
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: MerkleTree<F>,
>(
    dft: &Dft,
    transcript: &mut Transcript,
    poly: &[F],
    rate: usize,
    folding: usize,
    mt: &MT,
) -> Result<MT::MerkleData, Error>
where
    Transcript: Writer<<MT::MerkleData as MerkleData<F>>::Digest>,
{
    let k = log2_ceil_usize(poly.len());
    let width = 1 << (k - folding);
    assert_eq!(poly.len() % width, 0);
    let mut mat = tracing::info_span!("transpose")
        .in_scope(|| RowMajorMatrixView::new(poly, width).transpose());
    mat.pad_to_height(1 << (k + rate - folding), F::ZERO);
    let codeword = tracing::info_span!("dft-base", height = mat.height(), width = mat.width())
        .in_scope(|| dft.dft_batch(mat).to_row_major_matrix());
    mt.commit(transcript, codeword)
}

#[tracing::instrument(skip_all)]
pub fn commit_ext_interleaved<
    Transcript,
    F: TwoAdicField,
    Ext: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
    MT: MerkleTreeExt<F, Ext>,
>(
    dft: &Dft,
    transcript: &mut Transcript,
    poly: &[Ext],
    rate: usize,
    folding: usize,
    mt: &MT,
) -> Result<MT::MerkleData, Error>
where
    Transcript: Writer<<MT::MerkleData as MerkleData<Ext>>::Digest>,
{
    let width = 1 << (poly.k() - folding);
    let mut mat = tracing::info_span!("transpose")
        .in_scope(|| RowMajorMatrixView::new(poly, width).transpose());
    mat.pad_to_height(1 << (poly.k() + rate - folding), Ext::ZERO);
    let codeword = tracing::info_span!("dft-ext", h = mat.height(), w = mat.width())
        .in_scope(|| dft.dft_algebra_batch(mat));
    mt.commit(transcript, codeword)
}
