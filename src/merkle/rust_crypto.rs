use crate::merkle::{Compress, Hasher};
use digest::FixedOutputReset;
use p3_field::{ExtensionField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    extension::FlatMatrixView,
};
use rayon::iter::ParallelIterator;
use sha2::Digest;
use std::fmt::Debug;

#[derive(Debug, Clone, Default)]
pub struct RustCryptoCompress<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest + FixedOutputReset + Sync, const N: usize> Compress<[u8; 32], N>
    for RustCryptoCompress<D>
{
    fn apply(&self, input: [[u8; 32]; N]) -> [u8; 32] {
        let mut h = D::new();
        input.iter().for_each(|e| Digest::update(&mut h, e));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RustCryptoHasher<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest + FixedOutputReset + Sync, F: Field> Hasher<F, [u8; 32]> for RustCryptoHasher<D> {
    fn hash_base_data(&self, data: &RowMajorMatrix<F>) -> Vec<[u8; 32]> {
        data.par_row_slices()
            .map(|rows| {
                let mut h = D::new();
                rows.iter()
                    .for_each(|field| Digest::update(&mut h, bincode::serialize(&field).unwrap()));
                Digest::finalize(h).to_vec().try_into().unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn hash_ext_data<Ext: ExtensionField<F>>(
        &self,
        data: &FlatMatrixView<F, Ext, DenseMatrix<Ext>>,
    ) -> Vec<[u8; 32]> {
        data.par_row_slices()
            .map(|rows| {
                let mut h = D::new();
                rows.iter()
                    .flat_map(Ext::as_basis_coefficients_slice)
                    .for_each(|field| Digest::update(&mut h, bincode::serialize(&field).unwrap()));
                Digest::finalize(h).to_vec().try_into().unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn hash_ext_iter<I, Ext: ExtensionField<F>>(&self, row: I) -> [u8; 32]
    where
        I: IntoIterator<Item = Ext>,
    {
        let mut h = D::new();
        row.into_iter().for_each(|ext| {
            ext.as_basis_coefficients_slice()
                .iter()
                .for_each(|field| Digest::update(&mut h, bincode::serialize(&field).unwrap()))
        });
        Digest::finalize(h).to_vec().try_into().unwrap()
    }

    fn hash_base_iter<I>(&self, row: I) -> [u8; 32]
    where
        I: IntoIterator<Item = F>,
    {
        let mut h = D::new();
        row.into_iter()
            .for_each(|field| Digest::update(&mut h, bincode::serialize(&field).unwrap()));
        Digest::finalize(h).to_vec().try_into().unwrap()
    }
}
