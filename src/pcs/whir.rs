use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    merkle::matrix::{MatrixCommitment, MatrixCommitmentData},
    pcs::params::SecurityAssumption,
    poly::Poly,
    transcript::{Challenge, Writer},
};

pub fn commit<
    Transcript,
    F: TwoAdicField,
    Ext: ExtensionField<F> + TwoAdicField,
    DFT: TwoAdicSubgroupDft<Ext>,
    MCom: MatrixCommitment<Ext>,
>(
    transcript: &mut Transcript,
    poly: &Poly<Ext>,
    rate: usize,
    folding: usize,
    mat_comm: &MCom,
) -> Result<MatrixCommitmentData<Ext, MCom::Digest>, crate::Error>
where
    Transcript: Writer<MCom::Digest>,
{
    // pad
    let size = 1 << (poly.k() + rate);
    let mut padded: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec(size);
    padded[..poly.len()].copy_from_slice(&poly);

    // encode
    let mat = RowMajorMatrix::new(padded, 1 << folding);
    let cw = DFT::default().dft_algebra_batch(mat);

    // commit
    mat_comm.commit(transcript, cw)
}

pub struct Whir<
    F: TwoAdicField,
    Ext: ExtensionField<F>,
    MCom: MatrixCommitment<F>,
    MComExt: MatrixCommitment<Ext>,
> {
    pub k: usize,
    pub folding: usize,
    pub rate: usize,
    soundness: SecurityAssumption,
    security_level: usize,
    comm: MCom,
    comm_ext: MComExt,
    _marker: std::marker::PhantomData<(F, Ext)>,
}

impl<
        F: TwoAdicField,
        Ext: ExtensionField<F>,
        MCom: MatrixCommitment<F>,
        MComExt: MatrixCommitment<Ext>,
    > Whir<F, Ext, MCom, MComExt>
{
    pub fn new(
        k: usize,
        folding: usize,
        rate: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        comm: MCom,
        comm_ext: MComExt,
    ) -> Self {
        Self {
            k,
            folding,
            rate,
            soundness,
            security_level,
            comm,
            comm_ext,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn commit<Transcript, DFT: TwoAdicSubgroupDft<F>>(
        &self,
        transcript: &mut Transcript,
        poly: &Poly<F>,
    ) -> Result<MatrixCommitmentData<F, MCom::Digest>, crate::Error>
    where
        Transcript: Writer<Ext> + Writer<MCom::Digest> + Challenge<F, Ext>,
    {
        commit::<_, _, _, DFT, _>(transcript, poly, self.rate, self.folding, &self.comm)
    }

    pub fn n_ood_queries(&self, k: usize, rate: usize) -> usize {
        self.soundness
            .n_ood_queries(self.security_level, k, rate, Ext::bits())
    }

    pub fn n_stir_queries(&self, rate: usize) -> usize {
        self.soundness.n_stir_queries(self.security_level, rate)
    }
}
