use std::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use transcript::{Reader, Writer};

pub mod poseidon;
pub mod poseidon_packed;
// pub mod rust_crypto;

pub trait MerkleData<F: Field>: Debug {
    type Digest: Copy + Clone + Debug + Send + Sync + Eq + PartialEq;
    fn commitment(&self) -> Self::Digest;
    fn k(&self) -> usize;
    fn get(&self, index: usize) -> Vec<F>;
    fn data(&self) -> RowMajorMatrixView<'_, F>;
}

pub trait MerkleTree<F: Field> {
    type MerkleData: MerkleData<F>;

    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<F>,
    ) -> Result<Self::MerkleData, transcript::Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData<F>>::Digest>;

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &Self::MerkleData,
    ) -> Result<Vec<F>, transcript::Error>
    where
        Transcript: Writer<F> + Writer<<Self::MerkleData as MerkleData<F>>::Digest>;

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: <Self::MerkleData as MerkleData<F>>::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<F>, transcript::Error>
    where
        Transcript: Reader<F> + Reader<<Self::MerkleData as MerkleData<F>>::Digest>;
}

pub trait MerkleTreeExt<F: Field, Ext: ExtensionField<F>> {
    type MerkleData: MerkleData<Ext>;

    fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        data: RowMajorMatrix<Ext>,
    ) -> Result<Self::MerkleData, transcript::Error>
    where
        Transcript: Writer<<Self::MerkleData as MerkleData<Ext>>::Digest>;

    fn query<Transcript>(
        &self,
        transcript: &mut Transcript,
        index: usize,
        data: &Self::MerkleData,
    ) -> Result<Vec<Ext>, transcript::Error>
    where
        Transcript: Writer<F> + Writer<<Self::MerkleData as MerkleData<Ext>>::Digest>;

    fn get(&self, index: usize, data: &Self::MerkleData) -> Vec<Ext>;

    fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        comm: <Self::MerkleData as MerkleData<Ext>>::Digest,
        index: usize,
        width: usize,
        k: usize,
    ) -> Result<Vec<Ext>, transcript::Error>
    where
        Transcript: Reader<F> + Reader<<Self::MerkleData as MerkleData<Ext>>::Digest>;
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use crate::{
        MerkleData, MerkleTree, MerkleTreeExt, poseidon::PoseidonMerkleTree,
        poseidon_packed::PackedPoseidonMerkleTree,
    };
    use common::utils::rand_vec;
    use p3_field::{ExtensionField, Field, extension::BinomialExtensionField};
    use p3_koala_bear::Poseidon2KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::distr::{Distribution, StandardUniform};
    use transcript::{
        ChallengeBits, Reader, Writer,
        test_transcript::{TestReader, TestWriter},
    };

    fn prover_base<F: Field, Transcript, M: MerkleTree<F>>(
        mut rng: impl rand::Rng,
        transcript: &mut Transcript,
        m: M,
        width: Range<usize>,
        k: usize,
    ) where
        StandardUniform: Distribution<F>,
        Transcript: Writer<<M::MerkleData as MerkleData<F>>::Digest> + Writer<F> + Clone,
    {
        for width in width {
            let coeffs = rand_vec(&mut rng, (1 << k) * width);
            let data = RowMajorMatrix::new(coeffs, width);
            let comm0 = m.commit(transcript, data.clone()).unwrap();

            (0..1 << k).for_each(|index| {
                m.query(transcript, index, &comm0).unwrap();
            });
        }
    }

    fn verifier_base<F: Field, Transcript, M: MerkleTree<F>>(
        transcript: &mut Transcript,
        m: M,
        width: Range<usize>,
        k: usize,
    ) where
        Transcript: Reader<<M::MerkleData as MerkleData<F>>::Digest> + Reader<F>,
    {
        for width in width {
            let comm: <M::MerkleData as MerkleData<F>>::Digest = transcript.read().unwrap();

            (0..1 << k).for_each(|index| {
                m.verify(transcript, comm, index, width, k).unwrap();
            });
        }
    }

    fn prover_ext<F: Field, Ext: ExtensionField<F>, Transcript, M: MerkleTreeExt<F, Ext>>(
        mut rng: impl rand::Rng,
        transcript: &mut Transcript,
        m: M,
        width: Range<usize>,
        k: usize,
    ) where
        StandardUniform: Distribution<Ext>,
        Transcript: Writer<<M::MerkleData as MerkleData<Ext>>::Digest> + Writer<F> + Clone,
    {
        for width in width {
            let coeffs = rand_vec(&mut rng, (1 << k) * width);
            let data = RowMajorMatrix::new(coeffs, width);
            let comm = m.commit(transcript, data.clone()).unwrap();
            (0..1 << k).for_each(|index| {
                m.query(transcript, index, &comm).unwrap();
            });
        }
    }

    fn verifier_ext<F: Field, Ext: ExtensionField<F>, Transcript, M: MerkleTreeExt<F, Ext>>(
        transcript: &mut Transcript,
        m: M,
        width: Range<usize>,
        k: usize,
    ) where
        Transcript: Reader<<M::MerkleData as MerkleData<Ext>>::Digest> + Reader<F>,
    {
        for width in width {
            let comm: <M::MerkleData as MerkleData<Ext>>::Digest = transcript.read().unwrap();
            (0..1 << k).for_each(|index| {
                m.verify(transcript, comm, index, width, k).unwrap();
            });
        }
    }

    // #[test]
    // fn test_merkle_base_rustcrypto() {
    //     type F = p3_koala_bear::KoalaBear;

    //     let mut rng = common::test::rng(0);
    //     for k in 1..9 {
    //         {
    //             let mut transcript = TestWriter::<Vec<u8>, F>::init();
    //             let m = RustCryptoMerkleTree::<sha2::Sha256>::default();
    //             prover_base::<F, TestWriter<Vec<u8>, F>, _>(&mut rng, &mut transcript, m, 1..7, k);

    //             let proof = transcript.finalize();
    //             let mut transcript = TestReader::<&[u8], F>::init(&proof);
    //             let m = RustCryptoMerkleTree::<sha2::Sha256>::default();
    //             verifier_base::<F, _, _>(&mut transcript, m, 1..7, k);
    //         }
    //     }
    // }

    // #[test]
    // fn test_merkle_ext_rustcrypto() {
    //     type F = p3_koala_bear::KoalaBear;
    //     type Ext = BinomialExtensionField<F, 4>;

    //     let mut rng = common::test::rng(0);
    //     for k in 1..9 {
    //         {
    //             let mut transcript = TestWriter::<Vec<u8>, F>::init();
    //             let m = RustCryptoMerkleTree::<sha2::Sha256>::default();
    //             prover_ext::<F, Ext, TestWriter<Vec<u8>, F>, _>(
    //                 &mut rng,
    //                 &mut transcript,
    //                 m,
    //                 1..7,
    //                 k,
    //             );

    //             let proof = transcript.finalize();
    //             let mut transcript = TestReader::<&[u8], F>::init(&proof);
    //             let m = RustCryptoMerkleTree::<sha2::Sha256>::default();
    //             verifier_ext::<F, Ext, _, _>(&mut transcript, m, 1..7, k);
    //         }
    //     }
    // }

    #[test]
    fn test_merkle_base() {
        type F = p3_koala_bear::KoalaBear;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;

        let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());

        let mut rng = common::test::rng(0);
        for k in 1..9 {
            {
                let mut transcript = TestWriter::<Vec<u8>, F>::init();
                let m = PoseidonMerkleTree::new(hasher.clone(), compress.clone());
                prover_base::<F, _, _>(&mut rng, &mut transcript, m, 1..7, k);

                let proof = transcript.finalize();
                let mut transcript = TestReader::<&[u8], F>::init(&proof);
                let m = PoseidonMerkleTree::new(hasher.clone(), compress.clone());
                verifier_base::<F, _, _>(&mut transcript, m, 1..7, k);
            }
        }
    }

    #[test]
    fn test_merkle_ext() {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;

        let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());

        let mut rng = common::test::rng(0);
        for k in 1..9 {
            {
                let mut transcript = TestWriter::<Vec<u8>, F>::init();
                let m = PoseidonMerkleTree::new(hasher.clone(), compress.clone());
                prover_ext::<F, Ext, _, _>(&mut rng, &mut transcript, m, 1..7, k);

                let proof = transcript.finalize();
                let mut transcript = TestReader::<&[u8], F>::init(&proof);
                let m = PoseidonMerkleTree::new(hasher.clone(), compress.clone());
                verifier_ext::<F, Ext, _, _>(&mut transcript, m, 1..7, k);
            }
        }
    }

    #[test]
    fn test_merkle_base_packed() {
        type F = p3_koala_bear::KoalaBear;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;

        let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());

        let mut rng = common::test::rng(0);
        for k in 1..9 {
            {
                let mut transcript = TestWriter::<Vec<u8>, F>::init();
                let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
                prover_base::<F, _, _>(&mut rng, &mut transcript, m, 1..7, k);

                let proof = transcript.finalize();
                let mut transcript = TestReader::<&[u8], F>::init(&proof);
                let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
                verifier_base::<F, _, _>(&mut transcript, m, 1..7, k);
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_merkle_base_packed() {
        type F = p3_koala_bear::KoalaBear;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;

        let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());

        let mut rng = common::test::rng(0);
        let k = 21;
        let width = 32;

        common::test::init_tracing();
        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        let coeffs: Vec<F> = rand_vec(&mut rng, (1 << k) * width);
        let data = RowMajorMatrix::new(coeffs, width);
        let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
        let comm = MerkleTree::commit(&m, &mut transcript, data).unwrap();
        let idx: usize = transcript.draw(k);
        MerkleTree::query(&m, &mut transcript, idx, &comm).unwrap();

        let proof = transcript.finalize();
        let mut transcript = TestReader::<&[u8], F>::init(&proof);
        let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
        let comm = transcript.read().unwrap();
        let idx: usize = transcript.draw(k);
        MerkleTree::verify(&m, &mut transcript, comm, idx, width, k).unwrap();
    }

    #[test]
    fn test_merkle_ext_packed() {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;

        let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());

        {
            let mut rng = common::test::rng(0);
            for k in 1..9 {
                {
                    let mut transcript = TestWriter::<Vec<u8>, F>::init();
                    let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
                    prover_ext::<F, Ext, _, _>(&mut rng, &mut transcript, m, 1..7, k);

                    let proof = transcript.finalize();
                    let mut transcript = TestReader::<&[u8], F>::init(&proof);
                    let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
                    verifier_ext::<F, Ext, _, _>(&mut transcript, m, 1..7, k);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_merkle_ext_packed() {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;

        let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());

        let mut rng = common::test::rng(0);
        let k = 18;
        let width = 32;

        common::test::init_tracing();
        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        let coeffs: Vec<Ext> = rand_vec(&mut rng, (1 << k) * width);
        let data = RowMajorMatrix::new(coeffs, width);
        let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
        let comm = MerkleTreeExt::commit(&m, &mut transcript, data).unwrap();
        let idx: usize = transcript.draw(k);
        MerkleTreeExt::query(&m, &mut transcript, idx, &comm).unwrap();

        let proof = transcript.finalize();
        let mut transcript = TestReader::<&[u8], F>::init(&proof);
        let m = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
        let comm: [F; 8] = transcript.read().unwrap();
        let idx: usize = transcript.draw(k);
        MerkleTreeExt::<F, Ext>::verify(&m, &mut transcript, comm, idx, width, k).unwrap();
    }
}
