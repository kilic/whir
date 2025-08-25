use crate::poly::{Eval, Point, Poly};
use p3_field::{ExtensionField, Field};

pub mod params;
pub mod sumcheck;
pub mod whir;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Claim<F: Field, Ext: ExtensionField<F>> {
    pub(crate) point: Point<F>,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> Claim<F, Ext> {
    pub fn new(point: Point<F>, eval: Ext) -> Self {
        Self { point, eval }
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }

    pub fn eq(&self) -> Poly<F, Eval> {
        self.point.eq(F::ONE)
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn point(&self) -> Point<F> {
        self.point.clone()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        field::SerializedField,
        merkle::{
            MerkleTree, PoseidonCompress, PoseidonHasher, RustCryptoCompress, RustCryptoHasher,
        },
        pcs::{params::SecurityAssumption, whir::Whir, Claim},
        poly::{Coeff, Eval, Point, Poly},
        transcript::{
            poseidon::{PoseidonReader, PoseidonWriter},
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            Challenge, Reader, Writer,
        },
    };
    use p3_baby_bear::Poseidon2BabyBear;
    use p3_challenger::DuplexChallenger;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{
        distr::{Distribution, StandardUniform},
        Rng,
    };

    pub(crate) fn make_claims_ext<
        Transcript,
        F: Field,
        Ex0: ExtensionField<F>,
        Ex1: ExtensionField<Ex0> + ExtensionField<F>,
    >(
        transcript: &mut Transcript,
        n_points: usize,
        poly: &Poly<Ex0, Eval>,
    ) -> Result<Vec<Claim<Ex1, Ex1>>, crate::Error>
    where
        Transcript: Challenge<F, Ex1> + Writer<Ex1>,
    {
        (0..n_points)
            .map(|_| {
                let point = Point::expand(poly.k(), transcript.draw());
                let eval = poly.eval(&point);
                let claim = Claim { point, eval };
                transcript.write(claim.eval)?;
                Ok(claim)
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn make_claims_base<Transcript, F: Field, Ex0: ExtensionField<F>>(
        transcript: &mut Transcript,
        n_points: usize,
        poly: &Poly<Ex0, Eval>,
    ) -> Result<Vec<Claim<F, Ex0>>, crate::Error>
    where
        Transcript: Challenge<F, F> + Writer<Ex0>,
    {
        (0..n_points)
            .map(|_| {
                let point = Point::expand(poly.k(), transcript.draw());
                let eval = poly.eval(&point.as_ext::<Ex0>());
                let claim = Claim { point, eval };
                transcript.write(claim.eval)?;
                Ok(claim)
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn get_claims_base<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        k: usize,
        n: usize,
    ) -> Result<Vec<Claim<F, Ext>>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, F>,
    {
        (0..n)
            .map(|_| {
                let var: F = transcript.draw();
                let point = Point::expand(k, var);
                let eval = transcript.read()?;
                Ok(Claim::new(point, eval))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn get_claims_ext<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        k: usize,
        n: usize,
    ) -> Result<Vec<Claim<Ext, Ext>>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        (0..n)
            .map(|_| {
                let var: Ext = transcript.draw();
                let point = Point::expand(k, var);
                let eval = transcript.read()?;
                Ok(Claim::new(point, eval))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    #[test]
    fn test_final_folding() {
        type F = p3_goldilocks::Goldilocks;
        use p3_field::PrimeCharacteristicRing;
        use p3_field::TwoAdicField;

        let k = 7usize;
        let mut rng = &mut crate::test::rng(1);
        let poly_in_coeffs: Poly<F, Coeff> = Poly::rand(&mut rng, k);
        let poly_in_evals = poly_in_coeffs.clone().to_evals();
        let size = 1 << k;

        for folding in 0..k {
            let width = 1 << folding;

            // rearrange coefficients since variables are fixed in backwards order
            let mut transposed = vec![F::ZERO; size];
            transpose::transpose(&poly_in_coeffs, &mut transposed, size / width, width);

            for extend in 0..5 {
                let rate = k + extend;

                // pad as rate
                let pad_size = 1 << rate;
                let mut padded: Vec<F> = crate::utils::unsafe_allocate_zero_vec(pad_size);
                padded[..poly_in_coeffs.len()].copy_from_slice(&transposed);

                // encode and find the codeword that would be committed to
                let codeword = Radix2DitParallel::default()
                    .dft_algebra_batch(RowMajorMatrix::new(padded, width));

                // simulate sumcheck folding
                let mut poly_in_evals = poly_in_evals.clone();
                let rs: Point<F> = (0..folding)
                    .map(|_| {
                        let r: F = rng.random();
                        poly_in_evals.fix_var(r);
                        r
                    })
                    .collect::<Vec<_>>()
                    .into();

                // poly to send to verifier
                let poly_in_coeffs = poly_in_evals.to_coeffs();

                // find the domain generator
                let domain_size = rate - folding;
                let omega = F::two_adic_generator(domain_size);

                // check for each index
                for (index, row) in codeword.rows().enumerate() {
                    let wi = omega.exp_u64(index as u64);
                    let point = Point::<F>::expand(poly_in_coeffs.k(), wi);
                    let e0 = poly_in_coeffs.eval(&point);
                    let e1 = poly_in_coeffs.eval_univariate(wi);
                    assert_eq!(e0, e1);

                    let row: Poly<F, Coeff> = row.collect::<Vec<_>>().into();
                    let e1 = row.eval(&rs.reversed());
                    assert_eq!(e0, e1);
                }
            }
        }
    }

    fn run_whir_rust_crypto<
        F: TwoAdicField + Ord + SerializedField,
        Ext: ExtensionField<F> + TwoAdicField + Ord + SerializedField,
    >(
        k: usize,
        folding: usize,
        rate: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        n_points: usize,
    ) where
        StandardUniform: Distribution<F>,
    {
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        type Hasher = RustCryptoHasher<sha3::Keccak256>;
        type Compress = RustCryptoCompress<sha3::Keccak256>;

        let hasher = Hasher::default();
        let compress = Compress::default();

        let comm = MerkleTree::<F, F, [u8; 32], _, _>::new(hasher.clone(), compress.clone());
        let comm_ext = MerkleTree::<F, Ext, [u8; 32], _, _>::new(hasher, compress);

        let whir = Whir::new(k, folding, rate, soundness, security_level, comm, comm_ext);

        let mut rng = &mut crate::test::rng(1);
        let (proof, checkpoint_prover) = {
            let poly_in_coeffs: Poly<F, Coeff> = Poly::rand(&mut rng, k);
            let poly_in_evals = poly_in_coeffs.clone().to_evals();

            let mut transcript = Writer::init("test");
            let data = whir
                .commit::<_, Radix2DitParallel<F>>(&mut transcript, &poly_in_coeffs)
                .unwrap();

            let claims: Vec<Claim<Ext, Ext>> =
                make_claims_ext::<_, F, F, Ext>(&mut transcript, n_points, &poly_in_evals).unwrap();
            whir.open::<_, Radix2DitParallel<Ext>>(&mut transcript, claims, data, poly_in_coeffs)
                .unwrap();

            let checkpoint: F = transcript.draw();
            (transcript.finalize(), checkpoint)
        };

        let checkpoint_verifier = {
            let mut transcript = Reader::init(&proof, "test");
            let comm: [u8; 32] = transcript.read().unwrap();
            let claims = get_claims_ext::<_, F, Ext>(&mut transcript, whir.k, n_points).unwrap();
            whir.verify(&mut transcript, claims, comm).unwrap();
            let checkpoint: F = transcript.draw();
            checkpoint
        };

        assert_eq!(checkpoint_prover, checkpoint_verifier);
    }

    #[test]
    fn test_whir_rust_crypto() {
        type F = p3_goldilocks::Goldilocks;
        type Ext = BinomialExtensionField<F, 2>;

        let soundness_type = [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
            SecurityAssumption::UniqueDecoding,
        ];

        for soundness in soundness_type {
            for k in 4..=10 {
                for folding in 1..=3 {
                    for n_points in 1..=4 {
                        run_whir_rust_crypto::<F, Ext>(k, folding, 1, soundness, 32, n_points);
                    }
                }
            }
        }
    }

    fn run_whir_poseidon(
        k: usize,
        folding: usize,
        rate: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        n_points: usize,
    ) {
        type F = p3_baby_bear::BabyBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Perm = Poseidon2BabyBear<16>;
        type Hasher = PoseidonHasher<F, Perm, 16, 8, 8>;
        type Compress = PoseidonCompress<F, Perm, 2, 8, 16>;
        type Digest = [F; 8];
        type Challenger = DuplexChallenger<F, Perm, 16, 8>;
        type Writer = PoseidonWriter<Vec<u8>, F, Challenger>;
        type Reader<'a> = PoseidonReader<&'a [u8], F, Challenger>;

        let perm = Perm::new_from_rng_128(&mut crate::test::rng(1000));
        let hasher = Hasher::new(perm.clone());
        let compress = Compress::new(perm.clone());
        let challenger = Challenger::new(perm.clone());

        let comm = MerkleTree::<F, F, Digest, _, _>::new(hasher.clone(), compress.clone());
        let comm_ext = MerkleTree::<F, Ext, Digest, _, _>::new(hasher, compress);
        let whir = Whir::new(k, folding, rate, soundness, security_level, comm, comm_ext);

        let mut rng = &mut crate::test::rng(1);
        let (proof, checkpoint_prover) = {
            let poly_in_coeffs: Poly<F, Coeff> = Poly::rand(&mut rng, k);
            let poly_in_evals = poly_in_coeffs.clone().to_evals();

            let mut transcript = Writer::init(challenger);
            let data = whir
                .commit::<_, Radix2DitParallel<F>>(&mut transcript, &poly_in_coeffs)
                .unwrap();

            let claims =
                make_claims_ext::<_, F, F, Ext>(&mut transcript, n_points, &poly_in_evals).unwrap();

            whir.open::<_, Radix2DitParallel<Ext>>(&mut transcript, claims, data, poly_in_coeffs)
                .unwrap();

            let checkpoint: F = transcript.draw();
            (transcript.finalize(), checkpoint)
        };

        let checkpoint_verifier = {
            let challenger = Challenger::new(perm.clone());
            let mut transcript = Reader::init(&proof, challenger);
            let comm: Digest = transcript.read().unwrap();
            let claims = get_claims_ext::<_, F, Ext>(&mut transcript, whir.k, n_points).unwrap();
            whir.verify(&mut transcript, claims, comm).unwrap();
            let checkpoint: F = transcript.draw();
            checkpoint
        };
        assert_eq!(checkpoint_prover, checkpoint_verifier);
    }

    #[test]
    fn test_whir_poseidon() {
        let soundness_type = [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
            SecurityAssumption::UniqueDecoding,
        ];
        for soundness in soundness_type {
            for k in 4..=10 {
                for folding in 1..=3 {
                    for n_points in 1..=4 {
                        run_whir_poseidon(k, folding, 1, soundness, 32, n_points);
                    }
                }
            }
        }
    }
}
