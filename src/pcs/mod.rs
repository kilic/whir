use crate::poly::{Point, Poly};
use p3_field::{ExtensionField, Field};

pub mod params;
pub mod sumcheck;
pub mod whir;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EqClaim<F: Field, Ext: ExtensionField<F>> {
    pub(crate) point: Point<F>,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> EqClaim<F, Ext> {
    pub fn new(point: Point<F>, eval: Ext) -> Self {
        Self { point, eval }
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }

    pub fn eq(&self) -> Poly<F> {
        self.point.eq(F::ONE)
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn point(&self) -> Point<F> {
        self.point.clone()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PowClaim<F: Field, Ext: ExtensionField<F>> {
    pub(crate) k: usize,
    pub(crate) var: F,
    pub(crate) eval: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> PowClaim<F, Ext> {
    pub fn new(var: F, eval: Ext, k: usize) -> Self {
        Self { var, eval, k }
    }

    pub fn pow(&self, k: usize) -> Poly<F> {
        self.var.powers().take(k).collect().into()
    }

    pub fn eval(&self) -> &Ext {
        &self.eval
    }

    pub fn var(&self) -> F {
        self.var
    }

    pub fn point(&self) -> Point<F> {
        Point::expand(self.k, self.var)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        field::SerializedField,
        merkle::{poseidon_packed::PackedPoseidonMerkleTree, rust_crypto::RustCryptoMerkleTree},
        pcs::{EqClaim, PowClaim, params::SecurityAssumption, whir::Whir},
        poly::{Point, Poly},
        transcript::{
            Challenge, Reader, Writer,
            poseidon::{PoseidonReader, PoseidonWriter},
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
        },
    };
    use p3_challenger::DuplexChallenger;
    use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
    use p3_field::{ExtensionField, Field, TwoAdicField, extension::BinomialExtensionField};
    use p3_koala_bear::Poseidon2KoalaBear;
    use p3_matrix::{Matrix, dense::RowMajorMatrixView};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    };

    #[test]
    fn test_simple_whir() {
        type F = p3_goldilocks::Goldilocks;
        use p3_field::PrimeCharacteristicRing;
        use p3_field::TwoAdicField;

        pub fn encode<F: TwoAdicField>(poly: &Poly<F>, rate: usize, folding: usize) -> Poly<F> {
            let width = 1 << (poly.k() - folding);
            let mut mat = RowMajorMatrixView::new(poly, width).transpose();
            mat.pad_to_height(1 << (poly.k() - folding + rate), F::ZERO);
            Radix2DFTSmallBatch::<F>::default()
                .dft_batch(mat)
                .values
                .into()
        }

        fn fold<F: TwoAdicField>(r: &Point<F>, evals: &Poly<F>) -> Poly<F> {
            let mut evals = evals.clone();
            r.iter().for_each(|&r| evals.fix_var_mut(r));
            evals
        }

        fn fold_reversed<F: TwoAdicField>(r: &Point<F>, evals: &Poly<F>) -> Poly<F> {
            let mut evals = evals.clone().reverse_index_bits();
            r.iter().for_each(|&r| evals.fix_var_mut(r));
            evals.reverse_index_bits()
        }

        fn query(i: usize, folding: usize, cw: &Poly<F>) -> Poly<F> {
            cw[i << folding..(i + 1) << folding].to_vec().into()
        }

        let mut rng = &mut crate::test::rng(1);

        let k = 5usize;
        let f0 = Poly::<F>::rand(&mut rng, k);

        {
            let var: F = rng.random();
            let z = Point::<F>::expand(f0.k(), var);
            let y = f0.eval(&z);
            let eq = z.eq(F::ONE);

            let mut acc = F::ZERO;
            for i in 0..1 << k {
                let point = Point::<F>::hypercube(i, k);
                acc += f0.eval(&point) * eq.eval(&point);
            }
            assert_eq!(y, acc);
        }

        for folding in 0..k {
            for rate in 0..k {
                let alpha = Point::<F>::rand(&mut rng, folding);
                let f1 = fold(&alpha, &f0);
                let f0_cw = encode(&f0, rate, folding);

                let f1_cw = fold_reversed(&alpha.reversed(), &f0_cw);
                assert_eq!(f1_cw, encode(&f1, rate, 0));

                let k_domain = k + rate - folding;
                let omega = F::two_adic_generator(k_domain);

                for i in 0..1 << k_domain {
                    let z = omega.exp_u64(i as u64);
                    let y0 = f1.eval_univariate(z);
                    let y1 = f1_cw[i];
                    assert_eq!(y0, y1);
                    let local_poly = query(i, folding, &f0_cw);
                    assert_eq!(y0, local_poly.eval(&alpha.reversed()));
                }
            }
        }
    }

    #[test]
    fn test_simple_whir_no_transpose() {
        type F = p3_goldilocks::Goldilocks;
        use p3_field::PrimeCharacteristicRing;
        use p3_field::TwoAdicField;

        pub fn encode<F: TwoAdicField>(poly: &Poly<F>, rate: usize, folding: usize) -> Poly<F> {
            let width = 1 << folding;
            let mut mat = RowMajorMatrixView::new(poly, width).to_row_major_matrix();
            mat.pad_to_height(1 << (poly.k() - folding + rate), F::ZERO);
            Radix2DFTSmallBatch::<F>::default()
                .dft_batch(mat)
                .values
                .into()
        }
        // should be equivalent:
        // pub fn encode<F: TwoAdicField>(poly: &Poly<F>, rate: usize, folding: usize) -> Poly<F> {
        //     let mut padded = F::zero_vec(poly.len() * (1 << rate));
        //     padded[..poly.len()].copy_from_slice(poly.as_slice());
        //     let padded = RowMajorMatrix::new(padded, 1 << folding);
        //     Radix2DFTSmallBatch::<F>::default()
        //         .dft_batch(padded)
        //         .values
        //         .into()
        // }

        fn fold_reversed<F: TwoAdicField>(r: &Point<F>, evals: &Poly<F>) -> Poly<F> {
            let mut evals = evals.clone().reverse_index_bits();
            r.iter().for_each(|&r| evals.fix_var_mut(r));
            evals.reverse_index_bits()
        }

        fn query(i: usize, folding: usize, cw: &Poly<F>) -> Poly<F> {
            cw[i << folding..(i + 1) << folding].to_vec().into()
        }

        let mut rng = &mut crate::test::rng(1);

        let k = 5usize;
        let f0 = Poly::<F>::rand(&mut rng, k);

        {
            let var: F = rng.random();
            let z = Point::<F>::expand(f0.k(), var);
            let y = f0.eval(&z);
            let eq = z.eq(F::ONE);

            let mut acc = F::ZERO;
            for i in 0..1 << k {
                let point = Point::<F>::hypercube(i, k);
                acc += f0.eval(&point) * eq.eval(&point);
            }
            assert_eq!(y, acc);
        }

        for folding in 0..k {
            for rate in 0..k {
                let alpha = Point::<F>::rand(&mut rng, folding);
                let f1 = fold_reversed(&alpha, &f0);
                let f0_cw = encode(&f0, rate, folding);

                let f1_cw = fold_reversed(&alpha, &f0_cw);
                assert_eq!(f1_cw, encode(&f1, rate, 0));

                let k_domain = k + rate - folding;
                let omega = F::two_adic_generator(k_domain);

                for i in 0..k_domain {
                    let z = omega.exp_u64(i as u64);
                    let y0 = f1.eval_univariate(z);
                    let y1 = f1_cw[i];
                    assert_eq!(y0, y1);
                    let local_poly = query(i, folding, &f0_cw);
                    assert_eq!(y0, local_poly.eval(&alpha));
                }
            }
        }
    }

    pub(crate) fn make_eq_claims_base<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        n_points: usize,
        poly: &Poly<F>,
    ) -> Result<Vec<EqClaim<Ext, Ext>>, crate::Error>
    where
        Transcript: Challenge<F, Ext> + Writer<Ext>,
    {
        (0..n_points)
            .map(|_| {
                let point = Point::expand(poly.k(), transcript.draw());
                let eval = poly.eval(&point);
                let claim = EqClaim { point, eval };
                transcript.write(claim.eval)?;
                Ok(claim)
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn make_eq_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        n_points: usize,
        poly: &Poly<Ext>,
    ) -> Result<Vec<EqClaim<Ext, Ext>>, crate::Error>
    where
        Transcript: Challenge<F, Ext> + Writer<Ext>,
    {
        (0..n_points)
            .map(|_| {
                let point = Point::expand(poly.k(), transcript.draw());
                let eval = poly.eval(&point);
                let claim = EqClaim { point, eval };
                transcript.write(claim.eval)?;
                Ok(claim)
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn make_pow_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        n_points: usize,
        poly: &Poly<Ext>,
    ) -> Result<Vec<PowClaim<F, Ext>>, crate::Error>
    where
        Transcript: Challenge<F, F> + Writer<Ext>,
    {
        let k = poly.k();
        (0..n_points)
            .map(|_| {
                let var = transcript.draw();
                let eval = poly
                    .iter()
                    .zip(var.powers().take(poly.len()))
                    .map(|(&c, p)| c * p)
                    .sum();
                let claim = PowClaim { var, eval, k };
                transcript.write(claim.eval)?;
                Ok(claim)
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn get_eq_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        k: usize,
        n: usize,
    ) -> Result<Vec<EqClaim<Ext, Ext>>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        (0..n)
            .map(|_| {
                let var: Ext = transcript.draw();
                let point = Point::expand(k, var);
                let eval = transcript.read()?;
                Ok(EqClaim::new(point, eval))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    pub(crate) fn get_pow_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        k: usize,
        n: usize,
    ) -> Result<Vec<PowClaim<F, Ext>>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, F>,
    {
        (0..n)
            .map(|_| {
                let var: F = transcript.draw();
                let eval = transcript.read()?;
                Ok(PowClaim::new(var, eval, k))
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn run_whir_rust_crypto<
        F: TwoAdicField + Ord + SerializedField,
        Ext: ExtensionField<F> + TwoAdicField + Ord + SerializedField,
    >(
        k: usize,
        folding: usize,
        rate: usize,
        initial_reduction: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        n_points: usize,
    ) where
        StandardUniform: Distribution<F>,
    {
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        let mt = RustCryptoMerkleTree::<sha2::Sha256>::default();
        let mt_ext = RustCryptoMerkleTree::<sha2::Sha256>::default();

        let whir = Whir::new(
            k,
            folding,
            rate,
            initial_reduction,
            soundness,
            security_level,
            mt,
            mt_ext,
        );

        let mut rng = &mut crate::test::rng(1);
        let dft = &tracing::info_span!("prepare dft")
            .in_scope(|| Radix2DFTSmallBatch::new(1 << (k - folding + rate)));
        let (proof, checkpoint_prover) = {
            let poly = Poly::<F>::rand(&mut rng, k);

            let mut transcript = Writer::init("test");
            let data = whir.commit(dft, &mut transcript, &poly).unwrap();

            let claims: Vec<EqClaim<Ext, Ext>> =
                make_eq_claims_base::<_, F, Ext>(&mut transcript, n_points, &poly).unwrap();
            whir.open(dft, &mut transcript, claims, data, poly).unwrap();

            let checkpoint: F = transcript.draw();
            (transcript.finalize(), checkpoint)
        };

        let checkpoint_verifier = {
            let mut transcript = Reader::init(&proof, "test");
            let comm: [u8; 32] = transcript.read().unwrap();
            let claims = get_eq_claims::<_, F, Ext>(&mut transcript, whir.k, n_points).unwrap();
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
            for k in 4..=15 {
                for folding in 1..=3 {
                    for initial_reduction in 1..=folding {
                        for n_points in 1..=4 {
                            run_whir_rust_crypto::<F, Ext>(
                                k,
                                folding,
                                1,
                                initial_reduction,
                                soundness,
                                32,
                                n_points,
                            );
                        }
                    }
                }
            }
        }
    }

    fn run_whir_poseidon(
        k: usize,
        folding: usize,
        rate: usize,
        initial_reduction: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        n_points: usize,
    ) {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Poseidon16 = Poseidon2KoalaBear<16>;
        type Poseidon24 = Poseidon2KoalaBear<24>;

        type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
        type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
        type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
        type Digest = [F; 8];
        type Writer = PoseidonWriter<Vec<u8>, F, Challenger>;
        type Reader<'a> = PoseidonReader<&'a [u8], F, Challenger>;

        let perm16 = Poseidon16::new_from_rng_128(&mut crate::test::rng(1000));
        let perm24 = Poseidon24::new_from_rng_128(&mut crate::test::rng(1000));

        let hasher = Hasher::new(perm24.clone());
        let compress = Compress::new(perm16.clone());
        let challenger = Challenger::new(perm16.clone());

        let whir = Whir::new(
            k,
            folding,
            rate,
            initial_reduction,
            soundness,
            security_level,
            PackedPoseidonMerkleTree::<F, _, _, 8>::new(hasher.clone(), compress.clone()),
            PackedPoseidonMerkleTree::<F, _, _, 8>::new(hasher.clone(), compress.clone()),
        );

        let mut rng = &mut crate::test::rng(1);

        let dft = &tracing::info_span!("prepare dft")
            .in_scope(|| Radix2DFTSmallBatch::new(1 << (k - folding + rate)));
        let (proof, checkpoint_prover) = {
            let poly = Poly::<F>::rand(&mut rng, k);

            let mut transcript = Writer::init(challenger);
            let data = tracing::info_span!("commit")
                .in_scope(|| whir.commit(dft, &mut transcript, &poly).unwrap());

            let claims =
                make_eq_claims_base::<_, F, Ext>(&mut transcript, n_points, &poly).unwrap();

            tracing::info_span!("open").in_scope(|| {
                whir.open(dft, &mut transcript, claims, data, poly).unwrap();
            });

            let checkpoint: F = transcript.draw();
            (transcript.finalize(), checkpoint)
        };

        let checkpoint_verifier = {
            let challenger = Challenger::new(perm16.clone());
            let mut transcript = Reader::init(&proof, challenger);
            let comm: Digest = transcript.read().unwrap();
            let claims = get_eq_claims::<_, F, Ext>(&mut transcript, whir.k, n_points).unwrap();
            whir.verify(&mut transcript, claims, comm).unwrap();
            transcript.draw()
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
            for k in 4..=12 {
                for folding in 1..=3 {
                    for initial_reduction in 1..=folding {
                        for n_points in 1..=4 {
                            run_whir_poseidon(
                                k,
                                folding,
                                1,
                                initial_reduction,
                                soundness,
                                32,
                                n_points,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn whir_bench() {
        // run with : RUSTFLAGS='-C target-cpu=native'
        crate::test::init_tracing();
        run_whir_poseidon(25, 5, 1, 3, SecurityAssumption::CapacityBound, 90, 1);
    }
}
