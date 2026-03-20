use crate::params::SecurityAssumption;
use crate::{
    sumcheck::{prover::ProverStack, verifier::VerifierStack},
    whir::Whir,
};

use merkle::poseidon_packed::PackedPoseidonMerkleTree;
use p3_challenger::DuplexChallenger;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::Poseidon2KoalaBear;
use p3_matrix::{Matrix, dense::RowMajorMatrixView};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use poly::{Point, Poly};
use rand::RngExt;
use transcript::{
    Challenge, Reader,
    poseidon::{PoseidonReader, PoseidonWriter},
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
        r.iter().for_each(|&r| evals.fix_hi_var_mut(r));
        evals
    }

    fn fold_reversed<F: TwoAdicField>(r: &Point<F>, evals: &Poly<F>) -> Poly<F> {
        let mut evals = evals.clone().reverse_vars();
        r.iter().for_each(|&r| evals.fix_hi_var_mut(r));
        evals.reverse_vars()
    }

    fn query(i: usize, folding: usize, cw: &Poly<F>) -> Poly<F> {
        cw[i << folding..(i + 1) << folding].to_vec().into()
    }

    let mut rng = common::test::rng(1);
    let k = 5usize;
    let f0 = Poly::<F>::rand(&mut rng, k);

    {
        let var: F = rng.random();
        let z = Point::<F>::expand(f0.k(), var);
        let y = f0.eval_base(&z);
        let eq = z.eq(F::ONE);

        let mut acc = F::ZERO;
        for i in 0..1 << k {
            let point = Point::<F>::hypercube(i, k);
            acc += f0.eval_base(&point) * eq.eval_base(&point);
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
                assert_eq!(y0, local_poly.eval_base(&alpha.reversed()));
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
        let mut evals = evals.clone().reverse_vars();
        r.iter().for_each(|&r| evals.fix_hi_var_mut(r));
        evals.reverse_vars()
    }

    fn query(i: usize, folding: usize, cw: &Poly<F>) -> Poly<F> {
        cw[i << folding..(i + 1) << folding].to_vec().into()
    }

    let mut rng = common::test::rng(1);

    let k = 5usize;
    let f0 = Poly::<F>::rand(&mut rng, k);

    {
        let var: F = rng.random();
        let z = Point::<F>::expand(f0.k(), var);
        let y = f0.eval_base(&z);
        let eq = z.eq(F::ONE);

        let mut acc = F::ZERO;
        for i in 0..1 << k {
            let point = Point::<F>::hypercube(i, k);
            acc += f0.eval_base(&point) * eq.eval_base(&point);
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
                assert_eq!(y0, local_poly.eval_base(&alpha));
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

    let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
    let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

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

    let mut rng = common::test::rng(1);

    let dft = &tracing::info_span!("prepare dft")
        .in_scope(|| Radix2DFTSmallBatch::new(1 << (k - folding + rate)));

    let (proof, checkpoint_prover) = {
        let poly = Poly::<F>::rand(&mut rng, k);
        let mut transcript = Writer::init(challenger);
        let data = whir.commit(dft, &mut transcript, &poly).unwrap();
        let mut stack = ProverStack::new(whir.folding);
        let id = stack.register(poly);
        for _ in 0..n_points {
            let point = Point::expand(stack.k_poly(id), transcript.draw());
            stack.eval(&mut transcript, id, &point).unwrap();
        }

        whir.open(dft, &mut transcript, data, stack.layout())
            .unwrap();
        let checkpoint: F = transcript.draw();
        (transcript.finalize(), checkpoint)
    };

    let checkpoint_verifier = {
        let challenger = Challenger::new(perm16.clone());
        let mut transcript = Reader::init(&proof, challenger);
        let mut stack = VerifierStack::<F, Ext>::new(whir.folding);
        let id = stack.register(k);
        let comm: Digest = transcript.read().unwrap();
        (0..n_points)
            .map(|_| {
                let point = Point::expand(stack.k_poly(id), transcript.draw());
                stack.read_eval(&mut transcript, id, &point)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        whir.verify(&mut transcript, stack.layout(), comm).unwrap();
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
// #[ignore]
fn whir_bench1() {
    // run with : RUSTFLAGS='-C target-cpu=native'
    // common::test::init_tracing();
    run_whir_poseidon(25, 5, 1, 3, SecurityAssumption::JohnsonBound, 123, 1);
}

#[test]
fn test_leanvm_whir1() {
    // common::test::init_tracing();

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

    let perm16 = Poseidon16::new_from_rng_128(&mut common::test::rng(1000));
    let perm24 = Poseidon24::new_from_rng_128(&mut common::test::rng(1000));

    let hasher = Hasher::new(perm24.clone());
    let compress = Compress::new(perm16.clone());
    let challenger = Challenger::new(perm16.clone());

    let k = 25;
    let folding = 5;
    let rate = 1;
    let initial_reduction = 3;
    let soundness = SecurityAssumption::JohnsonBound;
    let security_level = 123;
    let whir: Whir<F, Ext, _, _> = Whir::new(
        k,
        folding,
        rate,
        initial_reduction,
        soundness,
        security_level,
        PackedPoseidonMerkleTree::<F, _, _, 8>::new(hasher.clone(), compress.clone()),
        PackedPoseidonMerkleTree::<F, _, _, 8>::new(hasher.clone(), compress.clone()),
    );

    let mut rng = common::test::rng(1);

    let mut transcript = Writer::init(challenger.clone());
    let mut stack = ProverStack::<F, Ext>::new(folding);
    let ids_a = (0..20)
        .map(|_| stack.register(Poly::<F>::rand(&mut rng, 20)))
        .collect::<Vec<_>>();
    let ids_b = (0..29)
        .map(|_| stack.register(Poly::<F>::rand(&mut rng, 8)))
        .collect::<Vec<_>>();
    let ids_c = (0..96)
        .map(|_| stack.register(Poly::<F>::rand(&mut rng, 8)))
        .collect::<Vec<_>>();

    tracing::info_span!("evals").in_scope(|| {
        let apply_shared_point = |stack: &mut ProverStack<F, Ext>,
                                  transcript: &mut Writer,
                                  ids: &[usize],
                                  k: usize,
                                  selected: &[usize]| {
            let point = Point::expand(k, transcript.draw());
            for &i in selected {
                stack.eval(transcript, ids[i], &point).unwrap();
            }
        };

        // Table A: 20 polys, k = 20
        let p0 = vec![
            0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ];
        let p1 = vec![0, 1];
        let p2 = (0..20).collect::<Vec<_>>();
        apply_shared_point(&mut stack, &mut transcript, &ids_a, 20, &p0);
        apply_shared_point(&mut stack, &mut transcript, &ids_a, 20, &p1);
        apply_shared_point(&mut stack, &mut transcript, &ids_a, 20, &p2);

        // Table B: 29 polys, k = 8
        let p3 = (6..=23).collect::<Vec<_>>();
        let p4 = (0..=7).chain(24..=28).collect::<Vec<_>>();
        let p5 = (0..29).collect::<Vec<_>>();
        apply_shared_point(&mut stack, &mut transcript, &ids_b, 8, &p3);
        apply_shared_point(&mut stack, &mut transcript, &ids_b, 8, &p4);
        apply_shared_point(&mut stack, &mut transcript, &ids_b, 8, &p5);

        // Table C: 96 polys, k = 8
        let p6 = (1..=19).chain(88..=95).collect::<Vec<_>>();
        let p7 = (0..96).collect::<Vec<_>>();
        apply_shared_point(&mut stack, &mut transcript, &ids_c, 8, &p6);
        apply_shared_point(&mut stack, &mut transcript, &ids_c, 8, &p7);
    });

    let dft = &tracing::info_span!("prepare dft")
        .in_scope(|| Radix2DFTSmallBatch::<F>::new(1 << (k - folding + rate)));
    let (proof, checkpoint_prover) = {
        let layout = stack.layout();
        let data = whir.commit(dft, &mut transcript, layout.poly()).unwrap();
        whir.open(dft, &mut transcript, data, layout).unwrap();
        let checkpoint: F = transcript.draw();
        (transcript.finalize(), checkpoint)
    };

    // Verifier
    let checkpont_verifier = {
        let mut transcript = Reader::init(&proof, challenger);
        let mut stack = VerifierStack::<F, Ext>::new(whir.folding);
        let ids_a_v = (0..20).map(|_| stack.register(20)).collect::<Vec<_>>();
        let ids_b_v = (0..29).map(|_| stack.register(8)).collect::<Vec<_>>();
        let ids_c_v = (0..96).map(|_| stack.register(8)).collect::<Vec<_>>();

        let read_shared_point = |stack: &mut VerifierStack<F, Ext>,
                                 transcript: &mut Reader,
                                 ids: &[usize],
                                 k: usize,
                                 selected: &[usize]| {
            let point = Point::expand(k, transcript.draw());
            for &i in selected {
                let _ = stack.read_eval(transcript, ids[i], &point).unwrap();
            }
        };

        // Table A: 20 polys, k = 20
        let p0 = vec![
            0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ];
        let p1 = vec![0, 1];
        let p2 = (0..20).collect::<Vec<_>>();
        read_shared_point(&mut stack, &mut transcript, &ids_a_v, 20, &p0);
        read_shared_point(&mut stack, &mut transcript, &ids_a_v, 20, &p1);
        read_shared_point(&mut stack, &mut transcript, &ids_a_v, 20, &p2);

        // Table B: 29 polys, k = 8
        let p3 = (6..=23).collect::<Vec<_>>();
        let p4 = (0..=7).chain(24..=28).collect::<Vec<_>>();
        let p5 = (0..29).collect::<Vec<_>>();
        read_shared_point(&mut stack, &mut transcript, &ids_b_v, 8, &p3);
        read_shared_point(&mut stack, &mut transcript, &ids_b_v, 8, &p4);
        read_shared_point(&mut stack, &mut transcript, &ids_b_v, 8, &p5);

        // Table C: 96 polys, k = 8
        let p6 = (1..=19).chain(88..=95).collect::<Vec<_>>();
        let p7 = (0..96).collect::<Vec<_>>();
        read_shared_point(&mut stack, &mut transcript, &ids_c_v, 8, &p6);
        read_shared_point(&mut stack, &mut transcript, &ids_c_v, 8, &p7);

        let comm: Digest = transcript.read().unwrap();
        whir.verify(&mut transcript, stack.layout(), comm).unwrap();
        transcript.draw()
    };
    assert_eq!(checkpoint_prover, checkpont_verifier);
}
