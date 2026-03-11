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
        let data = tracing::info_span!("commit")
            .in_scope(|| whir.commit(dft, &mut transcript, &poly).unwrap());
        let mut stack = ProverStack::new(whir.folding);
        let id = stack.register(poly);
        for _ in 0..n_points {
            let point = Point::expand(stack.k_poly(id), transcript.draw());
            stack.eval(&mut transcript, id, &point).unwrap();
        }
        tracing::info_span!("open").in_scope(|| {
            whir.open(dft, &mut transcript, data, stack.layout())
                .unwrap()
        });
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
fn whir_bench() {
    // run with : RUSTFLAGS='-C target-cpu=native'
    common::test::init_tracing();
    run_whir_poseidon(25, 5, 1, 3, SecurityAssumption::JohnsonBound, 123, 1);
}
