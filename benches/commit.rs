use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::Poseidon2KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::PaddingFreeSponge;
use rand::rngs::StdRng;
use rand::SeedableRng;
use whir::merkle::comm::Commitment;
use whir::merkle::poseidon::{PoseidonCompress, PoseidonHasher};
use whir::merkle::MerkleTree;
use whir::transcript::poseidon::PoseidonWriter;
use whir::utils::n_rand;

criterion_group!(benches, commit_base, commit_ext);
criterion_main!(benches);

fn commit_base(c: &mut Criterion) {
    type F = p3_koala_bear::KoalaBear;
    type Poseidon16 = Poseidon2KoalaBear<16>;
    type Poseidon24 = Poseidon2KoalaBear<24>;

    type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
    type Compress = PoseidonCompress<F, Poseidon16, 2, 8, 16>;
    type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;

    type Writer = PoseidonWriter<Vec<u8>, F, Challenger>;

    let perm16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(16));
    let perm24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(24));

    let hasher = PoseidonHasher::<F, _, 8>::new(Hasher::new(perm24));
    let compress = Compress::new(perm16.clone());
    let challenger = Challenger::new(perm16);

    let merkle_tree = MerkleTree::<F, [F; 8], _, _>::new(hasher, compress);

    let mut rng = StdRng::seed_from_u64(0u64);
    let k = 25;
    let folding = 5;
    let coeffs: Vec<F> = n_rand(&mut rng, 1 << k);
    let data = RowMajorMatrix::new(coeffs, 1 << folding);

    let mut group = c.benchmark_group("commit_base".to_string());
    group.sample_size(10);
    group.bench_function(
        BenchmarkId::new("with-base-api", format!("{k}")),
        |bencher| {
            bencher.iter_batched(
                || (data.clone(), challenger.clone()),
                |(data, challenger)| {
                    let mut transcript = Writer::init(challenger);
                    merkle_tree
                        .commit_base::<_>(&mut transcript, black_box(data))
                        .unwrap();
                },
                criterion::BatchSize::LargeInput,
            );
        },
    );

    group.bench_function(
        BenchmarkId::new("with-ext-api", format!("{k}")),
        |bencher| {
            bencher.iter_batched(
                || (data.clone(), challenger.clone()),
                |(data, challenger)| {
                    let mut transcript = Writer::init(challenger);
                    merkle_tree
                        .commit_ext::<_, F>(&mut transcript, black_box(data))
                        .unwrap();
                },
                criterion::BatchSize::LargeInput,
            );
        },
    );

    group.finish();
}

fn commit_ext(c: &mut Criterion) {
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;
    type Poseidon16 = Poseidon2KoalaBear<16>;
    type Poseidon24 = Poseidon2KoalaBear<24>;

    type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
    type Compress = PoseidonCompress<F, Poseidon16, 2, 8, 16>;
    type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;

    type Writer = PoseidonWriter<Vec<u8>, F, Challenger>;

    let perm16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(16));
    let perm24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(24));

    let hasher = PoseidonHasher::<F, _, 8>::new(Hasher::new(perm24));
    let compress = Compress::new(perm16.clone());
    let challenger = Challenger::new(perm16);

    let merkle_tree = MerkleTree::<F, [F; 8], _, _>::new(hasher, compress);

    let mut rng = StdRng::seed_from_u64(0u64);
    let k = 20;
    let folding = 5;
    let coeffs: Vec<Ext> = n_rand(&mut rng, 1 << k);
    let data = RowMajorMatrix::new(coeffs, 1 << folding);

    let mut group = c.benchmark_group("commit_ext".to_string());

    group.bench_function(
        BenchmarkId::new("commit-base-with-ext-api", "{k}"),
        |bencher| {
            bencher.iter_batched(
                || (data.clone(), challenger.clone()),
                |(data, challenger)| {
                    let mut transcript = Writer::init(challenger);
                    merkle_tree
                        .commit_ext::<_, _>(&mut transcript, black_box(data))
                        .unwrap();
                },
                criterion::BatchSize::LargeInput,
            );
        },
    );

    group.finish();
}
