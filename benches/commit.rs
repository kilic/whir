use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::Poseidon2KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::StdRng;
use rand::SeedableRng;
use whir::merkle::poseidon_packed::PackedPoseidonMerkleTree;
use whir::merkle::MerkleTree;
use whir::merkle::MerkleTreeExt;
use whir::transcript::test_transcript::TestWriter;
use whir::utils::n_rand;

criterion_group!(benches, commit_base, commit_ext);
criterion_main!(benches);

fn commit_base(c: &mut Criterion) {
    type F = p3_koala_bear::KoalaBear;
    type Poseidon16 = Poseidon2KoalaBear<16>;
    type Poseidon24 = Poseidon2KoalaBear<24>;
    type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
    type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
    type Writer = TestWriter<Vec<u8>, F>;

    let perm16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(16));
    let perm24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(24));

    let hasher = Hasher::new(perm24.clone());
    let compress = Compress::new(perm16.clone());

    let mut rng = StdRng::seed_from_u64(0u64);
    let k = 26;
    let folding = 5;
    let coeffs: Vec<F> = n_rand(&mut rng, 1 << k);
    let data = RowMajorMatrix::new(coeffs, 1 << folding);

    let mt_packed = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());
    let mut group = c.benchmark_group("commit_base".to_string());
    group.sample_size(10);
    group.bench_function(BenchmarkId::new("packed", format!("{k}")), |bencher| {
        bencher.iter_batched(
            || data.clone(),
            |data| {
                let mut transcript = Writer::init();
                MerkleTree::commit(&mt_packed, &mut transcript, black_box(data)).unwrap();
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn commit_ext(c: &mut Criterion) {
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;
    type Poseidon16 = Poseidon2KoalaBear<16>;
    type Poseidon24 = Poseidon2KoalaBear<24>;
    type Hasher = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
    type Compress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
    type Writer = TestWriter<Vec<u8>, F>;

    let perm16 = Poseidon16::new_from_rng_128(&mut StdRng::seed_from_u64(16));
    let perm24 = Poseidon24::new_from_rng_128(&mut StdRng::seed_from_u64(24));

    let hasher = Hasher::new(perm24.clone());
    let compress = Compress::new(perm16.clone());

    let mut rng = StdRng::seed_from_u64(0u64);
    let k = 26;
    let folding = 5;
    let coeffs: Vec<Ext> = n_rand(&mut rng, 1 << k);
    let data = RowMajorMatrix::new(coeffs, 1 << folding);
    let mt_packed = PackedPoseidonMerkleTree::new(hasher.clone(), compress.clone());

    let mut group = c.benchmark_group("commit_ext".to_string());

    group.bench_function(BenchmarkId::new("packed", "{k}"), |bencher| {
        bencher.iter_batched(
            || data.clone(),
            |data| {
                let mut transcript = Writer::init();
                MerkleTreeExt::commit(&mt_packed, &mut transcript, black_box(data)).unwrap();
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}
