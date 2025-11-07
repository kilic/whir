use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_field::extension::BinomialExtensionField;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use whir::poly::eq_multi_p3;
use whir::poly::eq_multi_split;
use whir::poly::eq_single_p3;
use whir::poly::eq_single_split;
use whir::poly::Point;

criterion_group!(benches, eq_multi);
criterion_main!(benches);

// fn eq_single(c: &mut Criterion) {
//     type F = p3_koala_bear::KoalaBear;
//     type Ext = BinomialExtensionField<F, 4>;

//     let mut rng = StdRng::seed_from_u64(0u64);

//     let mut group = c.benchmark_group("eq_single".to_string());
//     // group.sample_size(10);

//     for k in 18..25 {
//         let point = Point::<Ext>::rand(&mut rng, k);

//         group.bench_function(BenchmarkId::new("eq_split_eq", format!("{k}")), |bencher| {
//             bencher.iter(|| black_box(eq_single_split::<F, Ext>(&point, k / 2)));
//         });

//         // group.bench_function(
//         //     BenchmarkId::new("eq_split_eq-alt", format!("{k}")),
//         //     |bencher| {
//         //         bencher.iter(|| {
//         //             let _ = eq_single_split::<F, Ext>(&point, k / 2);
//         //         });
//         //     },
//         // );

//         group.bench_function(BenchmarkId::new("p3", format!("{k}")), |bencher| {
//             bencher.iter(|| black_box(eq_single_p3::<F, Ext>(&point)));
//         });

//         // group.bench_function(BenchmarkId::new("p3-alt", format!("{k}")), |bencher| {
//         //     bencher.iter(|| {
//         //         let _ = eq_single_p3::<F, Ext>(&point);
//         //     });
//         // });
//     }

//     group.finish();
// }

fn eq_multi(c: &mut Criterion) {
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;

    let mut rng = StdRng::seed_from_u64(0u64);

    let mut group = c.benchmark_group("eq_multi".to_string());
    // group.sample_size(10);

    for k in 18..20 {
        for n in [1, 20, 100] {
            // let point = Point::<Ext>::rand(&mut rng, k)

            let points = (0..n)
                .map(|_| Point::<Ext>::rand(&mut rng, k))
                .collect::<Vec<_>>();
            let alpha: Ext = rng.random();

            group.bench_function(BenchmarkId::new("p3", format!("{k}, {n}")), |bencher| {
                bencher.iter(|| black_box(eq_multi_p3::<F, Ext>(&points, alpha)));
            });

            group.bench_function(BenchmarkId::new("split", format!("{k}, {n}")), |bencher| {
                bencher.iter(|| black_box(eq_multi_split::<F, Ext>(&points, alpha, k / 2)));
            });
        }
    }
    group.finish();
}
