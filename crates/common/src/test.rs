use rand::{SeedableRng, rngs::StdRng};
pub fn rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

pub fn init_tracing() {
    use tracing_forest::ForestLayer;
    use tracing_forest::util::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Registry};

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

pub fn bench<Perf, PerfOut, Setup, SetupOut, PreMut, PreMutOut, Pre, PreOut, Post>(
    name: &str,
    n_iter: usize,
    setup: Setup,
    pre_mut: PreMut,
    pre: Pre,
    perf: Perf,
    post: Post,
) where
    Setup: Fn() -> SetupOut,
    PreMut: Fn(&SetupOut) -> PreMutOut,
    Pre: Fn(&SetupOut) -> PreOut,
    Perf: Fn(&SetupOut, PreOut, &mut PreMutOut) -> PerfOut,
    Post: Fn(PreMutOut, PerfOut),
{
    let mut lowest = std::time::Duration::MAX;
    let mut total = std::time::Duration::default();

    let setup_out = setup();
    for _ in 0..n_iter {
        let mut pre_mut_out = pre_mut(&setup_out);
        let pre_out = pre(&setup_out);
        let start = std::time::Instant::now();
        let result = std::hint::black_box(perf(&setup_out, pre_out, &mut pre_mut_out));
        let perf_time = start.elapsed();
        lowest = lowest.min(perf_time);
        total += perf_time;
        post(pre_mut_out, result);
    }

    let avg = total / n_iter as u32;
    println!(
        "{:20} N: {n_iter} Avg: {:8.2?} Lowest: {:8.2?}",
        name, avg, lowest
    );
}
