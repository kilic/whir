pub mod field;
pub mod merkle;
pub mod pcs;
pub mod poly;
pub mod study;
pub mod transcript;
pub mod utils;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Error {
    Transcript,
    Verify,
    IO,
}

#[cfg(test)]
pub mod test {
    use rand::{rngs::StdRng, SeedableRng};

    #[allow(dead_code)]
    pub(crate) fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[allow(dead_code)]
    pub(crate) fn init_tracing() {
        use tracing_forest::util::LevelFilter;
        use tracing_forest::ForestLayer;
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

    pub(crate) fn bench<F, T>(name: &str, n_iter: usize, expected: Option<&T>, f: F)
    where
        F: Fn() -> T,
        T: PartialEq + std::fmt::Debug,
    {
        let mut lowest = std::time::Duration::MAX;
        let mut total = std::time::Duration::default();

        for _ in 0..n_iter {
            let start = std::time::Instant::now();
            let result = std::hint::black_box(f());
            let this_time = start.elapsed();
            lowest = lowest.min(this_time);
            total += this_time;

            if let Some(expected) = expected {
                assert_eq!(*expected, result, "{name}");
            }
        }

        let avg = total / n_iter as u32;
        println!(
            "{:20} N: {n_iter} Avg: {:8.2?} Lowest: {:8.2?}",
            name, avg, lowest
        );
    }
}
