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
    use p3_field::{ExtensionField, Field};
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

    pub(crate) fn bench<Perf, PerfOut, Setup, SetupOut, Pre, PreOut, Post>(
        name: &str,
        n_iter: usize,
        setup: Setup,
        pre: Pre,
        perf: Perf,
        post: Post,
    ) where
        Setup: Fn() -> SetupOut,
        Pre: Fn(&SetupOut) -> PreOut,
        Perf: Fn(&SetupOut, &mut PreOut) -> PerfOut,
        Post: Fn(PreOut, PerfOut),
    {
        let mut lowest = std::time::Duration::MAX;
        let mut total = std::time::Duration::default();

        let setup_out = setup();
        for _ in 0..n_iter {
            let mut pre_out = pre(&setup_out);
            let start = std::time::Instant::now();
            let result = std::hint::black_box(perf(&setup_out, &mut pre_out));
            let perf_time = start.elapsed();
            lowest = lowest.min(perf_time);
            total += perf_time;
            post(pre_out, result);
        }

        let avg = total / n_iter as u32;
        println!(
            "{:20} N: {n_iter} Avg: {:8.2?} Lowest: {:8.2?}",
            name, avg, lowest
        );
    }

    pub(crate) fn unpack_ext<F: Field, Ext: ExtensionField<F>>(
        packed: Vec<Ext::ExtensionPacking>,
    ) -> Vec<Ext> {
        use p3_field::PackedFieldExtension;
        Ext::ExtensionPacking::to_ext_iter(packed.into_iter()).collect()
    }
}
