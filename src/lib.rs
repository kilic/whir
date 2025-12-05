pub mod field;
pub mod merkle;
pub mod pcs;
pub mod poly;
pub mod transcript;
pub mod utils;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Error {
    Transcript,
    Verify,
    IO,
}

pub(crate) mod p3_field_prelude {
    #[allow(unused)]
    pub(crate) use p3_field::{
        Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
        PrimeCharacteristicRing, dot_product, extension::BinomialExtensionField,
        extension::PackedBinomialExtensionField,
    };
}

#[cfg(test)]
pub mod test {
    use rand::{SeedableRng, rngs::StdRng};

    #[allow(dead_code)]
    pub(crate) fn rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[allow(dead_code)]
    pub(crate) fn init_tracing() {
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
}
