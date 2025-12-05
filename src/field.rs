pub trait SerializedField: p3_field::Field {
    fn from_bytes(bytes: &[u8]) -> Result<Self, crate::Error>;
    fn to_bytes(&self) -> Result<Vec<u8>, crate::Error>;
}

// pub use SerializedField as Field;

macro_rules! impl_from_uniform_bytes {
    ($field:ty) => {
        impl SerializedField for $field {
            fn from_bytes(bytes: &[u8]) -> Result<Self, crate::Error> {
                bincode::deserialize(bytes).map_err(|_| crate::Error::IO)
            }

            fn to_bytes(&self) -> Result<Vec<u8>, crate::Error> {
                bincode::serialize(&self).map_err(|_| crate::Error::IO)
            }
        }
    };
}

impl_from_uniform_bytes!(p3_goldilocks::Goldilocks);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>);
impl_from_uniform_bytes!(p3_baby_bear::BabyBear);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_baby_bear::BabyBear, 4>);
impl_from_uniform_bytes!(p3_koala_bear::KoalaBear);
impl_from_uniform_bytes!(p3_field::extension::BinomialExtensionField<p3_koala_bear::KoalaBear, 4>);

#[test]
fn test_serialization() {
    let n = 1000000;

    use rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    };

    fn run_test<F: SerializedField>(rng: &mut impl Rng, n: usize)
    where
        StandardUniform: Distribution<F>,
    {
        for _ in 0..n {
            let a0: F = rng.random();
            let bytes = a0.to_bytes().unwrap();
            let a1 = F::from_bytes(&bytes).unwrap();
            assert_eq!(a0, a1);
        }
    }

    let mut rng = crate::test::rng(0);

    run_test::<p3_field::extension::BinomialExtensionField<p3_goldilocks::Goldilocks, 2>>(
        &mut rng, n,
    );
    run_test::<p3_field::extension::BinomialExtensionField<p3_baby_bear::BabyBear, 4>>(&mut rng, n);
    run_test::<p3_field::extension::BinomialExtensionField<p3_koala_bear::KoalaBear, 4>>(
        &mut rng, n,
    );
}
