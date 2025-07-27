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
