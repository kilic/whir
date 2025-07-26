use digest::Digest;
use digest::FixedOutputReset;
use p3_field::ExtensionField;
use p3_field::Field;
use std::io::Read;
use std::io::Write;

use crate::transcript::Challenge;
use crate::transcript::FieldReader;
use crate::transcript::FieldWriter;

const RUST_CRYPTO_UPDATE: u8 = 1;

// TODO: add buffer to hash less
#[derive(Debug, Clone)]
pub struct RustCryptoWriter<W: Write, D: Digest + FixedOutputReset> {
    hasher: D,
    writer: W,
}

fn draw_field<F: Field, D: Digest + FixedOutputReset>(hasher: D) -> F {
    // TODO: slice 2x bytes more than the order to remove the bias
    // TODO: or use rejection sampling for unbiased field elements
    let bytes = hasher.finalize().to_vec();
    assert!(bytes.len() >= 8);
    let value: u64 = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
    F::from_u64(value)
}

fn draw_ext_field<F: Field, Ext: ExtensionField<F>, D: Digest + FixedOutputReset + Clone>(
    hasher: &mut D,
) -> Ext {
    Ext::from_basis_coefficients_iter((0..Ext::DIMENSION).map(|_| {
        Digest::update(hasher, [RUST_CRYPTO_UPDATE]);
        draw_field::<F, _>(hasher.clone())
    }))
    .unwrap()
}

impl<W: Write + Default, D: Digest + FixedOutputReset + Clone> RustCryptoWriter<W, D> {
    pub fn init(prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoWriter {
            hasher: D::new_with_prefix(prefix),
            writer: W::default(),
        }
    }

    pub fn finalize(self) -> W {
        self.writer
    }
}

impl<W: Write, D: Digest + FixedOutputReset, F: Field> FieldWriter<F> for RustCryptoWriter<W, D> {
    fn write_hint(&mut self, e: F) -> Result<(), crate::Error> {
        let bytes = bincode::serialize(&e).unwrap();
        self.writer
            .write_all(&bytes)
            .map_err(|_| crate::Error::Transcript)?;
        Ok(())
    }

    fn write(&mut self, e: F) -> Result<(), crate::Error> {
        let bytes = bincode::serialize(&e).map_err(|_| crate::Error::Transcript)?;
        self.writer
            .write_all(&bytes)
            .map_err(|_| crate::Error::Transcript)?;
        Digest::update(&mut self.hasher, bytes);
        Ok(())
    }
}

impl<
        W: Write + Default,
        D: Digest + FixedOutputReset + Clone,
        F: Field,
        Ext: ExtensionField<F>,
    > Challenge<F, Ext> for RustCryptoWriter<W, D>
{
    fn draw(&mut self) -> Ext {
        draw_ext_field(&mut self.hasher)
    }
}

#[derive(Debug, Clone)]
pub struct RustCryptoReader<R: Read, D: Digest + FixedOutputReset> {
    hasher: D,
    reader: R,
}

impl<R: Read, D: Digest + FixedOutputReset + Clone> RustCryptoReader<R, D> {
    pub fn init(reader: R, prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoReader {
            hasher: D::new_with_prefix(prefix),
            reader,
        }
    }
}

impl<W: Read, D: Digest + FixedOutputReset, F: Field> FieldReader<F> for RustCryptoReader<W, D> {
    fn read_hint(&mut self) -> Result<F, crate::Error> {
        let mut bytes = vec![0u8; F::NUM_BYTES];
        self.reader
            .read_exact(bytes.as_mut())
            .map_err(|_| crate::Error::Transcript)?;
        bincode::deserialize(&bytes).map_err(|_| crate::Error::Transcript)
    }

    fn read(&mut self) -> Result<F, crate::Error> {
        let mut bytes = vec![0u8; F::NUM_BYTES];
        self.reader
            .read_exact(bytes.as_mut())
            .map_err(|_| crate::Error::Transcript)?;
        Digest::update(&mut self.hasher, &bytes);
        bincode::deserialize(&bytes).map_err(|_| crate::Error::Transcript)
    }
}

impl<R: Read, D: Digest + FixedOutputReset + Clone, F: Field, Ext: ExtensionField<F>>
    Challenge<F, Ext> for RustCryptoReader<R, D>
{
    fn draw(&mut self) -> Ext {
        draw_ext_field(&mut self.hasher)
    }
}

#[cfg(test)]
mod test {
    use crate::transcript::FieldReader;
    use crate::transcript::{Challenge, FieldWriter};
    #[cfg(test)]
    use digest::{Digest, FixedOutputReset};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    #[cfg(test)]
    use p3_field::{ExtensionField, Field};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use rand::Rng;

    #[cfg(test)]
    fn run_test<F: Field, Ext: ExtensionField<F>, D: Digest + FixedOutputReset + Clone>(seed: u64)
    where
        rand::distr::StandardUniform: rand::distr::Distribution<F>,
        rand::distr::StandardUniform: rand::distr::Distribution<Ext>,
    {
        use crate::transcript::rust_crypto::{RustCryptoReader, RustCryptoWriter};

        let mut w = RustCryptoWriter::<Vec<u8>, D>::init("");
        let mut rng = crate::test::rng(seed + 1);
        let mut rng_w = crate::test::rng(seed);

        let mut els_w = vec![];
        let mut els_ext_w = vec![];
        let mut chs_w = vec![];
        (0..10).for_each(|_| {
            let n_draw = rng_w.random_range(0..10);
            let n_write = rng_w.random_range(0..10);

            let chs: Vec<Ext> = w.draw_many(n_draw);
            let els: Vec<F> = (0..n_write).map(|_| rng.random()).collect();
            let els_ext: Vec<Ext> = (0..n_write).map(|_| rng.random()).collect();
            w.write_many(&els).unwrap();
            w.write_many(&els_ext).unwrap();

            els_w.extend(els);
            els_ext_w.extend(els_ext);
            chs_w.extend(chs);
        });

        let data = w.finalize();
        let mut r = RustCryptoReader::<&[u8], D>::init(&data, "");

        let mut rng_r = crate::test::rng(seed);
        let mut els_r = vec![];
        let mut els_ext_r = vec![];
        let mut chs_r = vec![];
        (0..10).for_each(|_| {
            let n_draw = rng_r.random_range(0..10);
            let n_read = rng_r.random_range(0..10);

            let chs: Vec<Ext> = r.draw_many(n_draw);
            let els: Vec<F> = r.read_many(n_read).unwrap();
            let els_ext: Vec<Ext> = r.read_many(n_read).unwrap();

            els_r.extend(els);
            els_ext_r.extend(els_ext);
            chs_r.extend(chs);
        });

        let must_be_err: Result<F, crate::Error> = r.read();
        assert!(must_be_err.is_err());

        assert_eq!(els_w, els_r);
        assert_eq!(els_ext_w, els_ext_r);
        assert_eq!(chs_w, chs_r);
    }

    #[test]
    fn test_transcript() {
        for i in 0..100 {
            run_test::<Goldilocks, Goldilocks, sha2::Sha256>(i);
            run_test::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, sha2::Sha256>(i);
            run_test::<BabyBear, BabyBear, sha2::Sha256>(i);
            run_test::<KoalaBear, KoalaBear, sha2::Sha256>(i);
            run_test::<KoalaBear, BinomialExtensionField<KoalaBear, 4>, sha2::Sha256>(i);
        }
    }
}
