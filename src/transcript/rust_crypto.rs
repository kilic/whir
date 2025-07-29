use crate::field::SerializedField;
use crate::transcript::BytesReader;
use crate::transcript::BytesWriter;
use crate::transcript::Challenge;
use crate::transcript::Reader;
use crate::transcript::Writer;
use digest::Digest;
use digest::FixedOutputReset;
use p3_field::ExtensionField;
use p3_field::Field;
use std::io::Read;
use std::io::Write;

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

impl<W: Write, D: Digest + FixedOutputReset, F: SerializedField> Writer<F>
    for RustCryptoWriter<W, D>
{
    fn write(&mut self, e: F) -> Result<(), crate::Error> {
        e.to_bytes()
            .map_err(|_| crate::Error::Transcript)
            .and_then(|bytes| <Self as BytesWriter>::write(self, &bytes))
    }

    fn write_hint(&mut self, e: F) -> Result<(), crate::Error> {
        e.to_bytes()
            .map_err(|_| crate::Error::Transcript)
            .and_then(|bytes| <Self as BytesWriter>::write_hint(self, &bytes))
    }
}

impl<W: Write, D: Digest + FixedOutputReset> Writer<[u8; 32]> for RustCryptoWriter<W, D> {
    fn write(&mut self, e: [u8; 32]) -> Result<(), crate::Error> {
        <Self as BytesWriter>::write(self, &e)
    }

    fn write_hint(&mut self, e: [u8; 32]) -> Result<(), crate::Error> {
        <Self as BytesWriter>::write_hint(self, &e)
    }
}

impl<W: Write, D: Digest + FixedOutputReset> BytesWriter for RustCryptoWriter<W, D> {
    fn write(&mut self, bytes: &[u8]) -> Result<(), crate::Error> {
        <Self as BytesWriter>::write_hint(self, bytes)?;
        Digest::update(&mut self.hasher, bytes);
        Ok(())
    }

    fn write_hint(&mut self, bytes: &[u8]) -> Result<(), crate::Error> {
        self.writer
            .write_all(bytes)
            .map_err(|_| crate::Error::Transcript)
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
        draw_ext_field::<F, Ext, _>(&mut self.hasher)
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

impl<W: Read, D: Digest + FixedOutputReset, F: SerializedField> Reader<F>
    for RustCryptoReader<W, D>
{
    fn read(&mut self) -> Result<F, crate::Error> {
        <Self as BytesReader>::read(self, F::NUM_BYTES)
            .and_then(|bytes| F::from_bytes(&bytes).map_err(|_| crate::Error::Transcript))
    }

    fn read_hint(&mut self) -> Result<F, crate::Error> {
        <Self as BytesReader>::read_hint(self, F::NUM_BYTES)
            .and_then(|bytes| F::from_bytes(&bytes).map_err(|_| crate::Error::Transcript))
    }
}

impl<W: Read, D: Digest + FixedOutputReset> Reader<[u8; 32]> for RustCryptoReader<W, D> {
    fn read(&mut self) -> Result<[u8; 32], crate::Error> {
        Ok(<Self as BytesReader>::read(self, 32)?.try_into().unwrap())
    }

    fn read_hint(&mut self) -> Result<[u8; 32], crate::Error> {
        Ok(<Self as BytesReader>::read_hint(self, 32)?
            .try_into()
            .unwrap())
    }
}

impl<R: Read, D: Digest + FixedOutputReset> BytesReader for RustCryptoReader<R, D> {
    fn read(&mut self, n: usize) -> Result<Vec<u8>, crate::Error> {
        let bytes = <Self as BytesReader>::read_hint(self, n)?;
        Digest::update(&mut self.hasher, &bytes);
        Ok(bytes)
    }

    fn read_hint(&mut self, n: usize) -> Result<Vec<u8>, crate::Error> {
        let mut bytes = vec![0u8; n];
        self.reader
            .read_exact(bytes.as_mut())
            .map_err(|_| crate::Error::Transcript)?;
        Ok(bytes)
    }
}

impl<R: Read, D: Digest + FixedOutputReset + Clone, F: Field, Ext: ExtensionField<F>>
    Challenge<F, Ext> for RustCryptoReader<R, D>
{
    fn draw(&mut self) -> Ext {
        draw_ext_field::<F, Ext, _>(&mut self.hasher)
    }
}

#[cfg(test)]
mod test {
    #[cfg(test)]
    use crate::field::SerializedField;
    use crate::transcript::{Challenge, Reader, Writer};
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
    fn run_test<
        F: Field + SerializedField,
        Ext: ExtensionField<F> + SerializedField,
        D: Digest + FixedOutputReset + Clone,
    >(
        seed: u64,
    ) where
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
        let mut bytes_w = vec![];
        (0..10).for_each(|_| {
            use crate::transcript::BytesWriter;

            let n_draw = rng_w.random_range(0..10);
            let n_write = rng_w.random_range(0..10);
            let n_write_ext = rng_w.random_range(0..10);
            let n_bytes = rng_w.random_range(0..128);

            let chs: Vec<Ext> = w.draw_many(n_draw);
            let els: Vec<F> = (0..n_write).map(|_| rng.random()).collect();
            let els_ext: Vec<Ext> = (0..n_write_ext).map(|_| rng.random()).collect();
            let bytes: Vec<u8> = (0..n_bytes).map(|_| rng.random()).collect();

            w.write_many(&els).unwrap();
            w.write_many(&els_ext).unwrap();
            <RustCryptoWriter<_, D> as BytesWriter>::write(&mut w, &bytes).unwrap();

            els_w.extend(els);
            els_ext_w.extend(els_ext);
            chs_w.extend(chs);
            bytes_w.extend(bytes);
        });

        let data = w.finalize();
        let mut r = RustCryptoReader::<&[u8], D>::init(&data, "");

        let mut rng_r = crate::test::rng(seed);
        let mut els_r = vec![];
        let mut els_ext_r = vec![];
        let mut chs_r = vec![];
        let mut bytes_r = vec![];
        (0..10).for_each(|_| {
            use crate::transcript::BytesReader;

            let n_draw = rng_r.random_range(0..10);
            let n_read = rng_r.random_range(0..10);
            let n_read_ext = rng_r.random_range(0..10);
            let n_bytes = rng_r.random_range(0..128);

            let chs: Vec<Ext> = r.draw_many(n_draw);
            let els: Vec<F> = r.read_many(n_read).unwrap();
            let els_ext: Vec<Ext> = r.read_many(n_read_ext).unwrap();
            let bytes: Vec<u8> =
                <RustCryptoReader<_, D> as BytesReader>::read(&mut r, n_bytes).unwrap();

            els_r.extend(els);
            els_ext_r.extend(els_ext);
            chs_r.extend(chs);
            bytes_r.extend(bytes);
        });

        let must_be_err: Result<F, crate::Error> = r.read();
        assert!(must_be_err.is_err());

        assert_eq!(els_w, els_r);
        assert_eq!(els_ext_w, els_ext_r);
        assert_eq!(chs_w, chs_r);
        assert_eq!(bytes_w, bytes_r);
    }

    #[test]
    fn test_transcript() {
        for i in 0..100 {
            run_test::<Goldilocks, Goldilocks, sha2::Sha256>(i);
            run_test::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, sha2::Sha256>(i);
            run_test::<BabyBear, BabyBear, sha2::Sha256>(i);
            run_test::<BabyBear, BinomialExtensionField<BabyBear, 4>, sha2::Sha256>(i);
            run_test::<KoalaBear, KoalaBear, sha2::Sha256>(i);
            run_test::<KoalaBear, BinomialExtensionField<KoalaBear, 4>, sha2::Sha256>(i);
        }
    }
}
