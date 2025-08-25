use crate::field::SerializedField;
use crate::transcript::BytesReader;
use crate::transcript::BytesWriter;
use crate::transcript::Challenge;
use crate::transcript::ChallengeBits;
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
    challenger: D,
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

fn draw_bits<D: Digest + FixedOutputReset + Clone>(hasher: &mut D, bits: usize) -> usize {
    Digest::update(hasher, [RUST_CRYPTO_UPDATE]);
    let bytes = hasher.clone().finalize().to_vec();
    let ret = usize::from_le_bytes(bytes[0..usize::BITS as usize / 8].try_into().unwrap());
    ret & ((1 << bits) - 1)
}

impl<W: Write + Default, D: Digest + FixedOutputReset + Clone> RustCryptoWriter<W, D> {
    pub fn init(prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoWriter {
            challenger: D::new_with_prefix(prefix),
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
        Digest::update(&mut self.challenger, bytes);
        Ok(())
    }

    fn write_hint(&mut self, bytes: &[u8]) -> Result<(), crate::Error> {
        self.writer
            .write_all(bytes)
            .map_err(|_| crate::Error::Transcript)
    }
}

impl<W: Write, D: Digest + FixedOutputReset + Clone, F: Field, Ext: ExtensionField<F>>
    Challenge<F, Ext> for RustCryptoWriter<W, D>
{
    fn draw(&mut self) -> Ext {
        draw_ext_field::<F, Ext, _>(&mut self.challenger)
    }
}

impl<W: Write, D: Digest + FixedOutputReset + Clone> ChallengeBits for RustCryptoWriter<W, D> {
    fn draw(&mut self, bits: usize) -> usize {
        draw_bits(&mut self.challenger, bits)
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

impl<R: Read + Default, D: Digest + FixedOutputReset + Clone> ChallengeBits
    for RustCryptoReader<R, D>
{
    fn draw(&mut self, bits: usize) -> usize {
        draw_bits(&mut self.hasher, bits)
    }
}
