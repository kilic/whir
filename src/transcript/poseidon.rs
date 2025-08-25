use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use std::{
    io::{Read, Write},
    marker::PhantomData,
};

use crate::{
    field::SerializedField,
    transcript::{Challenge, ChallengeBits, Reader, Writer},
};

#[derive(Debug, Clone)]
pub struct PoseidonWriter<W: Write, F: Field, Challenger> {
    challenger: Challenger,
    writer: W,
    _marker: PhantomData<F>,
}

impl<W: Write + Default, F: Field, Challenger> PoseidonWriter<W, F, Challenger> {
    pub fn init(challenger: Challenger) -> Self {
        PoseidonWriter {
            writer: W::default(),
            challenger,
            _marker: PhantomData,
        }
    }

    pub fn finalize(self) -> W {
        self.writer
    }
}

impl<W: Write, F: SerializedField, Ext: ExtensionField<F> + SerializedField, Challenger> Writer<Ext>
    for PoseidonWriter<W, F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn write(&mut self, e: Ext) -> Result<(), crate::Error> {
        self.write_hint(e)?;
        self.challenger.observe_algebra_element(e);
        Ok(())
    }

    fn write_hint(&mut self, e: Ext) -> Result<(), crate::Error> {
        let bytes = e.to_bytes().map_err(|_| crate::Error::Transcript)?;
        self.writer
            .write_all(&bytes)
            .map_err(|_| crate::Error::Transcript)
    }
}

impl<W: Write, F: SerializedField, Challenger, const D: usize> Writer<[F; D]>
    for PoseidonWriter<W, F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn write(&mut self, e: [F; D]) -> Result<(), crate::Error> {
        self.write_hint(e)?;
        self.challenger.observe_slice(&e);
        Ok(())
    }

    fn write_hint(&mut self, e: [F; D]) -> Result<(), crate::Error> {
        e.into_iter()
            .try_for_each(|e| <Self as Writer<F>>::write_hint(self, e))
    }
}

impl<W: Write, F: SerializedField, Ext: ExtensionField<F>, Challenger> Challenge<F, Ext>
    for PoseidonWriter<W, F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn draw(&mut self) -> Ext {
        self.challenger.sample_algebra_element::<Ext>()
    }
}

impl<W: Write, F: Field, Challenger> ChallengeBits for PoseidonWriter<W, F, Challenger>
where
    Challenger: GrindingChallenger,
{
    fn draw(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }
}

#[derive(Debug, Clone)]
pub struct PoseidonReader<R: Read, F: Field, Challenger> {
    challenger: Challenger,
    reader: R,
    _marker: PhantomData<F>,
}

impl<R: Read + Default, F: Field, Challenger> PoseidonReader<R, F, Challenger> {
    pub fn init(reader: R, challenger: Challenger) -> Self {
        PoseidonReader {
            reader,
            challenger,
            _marker: PhantomData,
        }
    }
}

impl<R: Read, F: SerializedField, Ext: ExtensionField<F> + SerializedField, Challenger> Reader<Ext>
    for PoseidonReader<R, F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn read(&mut self) -> Result<Ext, crate::Error> {
        let e: Ext = self.read_hint()?;
        self.challenger.observe_algebra_element(e);
        Ok(e)
    }

    fn read_hint(&mut self) -> Result<Ext, crate::Error> {
        let mut bytes = vec![0u8; Ext::NUM_BYTES];
        self.reader
            .read_exact(bytes.as_mut())
            .map_err(|_| crate::Error::Transcript)?;
        Ext::from_bytes(&bytes).map_err(|_| crate::Error::Transcript)
    }
}

impl<R: Read, F: SerializedField, Challenger, const D: usize> Reader<[F; D]>
    for PoseidonReader<R, F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn read(&mut self) -> Result<[F; D], crate::Error> {
        let result: [F; D] = self.read_hint()?;
        self.challenger.observe_slice(&result);
        Ok(result)
    }

    fn read_hint(&mut self) -> Result<[F; D], crate::Error> {
        Ok((0..D)
            .map(|_| self.read_hint())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| crate::Error::Transcript)?
            .try_into()
            .unwrap())
    }
}

impl<R: Read, F: SerializedField, Ext: ExtensionField<F>, Challenger> Challenge<F, Ext>
    for PoseidonReader<R, F, Challenger>
where
    Challenger: FieldChallenger<F>,
{
    fn draw(&mut self) -> Ext {
        self.challenger.sample_algebra_element::<Ext>()
    }
}

impl<R: Read, F: Field, Challenger> ChallengeBits for PoseidonReader<R, F, Challenger>
where
    Challenger: GrindingChallenger,
{
    fn draw(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }
}
