pub mod field;
pub mod poseidon;
pub use common::Error;

use p3_field::{ExtensionField, Field};

pub trait ChallengeBits {
    fn draw(&mut self, bits: usize) -> usize;
    fn draw_many(&mut self, n: usize, bits: usize) -> Vec<usize> {
        (0..n).map(|_| self.draw(bits)).collect()
    }
}

pub trait Challenge<F: Field, Ext: ExtensionField<F>> {
    fn draw(&mut self) -> Ext;
    fn draw_many(&mut self, n: usize) -> Vec<Ext> {
        (0..n).map(|_| self.draw()).collect()
    }
}

pub trait Writer<T> {
    fn write(&mut self, el: T) -> Result<(), Error>;
    fn write_hint(&mut self, el: T) -> Result<(), Error>;
    fn write_many(&mut self, el: &[T]) -> Result<(), Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write(e))
    }
    fn write_hint_many(&mut self, el: &[T]) -> Result<(), Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write_hint(e))
    }
}

pub trait Reader<T> {
    fn read(&mut self) -> Result<T, Error>;
    fn read_hint(&mut self) -> Result<T, Error>;
    fn read_many(&mut self, n: usize) -> Result<Vec<T>, Error> {
        (0..n).map(|_| self.read()).collect::<Result<Vec<_>, _>>()
    }
    fn read_hint_many(&mut self, n: usize) -> Result<Vec<T>, Error> {
        (0..n)
            .map(|_| self.read_hint())
            .collect::<Result<Vec<_>, _>>()
    }
}

pub trait BytesWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error>;
    fn write_hint(&mut self, bytes: &[u8]) -> Result<(), Error>;
}

pub trait BytesReader {
    fn read(&mut self, n: usize) -> Result<Vec<u8>, Error>;
    fn read_hint(&mut self, n: usize) -> Result<Vec<u8>, Error>;
}

// TODO: under test feature
pub mod test_transcript {
    use crate::field::SerializedField;
    use rand::{
        RngExt, SeedableRng,
        distr::{Distribution, StandardUniform},
        rngs::SmallRng,
    };
    use std::{
        io::{Read, Write},
        marker::PhantomData,
    };

    use super::*;

    #[derive(Debug, Clone)]
    pub struct TestWriter<W: Write, F: Field> {
        challenger: SmallRng,
        writer: W,
        _marker: PhantomData<F>,
    }

    impl<W: Write + Default, F: Field> TestWriter<W, F> {
        pub fn init() -> Self {
            TestWriter {
                writer: W::default(),
                challenger: SmallRng::seed_from_u64(5000),
                _marker: PhantomData,
            }
        }

        pub fn finalize(self) -> W {
            self.writer
        }
    }

    impl<W: Write, F: SerializedField, Ext: ExtensionField<F> + SerializedField> Writer<Ext>
        for TestWriter<W, F>
    {
        fn write(&mut self, e: Ext) -> Result<(), Error> {
            <Self as Writer<Ext>>::write_hint(self, e)
        }

        fn write_hint(&mut self, e: Ext) -> Result<(), Error> {
            let bytes = e.to_bytes().map_err(|_| Error::Transcript)?;
            self.writer.write_all(&bytes).map_err(|_| Error::Transcript)
        }
    }

    impl<W: Write, F: SerializedField, const D: usize> Writer<[F; D]> for TestWriter<W, F> {
        fn write(&mut self, e: [F; D]) -> Result<(), Error> {
            <Self as Writer<F>>::write_many(self, &e)
        }

        fn write_hint(&mut self, e: [F; D]) -> Result<(), Error> {
            e.into_iter()
                .try_for_each(|e| <Self as Writer<F>>::write_hint(self, e))
        }
    }

    impl<W: Write, F: Field> BytesWriter for TestWriter<W, F> {
        fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
            <Self as BytesWriter>::write_hint(self, bytes)?;
            Ok(())
        }

        fn write_hint(&mut self, bytes: &[u8]) -> Result<(), Error> {
            self.writer.write_all(bytes).map_err(|_| Error::Transcript)
        }
    }

    impl<W: Write, F: Field> Writer<[u8; 32]> for TestWriter<W, F> {
        fn write(&mut self, e: [u8; 32]) -> Result<(), Error> {
            <Self as BytesWriter>::write(self, &e)
        }

        fn write_hint(&mut self, e: [u8; 32]) -> Result<(), Error> {
            <Self as BytesWriter>::write_hint(self, &e)
        }
    }

    impl<W: Write, F: SerializedField, Ext: ExtensionField<F>> Challenge<F, Ext> for TestWriter<W, F>
    where
        StandardUniform: Distribution<Ext>,
    {
        fn draw(&mut self) -> Ext {
            self.challenger.random()
        }
    }

    impl<W: Write, F: Field> ChallengeBits for TestWriter<W, F> {
        fn draw(&mut self, bits: usize) -> usize {
            self.challenger.random_range(0..1 << bits)
        }
    }

    #[derive(Debug, Clone)]
    pub struct TestReader<R: Read, F: Field> {
        challenger: SmallRng,
        reader: R,
        _marker: PhantomData<F>,
    }

    impl<R: Read + Default, F: Field> TestReader<R, F> {
        pub fn init(reader: R) -> Self {
            TestReader {
                reader,
                challenger: SmallRng::seed_from_u64(5000),
                _marker: PhantomData,
            }
        }
    }

    impl<R: Read, F: SerializedField, Ext: ExtensionField<F> + SerializedField> Reader<Ext>
        for TestReader<R, F>
    {
        fn read(&mut self) -> Result<Ext, Error> {
            <Self as Reader<Ext>>::read_hint(self)
        }

        fn read_hint(&mut self) -> Result<Ext, Error> {
            let mut bytes = vec![0u8; Ext::NUM_BYTES];
            self.reader
                .read_exact(bytes.as_mut())
                .map_err(|_| Error::Transcript)?;
            Ext::from_bytes(&bytes).map_err(|_| Error::Transcript)
        }
    }

    impl<R: Read, F: SerializedField, const D: usize> Reader<[F; D]> for TestReader<R, F> {
        fn read(&mut self) -> Result<[F; D], Error> {
            <Self as Reader<[F; D]>>::read_hint(self)
        }

        fn read_hint(&mut self) -> Result<[F; D], Error> {
            Ok((0..D)
                .map(|_| <Self as Reader<F>>::read_hint(self))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Error::Transcript)?
                .try_into()
                .unwrap())
        }
    }

    impl<W: Read, F: Field> Reader<[u8; 32]> for TestReader<W, F> {
        fn read(&mut self) -> Result<[u8; 32], Error> {
            Ok(<Self as BytesReader>::read(self, 32)?.try_into().unwrap())
        }

        fn read_hint(&mut self) -> Result<[u8; 32], Error> {
            Ok(<Self as BytesReader>::read_hint(self, 32)?
                .try_into()
                .unwrap())
        }
    }

    impl<R: Read, F: Field> BytesReader for TestReader<R, F> {
        fn read(&mut self, n: usize) -> Result<Vec<u8>, Error> {
            let bytes = <Self as BytesReader>::read_hint(self, n)?;
            Ok(bytes)
        }

        fn read_hint(&mut self, n: usize) -> Result<Vec<u8>, Error> {
            let mut bytes = vec![0u8; n];
            self.reader
                .read_exact(bytes.as_mut())
                .map_err(|_| Error::Transcript)?;
            Ok(bytes)
        }
    }

    impl<R: Read, F: SerializedField, Ext: ExtensionField<F>> Challenge<F, Ext> for TestReader<R, F>
    where
        StandardUniform: Distribution<Ext>,
    {
        fn draw(&mut self) -> Ext {
            self.challenger.random()
        }
    }

    impl<R: Read, F: Field> ChallengeBits for TestReader<R, F> {
        fn draw(&mut self, bits: usize) -> usize {
            self.challenger.random_range(0..1 << bits)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::field::SerializedField;
    use crate::{
        Challenge, ChallengeBits, Reader, Writer,
        poseidon::{PoseidonReader, PoseidonWriter},
        test_transcript::{TestReader, TestWriter},
    };
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
    use rand::{
        RngExt,
        distr::{Distribution, StandardUniform},
    };

    fn writer<W, F: Field + SerializedField, Ext: ExtensionField<F> + SerializedField>(
        seed: u64,
        w: &mut W,
    ) -> (Vec<F>, Vec<Ext>, Vec<Ext>)
    where
        StandardUniform: Distribution<F>,
        StandardUniform: Distribution<Ext>,
        W: Writer<F> + Writer<Ext> + Challenge<F, Ext> + ChallengeBits,
    {
        let mut rng_range = common::test::rng(seed);
        let mut rng_number = common::test::rng(seed + 1);

        let mut els_w = vec![];
        let mut els_ext_w = vec![];
        let mut chs_w = vec![];

        (0..10).for_each(|_| {
            let n_draw = rng_range.random_range(0..10);
            let n_write = rng_range.random_range(0..10);
            let n_write_ext = rng_range.random_range(0..10);

            let chs: Vec<Ext> = <W as Challenge<F, Ext>>::draw_many(w, n_draw);
            let els: Vec<F> = (0..n_write).map(|_| rng_number.random()).collect();
            let els_ext: Vec<Ext> = (0..n_write_ext).map(|_| rng_number.random()).collect();

            w.write_many(&els).unwrap();
            w.write_many(&els_ext).unwrap();

            els_w.extend(els);
            els_ext_w.extend(els_ext);
            chs_w.extend(chs);
        });
        (els_w, els_ext_w, chs_w)
    }

    fn reader<R, F: Field + SerializedField, Ext: ExtensionField<F> + SerializedField>(
        seed: u64,
        r: &mut R,
    ) -> (Vec<F>, Vec<Ext>, Vec<Ext>)
    where
        StandardUniform: Distribution<F>,
        StandardUniform: Distribution<Ext>,
        R: Reader<F> + Reader<Ext> + Challenge<F, Ext> + ChallengeBits,
    {
        let mut rng = common::test::rng(seed);
        let mut els_r = vec![];
        let mut els_ext_r = vec![];
        let mut chs_r = vec![];

        (0..10).for_each(|_| {
            let n_draw = rng.random_range(0..10);
            let n_read = rng.random_range(0..10);
            let n_read_ext = rng.random_range(0..10);

            let chs: Vec<Ext> = <R as Challenge<F, Ext>>::draw_many(r, n_draw);
            let els: Vec<F> = r.read_many(n_read).unwrap();
            let els_ext: Vec<Ext> = r.read_many(n_read_ext).unwrap();

            els_r.extend(els);
            els_ext_r.extend(els_ext);
            chs_r.extend(chs);
        });

        (els_r, els_ext_r, chs_r)
    }

    #[test]
    fn test_test_transcript() {
        type F = Goldilocks;
        type Ext = BinomialExtensionField<F, 2>;
        for i in 0..100 {
            let mut w = TestWriter::<Vec<u8>, F>::init();
            let (els0, els_ext0, chs0) = writer::<_, F, Ext>(i, &mut w);
            let checkpoint0: Ext = <TestWriter<_, F> as Challenge<F, Ext>>::draw(&mut w);
            let data = w.finalize();

            let mut r = TestReader::<&[u8], F>::init(&data);
            let (els1, els_ext1, chs1) = reader::<_, F, Ext>(i, &mut r);
            let checkpoint1: Ext = <TestReader<_, F> as Challenge<F, Ext>>::draw(&mut r);

            assert_eq!(checkpoint0, checkpoint1);
            assert_eq!(els0, els1);
            assert_eq!(els_ext0, els_ext1);
            assert_eq!(chs0, chs1);

            let must_be_err: Result<F, crate::Error> = r.read();
            assert!(must_be_err.is_err());
        }
    }

    #[test]
    fn test_transcript_poseidon() {
        fn run_test<
            F: Field + SerializedField,
            Ext: ExtensionField<F> + SerializedField,
            Challenger: FieldChallenger<F> + GrindingChallenger,
        >(
            challenger: &Challenger,
        ) where
            StandardUniform: Distribution<F>,
            StandardUniform: Distribution<Ext>,
        {
            for i in 0..100 {
                let mut w = PoseidonWriter::<Vec<u8>, F, _>::init(challenger.clone());
                let (els0, els_ext0, chs0) = writer::<_, F, Ext>(i, &mut w);
                let checkpoint0: Ext = <PoseidonWriter<_, _, _> as Challenge<F, Ext>>::draw(&mut w);
                let data = w.finalize();

                let mut r = PoseidonReader::<&[u8], F, _>::init(&data, challenger.clone());
                let (els1, els_ext1, chs1) = reader::<_, F, Ext>(i, &mut r);
                let checkpoint1: Ext = <PoseidonReader<_, _, _> as Challenge<F, Ext>>::draw(&mut r);

                assert_eq!(checkpoint0, checkpoint1);
                assert_eq!(els0, els1);
                assert_eq!(els_ext0, els_ext1);
                assert_eq!(chs0, chs1);

                let must_be_err: Result<F, crate::Error> = r.read();
                assert!(must_be_err.is_err());
            }
        }

        {
            type F = BabyBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2BabyBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger = Challenger::new(Perm::new_from_rng_128(&mut common::test::rng(1000)));
            run_test::<F, Ext, _>(&challenger);
        }

        {
            type F = KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2KoalaBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger = Challenger::new(Perm::new_from_rng_128(&mut common::test::rng(1000)));
            run_test::<F, Ext, _>(&challenger);
        }
    }
}
