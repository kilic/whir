use p3_field::{ExtensionField, Field};

pub mod poseidon;
pub mod rust_crypto;

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
    fn write(&mut self, el: T) -> Result<(), crate::Error>;
    fn write_hint(&mut self, el: T) -> Result<(), crate::Error>;
    fn write_many(&mut self, el: &[T]) -> Result<(), crate::Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write(e))
    }
    fn write_hint_many(&mut self, el: &[T]) -> Result<(), crate::Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write_hint(e))
    }
}

pub trait Reader<T> {
    fn read(&mut self) -> Result<T, crate::Error>;
    fn read_hint(&mut self) -> Result<T, crate::Error>;
    fn read_many(&mut self, n: usize) -> Result<Vec<T>, crate::Error> {
        (0..n).map(|_| self.read()).collect::<Result<Vec<_>, _>>()
    }
    fn read_hint_many(&mut self, n: usize) -> Result<Vec<T>, crate::Error> {
        (0..n)
            .map(|_| self.read_hint())
            .collect::<Result<Vec<_>, _>>()
    }
}

pub trait BytesWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), crate::Error>;
    fn write_hint(&mut self, bytes: &[u8]) -> Result<(), crate::Error>;
}

pub trait BytesReader {
    fn read(&mut self, n: usize) -> Result<Vec<u8>, crate::Error>;
    fn read_hint(&mut self, n: usize) -> Result<Vec<u8>, crate::Error>;
}

#[cfg(test)]
mod test {
    #[cfg(test)]
    use crate::field::SerializedField;
    use crate::transcript::{
        poseidon::{PoseidonReader, PoseidonWriter},
        rust_crypto::{RustCryptoReader, RustCryptoWriter},
        Challenge, ChallengeBits, Reader, Writer,
    };
    #[cfg(test)]
    use digest::{Digest, FixedOutputReset};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{DuplexChallenger, FieldChallenger, GrindingChallenger};
    use p3_field::extension::BinomialExtensionField;
    #[cfg(test)]
    use p3_field::{ExtensionField, Field};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
    use rand::{
        distr::{Distribution, StandardUniform},
        Rng,
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
        let mut rng_range = crate::test::rng(seed);
        let mut rng_number = crate::test::rng(seed + 1);

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

    #[cfg(test)]
    fn reader<R, F: Field + SerializedField, Ext: ExtensionField<F> + SerializedField>(
        seed: u64,
        r: &mut R,
    ) -> (Vec<F>, Vec<Ext>, Vec<Ext>)
    where
        StandardUniform: Distribution<F>,
        StandardUniform: Distribution<Ext>,
        R: Reader<F> + Reader<Ext> + Challenge<F, Ext> + ChallengeBits,
    {
        let mut rng = crate::test::rng(seed);
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
    fn test_transcript_rustcrypto() {
        fn run_test<
            F: Field + SerializedField,
            Ext: ExtensionField<F> + SerializedField,
            D: Digest + FixedOutputReset + Clone,
        >()
        where
            StandardUniform: Distribution<F>,
            StandardUniform: Distribution<Ext>,
        {
            for i in 0..100 {
                let mut w = RustCryptoWriter::<Vec<u8>, D>::init("");
                let (els0, els_ext0, chs0) = writer::<_, F, Ext>(i, &mut w);
                let checkpoint0: Ext = <RustCryptoWriter<_, _> as Challenge<F, Ext>>::draw(&mut w);
                let data = w.finalize();

                let mut r = RustCryptoReader::<&[u8], D>::init(&data, "");
                let (els1, els_ext1, chs1) = reader::<_, F, Ext>(i, &mut r);
                let checkpoint1: Ext = <RustCryptoReader<_, _> as Challenge<F, Ext>>::draw(&mut r);

                assert_eq!(checkpoint0, checkpoint1);
                assert_eq!(els0, els1);
                assert_eq!(els_ext0, els_ext1);
                assert_eq!(chs0, chs1);

                let must_be_err: Result<F, crate::Error> = r.read();
                assert!(must_be_err.is_err());
            }
        }

        run_test::<Goldilocks, Goldilocks, sha2::Sha256>();
        run_test::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, sha2::Sha256>();
        run_test::<BabyBear, BabyBear, sha2::Sha256>();
        run_test::<BabyBear, BinomialExtensionField<BabyBear, 4>, sha2::Sha256>();
        run_test::<KoalaBear, KoalaBear, sha2::Sha256>();
        run_test::<KoalaBear, BinomialExtensionField<KoalaBear, 4>, sha2::Sha256>();
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
            let challenger = Challenger::new(Perm::new_from_rng_128(&mut crate::test::rng(1000)));
            run_test::<F, Ext, _>(&challenger);
        }

        {
            type F = KoalaBear;
            type Ext = BinomialExtensionField<F, 4>;
            type Perm = Poseidon2KoalaBear<16>;
            type Challenger = DuplexChallenger<F, Perm, 16, 8>;
            let challenger = Challenger::new(Perm::new_from_rng_128(&mut crate::test::rng(1000)));
            run_test::<F, Ext, _>(&challenger);
        }
    }
}
