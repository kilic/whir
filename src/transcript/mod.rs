use p3_field::{ExtensionField, Field};

pub mod rust_crypto;

pub trait ChallengeBits {
    fn draw(&mut self, bit_size: usize) -> usize;
    fn draw_many(&mut self, n: usize, bit_size: usize) -> Vec<usize> {
        (0..n).map(|_| self.draw(bit_size)).collect()
    }
}

pub trait Challenge<F: Field, Ext: ExtensionField<F>> {
    fn draw(&mut self) -> Ext;
    fn draw_many(&mut self, n: usize) -> Vec<Ext> {
        (0..n).map(|_| self.draw()).collect()
    }
}

pub trait FieldWriter<F: Field> {
    fn write(&mut self, el: F) -> Result<(), crate::Error>;
    fn write_hint(&mut self, el: F) -> Result<(), crate::Error>;
    fn write_many(&mut self, el: &[F]) -> Result<(), crate::Error>
    where
        F: Copy,
    {
        el.iter().try_for_each(|&e| self.write(e))
    }
    fn write_hint_many(&mut self, el: &[F]) -> Result<(), crate::Error>
    where
        F: Copy,
    {
        el.iter().try_for_each(|&e| self.write_hint(e))
    }
}

pub trait FieldReader<F: Field> {
    fn read(&mut self) -> Result<F, crate::Error>;
    fn read_hint(&mut self) -> Result<F, crate::Error>;
    fn read_many(&mut self, n: usize) -> Result<Vec<F>, crate::Error> {
        (0..n).map(|_| self.read()).collect::<Result<Vec<_>, _>>()
    }
    fn read_hint_many(&mut self, n: usize) -> Result<Vec<F>, crate::Error> {
        (0..n)
            .map(|_| self.read_hint())
            .collect::<Result<Vec<_>, _>>()
    }
}
