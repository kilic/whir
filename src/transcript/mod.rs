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
