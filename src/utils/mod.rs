use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub mod arithmetic;
pub use arithmetic::*;

pub fn n_rand<F>(rng: impl rand::RngCore, n: usize) -> Vec<F>
where
    rand::distr::StandardUniform: rand::distr::Distribution<F>,
{
    use rand::Rng;
    rng.random_iter().take(n).collect::<Vec<F>>()
}

#[inline(always)]
pub fn log2_strict(n: usize) -> usize {
    let res = n.trailing_zeros();
    debug_assert_eq!(n.wrapping_shr(res), 1);
    res as usize
}

pub trait TwoAdicSlice<T>: core::ops::Deref<Target = [T]> {
    #[inline(always)]
    fn k(&self) -> usize {
        log2_strict(self.len())
    }
}

impl<V> TwoAdicSlice<V> for Vec<V> {}
impl<V> TwoAdicSlice<V> for &[V] {}
impl<V> TwoAdicSlice<V> for &mut [V] {}

// copy from a16z/jolt
pub(crate) fn unsafe_allocate_zero_vec<F: Default + Sized>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    unsafe {
        let value = &F::default();
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocaiton failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

impl<V: Field> VecOps<V> for Vec<V> {}
impl<V: Field> VecOps<V> for &[V] {}
impl<V: Field> VecOps<V> for &mut [V] {}

pub trait VecOps<F: Field>: core::ops::Deref<Target = [F]> {
    fn hadamard<E: ExtensionField<F>>(&self, other: &[E]) -> Vec<E> {
        self.iter()
            .zip_eq(other.iter())
            .map(|(&a, &b)| b * a)
            .collect()
    }

    fn dot<E: ExtensionField<F>>(&self, other: &[E]) -> E {
        assert_eq!(self.len(), other.len());
        self.iter().zip_eq(other.iter()).map(|(&a, &b)| b * a).sum()
    }

    fn par_hadamard<E: ExtensionField<F>>(&self, other: &[E]) -> Vec<E> {
        self.par_iter()
            .zip_eq(other.par_iter())
            .map(|(&a, &b)| b * a)
            .collect()
    }

    fn par_dot<E: ExtensionField<F>>(&self, other: &[E]) -> E {
        assert_eq!(self.len(), other.len());
        self.par_iter()
            .zip_eq(other.par_iter())
            .map(|(&a, &b)| b * a)
            .sum()
    }

    fn horner<E: ExtensionField<F>>(&self, x: E) -> E {
        self.iter().fold(E::ZERO, |acc, &coeff| acc * x + coeff)
    }

    fn rhorner<E: ExtensionField<F>>(&self, x: E) -> E {
        self.iter().rfold(E::ZERO, |acc, &coeff| acc * x + coeff)
    }
}
