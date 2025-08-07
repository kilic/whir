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

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use rand::Rng;

    #[test]
    fn test_par_horner() {
        type F = Goldilocks;
        type Ext = BinomialExtensionField<F, 2>;

        let rng = &mut crate::test::rng(1);
        let k_max = 16;

        let poly: Vec<F> = (0..(1 << k_max)).map(|_| rng.random()).collect();
        for k in 0..15 {
            let poly = &poly[..(1 << k)];
            let x = rng.random::<Ext>();
            let e0 = poly.iter().horner(x);
            let e1 = par_horner(poly, x);
            assert_eq!(e0, e1);
        }
    }
}
