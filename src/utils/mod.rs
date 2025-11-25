pub mod arithmetic;
use crate::p3_field_prelude::*;
pub use arithmetic::*;
use p3_util::log2_strict_usize;
use rayon::prelude::*;

pub fn n_rand<F>(rng: impl rand::RngCore, n: usize) -> Vec<F>
where
    rand::distr::StandardUniform: rand::distr::Distribution<F>,
{
    use rand::Rng;
    rng.random_iter().take(n).collect::<Vec<F>>()
}

pub trait TwoAdicSlice<T>: core::ops::Deref<Target = [T]> {
    #[inline(always)]
    fn k(&self) -> usize {
        log2_strict_usize(self.len())
    }
}

impl<V> TwoAdicSlice<V> for Vec<V> {}
impl<V> TwoAdicSlice<V> for &[V] {}
impl<V> TwoAdicSlice<V> for &mut [V] {}

#[inline]
pub fn unpack_into<F: Field, Ext: ExtensionField<F>>(
    out: &mut [Ext],
    packed: &[Ext::ExtensionPacking],
) {
    let width = F::Packing::WIDTH;
    assert_eq!(out.len(), packed.len() * width);
    packed
        .par_iter()
        .zip(out.par_chunks_mut(width))
        .with_min_len(1 << 14)
        .for_each(|(packed, out_chunk)| {
            let packed_coeffs = packed.as_basis_coefficients_slice();
            for (i, out) in out_chunk.iter_mut().enumerate().take(width) {
                *out = Ext::from_basis_coefficients_fn(|j| packed_coeffs[j].as_slice()[i]);
            }
        });
}

#[inline]
pub fn unpack<F: Field, Ext: ExtensionField<F>>(packed: &[Ext::ExtensionPacking]) -> Vec<Ext> {
    let mut out = Ext::zero_vec(packed.len() * F::Packing::WIDTH);
    unpack_into(&mut out, packed);
    out
}

#[inline]
pub fn pack<F: Field, Ext: ExtensionField<F>>(packed: &[Ext]) -> Vec<Ext::ExtensionPacking> {
    packed
        .par_chunks(F::Packing::WIDTH)
        .with_min_len(1 << 14)
        .map(|ext| Ext::ExtensionPacking::from_ext_slice(ext))
        .collect()
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
