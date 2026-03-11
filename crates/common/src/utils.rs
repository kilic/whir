use crate::field::*;
use p3_util::log2_strict_usize;
use rand::RngExt;
use rayon::prelude::*;

pub fn rand_vec<F>(rng: &mut impl rand::Rng, n: usize) -> Vec<F>
where
    rand::distr::StandardUniform: rand::distr::Distribution<F>,
{
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

pub trait BatchInverse<F> {
    fn inverse(self) -> F;
}

impl<'a, F, I> BatchInverse<F> for I
where
    F: Field,
    I: IntoIterator<Item = &'a mut F>,
{
    fn inverse(self) -> F {
        let mut acc = F::ONE;
        let inter = self
            .into_iter()
            .map(|p| {
                let prev = acc;
                if !p.is_zero() {
                    acc *= *p
                }
                (prev, p)
            })
            .collect::<Vec<_>>();
        acc = acc.inverse();
        let prod = acc;
        for (mut tmp, p) in inter.into_iter().rev() {
            tmp *= acc;
            if !p.is_zero() {
                acc *= *p;
                *p = tmp;
            }
        }
        prod
    }
}

pub fn par_horner<F: Field, E: ExtensionField<F>>(poly: &[F], point: E) -> E {
    // taken from pse/halo2
    let n = poly.len();
    let num_threads = rayon::current_num_threads();
    if n * 2 < num_threads {
        poly.iter()
            .rfold(E::ZERO, |acc, &coeff| acc * point + coeff)
    } else {
        let chunk_size = n.div_ceil(num_threads);
        let mut parts = vec![E::ZERO; num_threads];
        rayon::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    out[0] =
                        poly.iter().horner(point) * point.exp_u64((chunk_idx * chunk_size) as u64);
                });
            }
        });
        parts.iter().fold(E::ZERO, |acc, coeff| acc + *coeff)
    }
}

pub trait VecOps<F: Field> {
    fn horner<E: ExtensionField<F>>(self, x: E) -> E;
    fn horner_shifted<E: ExtensionField<F>>(self, x: E, shift: E) -> E;
}

impl<I, F> VecOps<F> for I
where
    F: Field,
    I: DoubleEndedIterator,
    I::Item: std::ops::Deref<Target = F>,
{
    fn horner<E: ExtensionField<F>>(self, x: E) -> E {
        self.rfold(E::ZERO, |acc, coeff| acc * x + *coeff)
    }

    fn horner_shifted<E: ExtensionField<F>>(self, x: E, shift: E) -> E {
        self.rfold(E::ZERO, |acc, coeff| acc * x + *coeff) * shift
    }
}
