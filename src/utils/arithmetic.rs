use p3_field::{ExtensionField, Field};

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
