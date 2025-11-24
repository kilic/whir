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

pub(crate) fn extrapolate<F: Field, EF: ExtensionField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len()).map(F::from_usize).collect::<Vec<_>>();
    interpolate(&points, evals).iter().horner(target)
}

pub(crate) fn interpolate<F: Field, Ext: ExtensionField<F>>(
    points: &[F],
    evals: &[Ext],
) -> Vec<Ext> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        points.iter().enumerate().for_each(|(j, x_j)| {
            let mut denom = Vec::with_capacity(points.len() - 1);
            points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .for_each(|x_k| denom.push(*x_j - *x_k));
            denoms.push(denom);
        });
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).inverse();

        let mut final_poly = vec![Ext::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
                .for_each(|(x_k, denom)| {
                    product.resize(tmp.len() + 1, F::ZERO);
                    tmp.iter()
                        .chain(core::iter::once(&F::ZERO))
                        .zip(core::iter::once(&F::ZERO).chain(tmp.iter()))
                        .zip(product.iter_mut())
                        .for_each(|((a, b), product)| *product = *a * (-denom * *x_k) + *b * denom);
                    core::mem::swap(&mut tmp, &mut product);
                });
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            final_poly.iter_mut().zip(tmp.into_iter()).for_each(
                |(final_coeff, interpolation_coeff)| *final_coeff += *eval * interpolation_coeff,
            );
        }
        final_poly
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
