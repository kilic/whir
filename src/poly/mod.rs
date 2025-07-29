use crate::utils::{log2_strict, n_rand, unsafe_allocate_zero_vec, TwoAdicSlice};
use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Point<F>(pub Vec<F>);

impl<F> std::ops::Deref for Point<F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> std::ops::DerefMut for Point<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F> From<Vec<F>> for Point<F> {
    fn from(v: Vec<F>) -> Self {
        Self(v)
    }
}

impl<'a, F> From<&'a [F]> for Point<F>
where
    F: Clone,
{
    fn from(v: &'a [F]) -> Self {
        Self(v.to_vec())
    }
}

impl<F: Clone> Point<F> {
    pub fn new(zs: &[F]) -> Self {
        zs.into()
    }

    pub fn rand<R: rand::Rng>(rng: &mut R, k: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self(n_rand(rng, k))
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.0.split_at(mid);
        (left.into(), right.into())
    }
}

impl<F: Field> Point<F> {
    pub fn expand(k: usize, mut z: F) -> Self {
        let mut point = (0..k)
            .map(|_| {
                let ret = z;
                z = z.square();
                ret
            })
            .collect::<Vec<_>>();
        point.reverse();
        point.into()
    }

    pub fn hypercube(value: usize, k: usize) -> Self {
        (0..k)
            .map(|i| {
                if (value >> i) & 1 == 1 {
                    F::ONE
                } else {
                    F::ZERO
                }
            })
            .collect::<Vec<_>>()
            .into()
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }

    pub fn reversed(&self) -> Point<F> {
        self.0.iter().rev().cloned().collect::<Vec<_>>().into()
    }

    pub fn as_ext<Ext: ExtensionField<F>>(&self) -> Point<Ext> {
        self.iter().map(|&z| z.into()).collect_vec().into()
    }

    pub fn eval_lagrange<BaseField: Field>(&self, poly: &Poly<BaseField>) -> F
    where
        F: ExtensionField<BaseField>,
    {
        let k = poly.k();
        assert_eq!(k, self.len());

        if let Some(constant) = poly.constant() {
            return constant.into();
        }

        let mid = 1 << (k - 1);
        let (lo, hi) = poly.split_at(mid);
        let mut poly: Vec<F> = unsafe_allocate_zero_vec(mid);
        let z0 = *self.last().unwrap();
        lo.par_iter()
            .zip_eq(hi.par_iter())
            .zip_eq(poly.par_iter_mut())
            .for_each(|((&a0, &a1), t)| *t = z0 * (a1 - a0) + a0);

        for &zi in self.iter().rev().skip(1) {
            let mid = poly.len() / 2;
            let (lo, hi) = poly.split_at_mut(mid);
            lo.par_iter_mut()
                .zip(hi.par_iter())
                .for_each(|(a0, a1)| *a0 += zi * (*a1 - *a0));
            poly.truncate(mid);
        }
        assert_eq!(poly.k(), 0);
        poly[0]
    }

    pub fn eval_coeffs<BaseField: Field>(&self, poly: &Poly<BaseField>) -> F
    where
        F: ExtensionField<BaseField>,
    {
        let k = poly.k();
        assert_eq!(k, self.len());

        if let Some(constant) = poly.constant() {
            return constant.into();
        }

        let mut bss = unsafe_allocate_zero_vec(1 << (k - 1));
        bss[0] = F::ONE;
        let mut sum = (*poly.values.first().unwrap()).into();
        for (i, &zi) in self.iter().enumerate() {
            sum += bss
                .par_iter()
                .zip(poly.values.par_iter().skip(1 << (i)))
                .take(1 << (i))
                .map(|(&u0, &u1)| u0 * u1)
                .sum::<F>()
                * zi;
            let (eqlo, eqhi) = bss.split_at_mut(1 << i);
            eqlo.par_iter_mut()
                .zip(eqhi.par_iter_mut())
                .for_each(|(lo, hi)| *hi = *lo * zi);
        }
        sum
    }

    pub fn eq(&self, scale: F) -> Poly<F> {
        assert!(!self.is_empty());
        assert_ne!(scale, F::ZERO);
        let k = self.len();
        let mut eq = unsafe_allocate_zero_vec(1 << k);
        eq[0] = scale;
        for (i, &zi) in self.iter().enumerate() {
            let (lo, hi) = eq.split_at_mut(1 << i);
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .for_each(|(a0, a1)| {
                    *a1 = *a0 * zi;
                    *a0 -= *a1;
                });
        }
        eq.into()
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct Poly<F> {
    pub values: Vec<F>,
}

impl<F> std::ops::Deref for Poly<F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<F> std::ops::DerefMut for Poly<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<F: Field> From<Vec<F>> for Poly<F> {
    fn from(values: Vec<F>) -> Self {
        Self::new(values)
    }
}

impl<F: Field> From<&[F]> for Poly<F> {
    fn from(values: &[F]) -> Self {
        Self::new(values.to_vec())
    }
}

impl<F: Field> From<&Vec<F>> for Poly<F> {
    fn from(values: &Vec<F>) -> Self {
        Self::new(values.clone())
    }
}

impl<F: Field> Poly<F> {
    pub fn new(values: Vec<F>) -> Self {
        assert!(!values.is_empty());
        assert!(values.len().is_power_of_two());
        Self { values }
    }

    pub fn rand(rng: &mut impl rand::Rng, k: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self::new(n_rand(rng, 1 << k))
    }

    pub fn zero(k: usize) -> Self {
        Self {
            values: unsafe_allocate_zero_vec(1 << k),
        }
    }

    pub fn k(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn constant(&self) -> Option<F> {
        (self.len() == 1).then_some(*self.first().unwrap())
    }

    pub fn eval_lagrange<Ext: ExtensionField<F>>(&self, point: &Point<Ext>) -> Ext {
        point.eval_lagrange(self)
    }

    pub fn eval_coeffs<Ext: ExtensionField<F>>(&self, point: &Point<Ext>) -> Ext {
        point.eval_coeffs(self)
    }

    pub fn evals(&self) -> &[F] {
        &self.values
    }

    pub fn from_evals(evals: &[F]) -> Self {
        assert!(!evals.is_empty());
        Self {
            values: evals.to_vec(),
        }
    }

    pub fn to_coeffs(mut self) -> Poly<F> {
        self.values
            .par_chunks_mut(2)
            .for_each(|chunk| chunk[1] -= chunk[0]);

        for i in 2..=self.k() {
            let chunk_size = 1 << i;
            self.values.par_chunks_mut(chunk_size).for_each(|chunk| {
                let mid = chunk_size >> 1;
                (mid..chunk_size).for_each(|j| chunk[j] -= chunk[j - mid]);
            });
        }

        self.values.into()
    }

    pub fn to_lagrange(mut self) -> Poly<F> {
        self.values
            .par_chunks_mut(2)
            .for_each(|chunk| chunk[1] += chunk[0]);

        for i in 2..=self.k() {
            let chunk_size = 1 << i;
            self.values.par_chunks_mut(chunk_size).for_each(|chunk| {
                let mid = chunk_size >> 1;
                (mid..chunk_size).for_each(|j| chunk[j] += chunk[j - mid]);
            });
        }
        self.values.into()
    }

    pub fn fix_var(&mut self, zi: F) {
        let mid = self.len() / 2;
        let (p0, p1) = self.split_at_mut(mid);
        p0.par_iter_mut()
            .zip(p1.par_iter())
            .for_each(|(a0, a1)| *a0 += zi * (*a1 - *a0));
        self.truncate(mid);
    }

    pub fn fix_var_ext<Ext: ExtensionField<F>>(&self, zi: Ext) -> Poly<Ext> {
        let mid = self.len() / 2;
        let (p0, p1) = self.split_at(mid);
        p0.par_iter()
            .zip(p1.par_iter())
            .map(|(&a0, &a1)| zi * (a1 - a0) + a0)
            .collect::<Vec<_>>()
            .into()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        poly::{Point, Poly},
        utils::VecOps,
    };
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    type F = Goldilocks;
    type Ext = BinomialExtensionField<F, 2>;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_eval() {
        let mut rng = crate::test::rng(1);
        let k = 5;
        let poly0 = Poly::<F>::rand(&mut rng, k);
        let poly1 = poly0.clone().to_coeffs();
        let poly2 = poly1.clone().to_lagrange();
        assert_eq!(poly0, poly2);
        for i in 0..(1 << k) {
            let e0 = poly0[i];

            let point = Point::<F>::hypercube(i, k);
            let e1 = point.eval_lagrange(&poly0);
            assert_eq!(e0, e1);
            let e1 = point.eval_coeffs(&poly1);
            assert_eq!(e0, e1);
            let mut poly2 = poly0.clone();
            point.iter().rev().for_each(|&var| poly2.fix_var(var));
            let e1 = poly2.constant().unwrap();
            assert_eq!(e0, e1);
        }

        for _ in 0..1000 {
            let point = Point::<Ext>::rand(&mut rng, k);
            let e0 = point.eval_lagrange(&poly0);
            let e1 = point.eval_coeffs(&poly1);
            assert_eq!(e0, e1);

            let mut poly2 = poly0.fix_var_ext(*point.last().unwrap());
            point
                .iter()
                .rev()
                .skip(1)
                .for_each(|&var| poly2.fix_var(var));

            let e1 = poly2.constant().unwrap();
            assert_eq!(e0, e1);
            let e1 = poly0.dot(&point.eq(Ext::ONE));
            assert_eq!(e0, e1);
        }
    }
}
