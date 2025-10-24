use crate::utils::{log2_strict, n_rand, unsafe_allocate_zero_vec};
use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_matrix::{dense::DenseMatrix, util::reverse_matrix_index_bits};
use rand::distr::{Distribution, StandardUniform};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSlice,
};

pub(crate) fn eval_eq_xy<F: Field, Ext: ExtensionField<F>>(x: &Point<F>, y: &Point<Ext>) -> Ext {
    assert_eq!(x.len(), y.len());
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| (yi * xi).double() - xi - yi + F::ONE)
        .product()
}

pub(crate) fn eval_pow_xy<F: Field, Ext: ExtensionField<F>>(x: &Point<F>, y: &Point<Ext>) -> Ext {
    assert_eq!(x.len(), y.len());
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| (yi * xi) - yi + F::ONE)
        .product()
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Point<F: Field>(pub Vec<F>);

impl<F: Field> std::ops::Deref for Point<F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Field> std::ops::DerefMut for Point<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Field, I: IntoIterator<Item = F>> From<I> for Point<F> {
    fn from(values: I) -> Self {
        Self::new(values.into_iter().collect::<Vec<_>>())
    }
}

impl<F: Field> Point<F> {
    pub fn new(zs: Vec<F>) -> Self {
        Self(zs)
    }

    pub fn rand<R: rand::Rng>(rng: &mut R, k: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self(n_rand(rng, k))
    }

    pub fn vars(&self) -> &[F] {
        &self.0
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.0.split_at(mid);
        (left.to_vec().into(), right.to_vec().into())
    }

    pub fn expand(k: usize, mut var: F) -> Self {
        (0..k)
            .map(|_| {
                let ret = var;
                var = var.square();
                ret
            })
            .collect::<Vec<_>>()
            .into()
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

    pub fn eq(&self, scale: F) -> Poly<F> {
        if self.is_empty() {
            return vec![F::ONE].into();
        }
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Poly<F>(Vec<F>);

impl<F> std::ops::Deref for Poly<F> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> std::ops::DerefMut for Poly<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Field, I: IntoIterator<Item = F>> From<I> for Poly<F> {
    fn from(values: I) -> Self {
        Self::new(values.into_iter().collect::<Vec<_>>())
    }
}

impl<F: Field> Poly<F> {
    pub fn new(values: Vec<F>) -> Self {
        assert!(!values.is_empty());
        assert!(values.len().is_power_of_two());
        Self(values)
    }

    pub fn rand(rng: &mut impl rand::Rng, k: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self::new(n_rand(rng, 1 << k))
    }

    pub fn zero(k: usize) -> Self {
        Self(unsafe_allocate_zero_vec(1 << k))
    }

    pub fn k(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn constant(&self) -> Option<F> {
        (self.len() == 1).then_some(*self.first().unwrap())
    }

    pub fn reverse_index_bits(self) -> Self {
        let mut mat = DenseMatrix::new(self.0, 1);
        reverse_matrix_index_bits(&mut mat);
        mat.values.into()
    }
}

impl<F: Field> Poly<F> {
    pub fn eval<Ext: ExtensionField<F>>(&self, point: &Point<Ext>) -> Ext {
        let constant = (self.len() == 1).then_some(*self.first().unwrap());
        if let Some(constant) = constant {
            return constant.into();
        }

        let (z0, z1) = point.split_at(point.len() / 2);
        let left = z0.eq(Ext::ONE);
        let right = z1.eq(Ext::ONE);

        assert_eq!(self.k(), left.k() + right.k());
        self.par_chunks(left.len())
            .zip_eq(right.par_iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<Ext>()
                    * c
            })
            .sum()
    }

    pub fn eval_univariate<Ext: ExtensionField<F>>(&self, point: Ext) -> Ext {
        crate::utils::par_horner(self, point)
    }

    pub fn fix_var_mut(&mut self, zi: F) {
        let mid = self.len() / 2;
        let (p0, p1) = self.split_at_mut(mid);
        p0.par_iter_mut()
            .zip(p1.par_iter())
            .for_each(|(a0, &a1)| *a0 += zi * (a1 - *a0));
        self.truncate(mid);
    }

    pub fn fix_var<Ext: ExtensionField<F>>(&self, zi: Ext) -> Poly<Ext> {
        let (p0, p1) = self.split_at(self.len() / 2);
        p0.par_iter()
            .zip(p1.par_iter())
            .map(|(&a0, &a1)| zi * (a1 - a0) + a0)
            .collect::<Vec<_>>()
            .into()
    }
}

#[cfg(test)]
mod test {
    use crate::poly::{Point, Poly};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::PrimeCharacteristicRing;
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;

    #[test]
    fn test_eq() {
        let mut rng = &mut crate::test::rng(1);
        for k in 1..10 {
            let x = Point::<F>::rand(&mut rng, k);
            let y = Point::<Ext>::rand(&mut rng, k);
            let e0 = super::eval_eq_xy(&x, &y);
            let eq = x.eq(F::ONE);
            let e1 = eq.eval(&y);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_eval() {
        let mut rng = crate::test::rng(1);

        for k in 1..10 {
            let poly = Poly::<F>::rand(&mut rng, k);

            for i in 0..(1 << k) {
                let e0 = poly[i];
                let point = Point::<F>::hypercube(i, k);
                assert_eq!(e0, poly.eval(&point));

                {
                    let mut poly = poly.clone();
                    point.iter().rev().for_each(|&var| poly.fix_var_mut(var));
                    let e1 = poly.constant().unwrap();
                    assert_eq!(e0, e1);
                }
            }

            let point = Point::<Ext>::rand(&mut rng, k);
            let e0 = poly.eval(&point);

            {
                let mut poly = poly.fix_var(*point.last().unwrap());
                point
                    .iter()
                    .rev()
                    .skip(1)
                    .for_each(|&var| poly.fix_var_mut(var));

                let e1 = poly.constant().unwrap();
                assert_eq!(e0, e1);
            }

            {
                let e1 = poly
                    .iter()
                    .zip(point.eq(Ext::ONE).iter())
                    .map(|(&coeff, &eq)| eq * coeff)
                    .sum::<Ext>();
                assert_eq!(e0, e1);
            }
        }
    }
}
