use crate::utils::{log2_strict, n_rand, unsafe_allocate_zero_vec, TwoAdicSlice};
use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use rand::distr::{Distribution, StandardUniform};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

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

impl<F: Field> From<Vec<F>> for Point<F> {
    fn from(v: Vec<F>) -> Self {
        Self(v)
    }
}

impl<'a, F: Field> From<&'a [F]> for Point<F>
where
    F: Clone,
{
    fn from(v: &'a [F]) -> Self {
        Self(v.to_vec())
    }
}

impl<F: Field> Point<F> {
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

    pub fn eval_lagrange<BaseField: Field>(&self, poly: &[BaseField]) -> F
    where
        F: ExtensionField<BaseField>,
    {
        let constant = (poly.len() == 1).then_some(*poly.first().unwrap());
        if let Some(constant) = constant {
            return constant.into();
        }

        let (z0, z1) = self.split_at(self.len() / 2);
        let left = z0.eq(F::ONE);
        let right = z1.eq(F::ONE);

        assert_eq!(poly.k(), left.k() + right.k());
        poly.par_chunks(left.len())
            .zip_eq(right.par_iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<F>()
                    * c
            })
            .sum()
    }

    pub fn eval_coeffs<BaseField: Field>(&self, poly: &[BaseField]) -> F
    where
        F: ExtensionField<BaseField>,
    {
        let constant = (poly.len() == 1).then_some(*poly.first().unwrap());
        if let Some(constant) = constant {
            return constant.into();
        }

        let k = poly.k();
        assert_eq!(k, self.len());

        let mut bss = unsafe_allocate_zero_vec(1 << (k - 1));
        bss[0] = F::ONE;
        let mut sum = (*poly.first().unwrap()).into();
        for (i, &zi) in self.iter().enumerate() {
            sum += bss
                .par_iter()
                .zip(poly.par_iter().skip(1 << (i)))
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

pub fn compressed_eq<F: Field>(points: &[Point<F>], alpha: F, shift: F) -> Poly<F> {
    assert!(!points.is_empty());
    let k = points.first().unwrap().len();
    points.iter().for_each(|point| assert_eq!(point.len(), k));
    let n = points.len();

    let mut eq = crate::utils::unsafe_allocate_zero_vec((1 << k) * n);
    // prepare RLC coeffs
    let alphas = alpha
        .powers()
        .take(n)
        .map(|alpha| alpha * shift)
        .collect::<Vec<_>>();
    // initialize first row with randomness factors
    eq.iter_mut()
        .zip(alphas.iter())
        .for_each(|(e, alpha)| *e = *alpha);
    // build eqs independently
    for i in 0..k {
        let (lo, hi) = eq.split_at_mut((1 << i) * n);
        lo.par_chunks_mut(n)
            .zip(hi.par_chunks_mut(n))
            .for_each(|(lo, hi)| {
                points
                    .iter()
                    .zip(lo.iter_mut())
                    .zip(hi.iter_mut())
                    .for_each(|((point, a0), a1)| {
                        let zi = point[i];
                        *a1 = *a0 * zi;
                        *a0 -= *a1;
                    });
            });
    }

    // sum each row
    eq.par_chunks(n)
        .map(|row| row.iter().fold(F::ZERO, |acc, &v| acc + v))
        .collect::<Vec<_>>()
        .into()
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

    pub fn eval_univariate<Ext: ExtensionField<F>>(&self, point: Ext) -> Ext {
        crate::utils::par_horner(&self.values, point)
    }

    pub fn eval_coeffs<Ext: ExtensionField<F>>(&self, point: &Point<Ext>) -> Ext {
        point.eval_coeffs(self)
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn to_coeffs(mut self) -> Poly<F> {
        self.par_chunks_mut(2)
            .for_each(|chunk| chunk[1] -= chunk[0]);

        for i in 2..=self.k() {
            let chunk_size = 1 << i;
            self.par_chunks_mut(chunk_size).for_each(|chunk| {
                let mid = chunk_size >> 1;
                (mid..chunk_size).for_each(|j| chunk[j] -= chunk[j - mid]);
            });
        }

        self
    }

    pub fn to_lagrange(mut self) -> Poly<F> {
        self.par_chunks_mut(2)
            .for_each(|chunk| chunk[1] += chunk[0]);

        for i in 2..=self.k() {
            let chunk_size = 1 << i;
            self.par_chunks_mut(chunk_size).for_each(|chunk| {
                let mid = chunk_size >> 1;
                (mid..chunk_size).for_each(|j| chunk[j] += chunk[j - mid]);
            });
        }
        self
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
        poly::{compressed_eq, Point, Poly},
        utils::{unsafe_allocate_zero_vec, VecOps},
    };
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;

    type F = BabyBear;
    type Ext = BinomialExtensionField<F, 4>;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::Rng;
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    };

    #[test]
    fn test_compressed_eq() {
        let k = 18;
        let n = 12;
        let mut rng = crate::test::rng(1);
        let alpha: F = rng.random();
        let shift: F = rng.random();
        // crate::test::init_tracing();

        let points = (0..n)
            .map(|_| Point::<F>::rand(&mut rng, k))
            .collect::<Vec<_>>();

        let eq0: Poly<_> = tracing::info_span!("eq0").in_scope(|| {
            let eqs = points
                .iter()
                .map(|point| point.eq(F::ONE))
                .collect::<Vec<_>>();
            (0..1 << k)
                .map(|i| eqs.iter().map(|eq| &eq[i]).horner_shifted(alpha, shift))
                .collect::<Vec<_>>()
                .into()
        });

        let eq1 = tracing::info_span!("eq1").in_scope(|| compressed_eq(&points, alpha, shift));
        assert_eq!(eq0, eq1);

        let alphas = alpha
            .powers()
            .take(n)
            .map(|alpha| alpha * shift)
            .collect::<Vec<_>>();
        let mut eq2: Vec<F> = unsafe_allocate_zero_vec(1 << k);
        tracing::info_span!("eq2").in_scope(|| {
            let eqs = points
                .iter()
                .map(|point| point.eq(F::ONE))
                .collect::<Vec<_>>();
            for (eq, alpha) in eqs.iter().zip(alphas.iter()) {
                eq2.par_iter_mut()
                    .zip(eq.par_iter())
                    .for_each(|(e, &v)| *e += v * *alpha);
            }
        });
        assert_eq!(eq0, eq2.into());
    }

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
            let e1 = poly0
                .iter()
                .zip(point.eq(Ext::ONE).iter())
                .map(|(&coeff, &eq)| eq * coeff)
                .sum::<Ext>();
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_domain() {
        type F = p3_goldilocks::Goldilocks;
        use p3_field::TwoAdicField;

        let mut rng = &mut crate::test::rng(1);
        let k = 4usize;

        let poly: Poly<F> = Poly::rand(&mut rng, k);
        let poly_in_coeffs: Poly<F> = poly.clone().to_coeffs();
        let extend = 3;
        let rate = poly.k() + extend;

        let size = 1 << rate;
        let mut padded: Vec<F> = crate::utils::unsafe_allocate_zero_vec(size);
        padded[..poly.len()].copy_from_slice(&poly);

        let mat = RowMajorMatrix::new(padded.clone(), 1);

        let cw = Radix2DitParallel::default().dft_algebra_batch(mat);
        let cw = &cw.values;

        let omega = F::two_adic_generator(rate);

        cw.iter().enumerate().for_each(|(i, &cw_i)| {
            let wi = omega.exp_u64(i as u64);
            let ei = poly.eval_univariate(wi);
            assert_eq!(ei, cw_i);

            let point = Point::<F>::expand(poly.k(), wi);
            let u0 = poly.eval_lagrange(&point);
            let u1 = poly_in_coeffs.eval_univariate::<F>(wi);
            assert_eq!(u0, u1);
        });
    }

    #[test]
    fn test_univariate() {
        type F = p3_goldilocks::Goldilocks;

        let mut rng = &mut crate::test::rng(1);
        let k = 4usize;

        for _ in 0..1000 {
            let poly_in_lagrange: Poly<F> = Poly::rand(&mut rng, k);
            let poly_in_coeffs = poly_in_lagrange.clone().to_coeffs();
            let r: F = rng.random();
            let point = Point::<F>::expand(poly_in_coeffs.k(), r);
            let e0 = poly_in_lagrange.eval_lagrange(&point);
            let e1 = poly_in_coeffs.eval_coeffs(&point);
            assert_eq!(e0, e1);
            let e1 = poly_in_coeffs.eval_univariate(r);
            assert_eq!(e0, e1);
        }

        for _ in 0..1000 {
            let poly_in_coeffs: Poly<F> = Poly::rand(&mut rng, k);
            let poly_in_lagrange = poly_in_coeffs.clone().to_lagrange();
            let r: F = rng.random();
            let point = Point::<F>::expand(poly_in_lagrange.k(), r);
            let e0 = poly_in_coeffs.eval_coeffs(&point);
            let e1 = poly_in_lagrange.eval_lagrange(&point);
            assert_eq!(e0, e1);
            let e1 = poly_in_coeffs.eval_univariate(r);
            assert_eq!(e0, e1);
        }
    }
}
