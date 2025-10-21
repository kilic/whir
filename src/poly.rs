use crate::utils::{log2_strict, n_rand, unsafe_allocate_zero_vec, TwoAdicSlice};
use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_matrix::{dense::DenseMatrix, util::reverse_matrix_index_bits};
use rand::distr::{Distribution, StandardUniform};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
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

    pub(super) fn eval_poly_in_evals<BaseField: Field>(&self, poly: &[BaseField]) -> F
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

    pub(super) fn eval_poly_in_coeffs<BaseField: Field>(&self, poly: &[BaseField]) -> F
    where
        F: ExtensionField<BaseField>,
    {
        let constant = (poly.len() == 1).then_some(*poly.first().unwrap());
        if let Some(constant) = constant {
            return constant.into();
        }

        let k = poly.k();
        assert_eq!(k, self.len());

        let mut buf = unsafe_allocate_zero_vec(1 << (k - 1));
        buf[0] = F::ONE;
        let mut sum = (*poly.first().unwrap()).into();
        for (i, &zi) in self.iter().enumerate() {
            sum += buf
                .par_iter()
                .zip(poly.par_iter().skip(1 << i))
                .take(1 << i)
                .map(|(&u0, &u1)| u0 * u1)
                .sum::<F>()
                * zi;
            let (lo, hi) = buf.split_at_mut(1 << i);
            lo.par_iter_mut()
                .zip(hi.par_iter_mut())
                .for_each(|(lo, hi)| *hi = *lo * zi);
        }
        sum
    }

    pub fn eq(&self, scale: F) -> Poly<F, Eval> {
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

pub trait Basis: Default {}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct Coeff {}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct Eval {}

impl Basis for Coeff {}
impl Basis for Eval {}

#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct Poly<F, Basis> {
    pub values: Vec<F>,
    pub basis: Basis,
}

impl<F, Basis> std::ops::Deref for Poly<F, Basis> {
    type Target = Vec<F>;
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<F, Basis> std::ops::DerefMut for Poly<F, Basis> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<F: Field, B: Basis, I: IntoIterator<Item = F>> From<I> for Poly<F, B> {
    fn from(values: I) -> Self {
        Self::new(values.into_iter().collect::<Vec<_>>())
    }
}

impl<F: Field, B: Basis> Poly<F, B> {
    pub fn new(values: Vec<F>) -> Self {
        assert!(!values.is_empty());
        assert!(values.len().is_power_of_two());
        Self {
            values,
            basis: B::default(),
        }
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
            basis: B::default(),
        }
    }

    pub fn k(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn constant(&self) -> Option<F> {
        (self.len() == 1).then_some(*self.first().unwrap())
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn reverse_index_bits(self) -> Self {
        let mut mat = DenseMatrix::new(self.values, 1);
        reverse_matrix_index_bits(&mut mat);
        mat.values.into()
    }
}

impl<F: Field> Poly<F, Coeff> {
    pub fn to_evals(mut self) -> Poly<F, Eval> {
        self.par_chunks_mut(2)
            .for_each(|chunk| chunk[1] += chunk[0]);

        for i in 2..=self.k() {
            let chunk_size = 1 << i;
            self.par_chunks_mut(chunk_size).for_each(|chunk| {
                let mid = chunk_size >> 1;
                (mid..chunk_size).for_each(|j| chunk[j] += chunk[j - mid]);
            });
        }

        Poly {
            values: self.values,
            basis: Eval {},
        }
    }

    pub fn eval<Ext: ExtensionField<F>>(&self, point: &Point<Ext>) -> Ext {
        point.eval_poly_in_coeffs(self)
    }

    pub fn eval_univariate<Ext: ExtensionField<F>>(&self, point: Ext) -> Ext {
        crate::utils::par_horner(&self.values, point)
    }

    pub fn fix_var_mut(&mut self, zi: F) {
        let mid = self.len() / 2;
        let (p0, p1) = self.split_at_mut(mid);
        p0.par_iter_mut()
            .zip(p1.par_iter())
            .for_each(|(a0, &a1)| *a0 += zi * a1);
        self.truncate(mid);
    }

    pub fn fix_var<Ext: ExtensionField<F>>(&self, zi: Ext) -> Poly<Ext, Coeff> {
        let (p0, p1) = self.split_at(self.len() / 2);
        p0.par_iter()
            .zip(p1.par_iter())
            .map(|(&a0, &a1)| zi * a1 + a0)
            .collect::<Vec<_>>()
            .into()
    }
}

impl<F: Field> Poly<F, Eval> {
    pub fn to_coeffs(mut self) -> Poly<F, Coeff> {
        self.par_chunks_mut(2)
            .for_each(|chunk| chunk[1] -= chunk[0]);

        for i in 2..=self.k() {
            let chunk_size = 1 << i;
            self.par_chunks_mut(chunk_size).for_each(|chunk| {
                let mid = chunk_size >> 1;
                (mid..chunk_size).for_each(|j| chunk[j] -= chunk[j - mid]);
            });
        }

        Poly {
            values: self.values,
            basis: Coeff {},
        }
    }

    pub fn eval<Ext: ExtensionField<F>>(&self, point: &Point<Ext>) -> Ext {
        point.eval_poly_in_evals(self)
    }

    pub fn eval_univariate<Ext: ExtensionField<F>>(&self, point: Ext) -> Ext {
        crate::utils::par_horner(&self.values, point)
    }

    pub fn fix_var_mut(&mut self, zi: F) {
        let mid = self.len() / 2;
        let (p0, p1) = self.split_at_mut(mid);
        p0.par_iter_mut()
            .zip(p1.par_iter())
            .for_each(|(a0, &a1)| *a0 += zi * (a1 - *a0));
        self.truncate(mid);
    }

    pub fn fix_var<Ext: ExtensionField<F>>(&self, zi: Ext) -> Poly<Ext, Eval> {
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

    use crate::poly::{Eval, Point, Poly};
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::Rng;

    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;

    #[test]
    #[ignore]
    fn test_bench_eval() {
        let mut rng = crate::test::rng(1);
        let k = 25;
        let poly0 = Poly::<F, Eval>::rand(&mut rng, k);
        let poly1 = poly0.clone().to_coeffs();

        let var: F = rng.random();
        let point = Point::<F>::expand(k, var);

        crate::test::init_tracing();
        let e0 = tracing::info_span!("e0").in_scope(|| poly0.eval(&point));
        let e1 = tracing::info_span!("e1").in_scope(|| poly1.eval(&point));
        assert_eq!(e0, e1);
        let e1 = tracing::info_span!("e1").in_scope(|| poly1.eval_univariate(var));
        assert_eq!(e0, e1);
    }

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
            let poly_in_evals = Poly::<F, Eval>::rand(&mut rng, k);
            let poly_in_coeffs = poly_in_evals.clone().to_coeffs();
            assert_eq!(poly_in_evals, poly_in_coeffs.clone().to_evals());

            for i in 0..(1 << k) {
                let e0 = poly_in_evals[i];
                let point = Point::<F>::hypercube(i, k);
                assert_eq!(e0, poly_in_coeffs.eval(&point));
                assert_eq!(e0, poly_in_evals.eval(&point));

                {
                    let mut poly = poly_in_evals.clone();
                    point.iter().rev().for_each(|&var| poly.fix_var_mut(var));
                    let e1 = poly.constant().unwrap();
                    assert_eq!(e0, e1);
                }

                {
                    let mut poly = poly_in_coeffs.clone();
                    point.iter().rev().for_each(|&var| poly.fix_var_mut(var));
                    let e1 = poly.constant().unwrap();
                    assert_eq!(e0, e1);
                }
            }

            let point = Point::<Ext>::rand(&mut rng, k);
            let e0 = poly_in_coeffs.eval(&point);
            assert_eq!(e0, poly_in_evals.eval(&point));

            {
                let mut poly = poly_in_evals.fix_var(*point.last().unwrap());
                point
                    .iter()
                    .rev()
                    .skip(1)
                    .for_each(|&var| poly.fix_var_mut(var));

                let e1 = poly.constant().unwrap();
                assert_eq!(e0, e1);
            }

            {
                let mut poly = poly_in_coeffs.fix_var(*point.last().unwrap());
                point
                    .iter()
                    .rev()
                    .skip(1)
                    .for_each(|&var| poly.fix_var_mut(var));

                let e1 = poly.constant().unwrap();
                assert_eq!(e0, e1);
            }

            {
                let e1 = poly_in_evals
                    .iter()
                    .zip(point.eq(Ext::ONE).iter())
                    .map(|(&coeff, &eq)| eq * coeff)
                    .sum::<Ext>();
                assert_eq!(e0, e1);
            }
        }
    }

    #[test]
    fn test_domain() {
        use p3_field::TwoAdicField;

        let mut rng = &mut crate::test::rng(1);
        let k = 4usize;

        let poly: Poly<F, Eval> = Poly::rand(&mut rng, k);
        let poly_in_coeffs = poly.clone().to_coeffs();
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
            let u0 = poly.eval(&point);
            let u1 = poly_in_coeffs.eval_univariate::<F>(wi);
            assert_eq!(u0, u1);
        });
    }
}
