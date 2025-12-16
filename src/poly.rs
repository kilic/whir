use std::ops::RangeBounds;
use std::slice::SliceIndex;

use crate::p3_field_prelude::*;
use crate::utils::{n_rand, unpack};
use itertools::Itertools;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{dense::DenseMatrix, util::reverse_matrix_index_bits};
use p3_util::log2_strict_usize;
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
        .zip_eq(y.par_iter())
        .map(|(&xi, &yi)| (yi * xi).double() - xi - yi + F::ONE)
        .product()
}

pub(crate) fn eval_pow_xy<F: Field, Ext: ExtensionField<F>>(x: &Point<F>, y: &Point<Ext>) -> Ext {
    assert_eq!(x.len(), y.len());
    x.par_iter()
        .zip_eq(y.par_iter())
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

    pub fn k(&self) -> usize {
        self.len()
    }

    pub fn vars(&self) -> &[F] {
        &self.0
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.0.split_at(mid);
        (left.to_vec().into(), right.to_vec().into())
    }

    pub fn range<R: RangeBounds<usize> + SliceIndex<[F], Output = [F]>>(&self, range: R) -> Self {
        self.0[range].to_vec().into()
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
        let mut eq = F::zero_vec(1 << k);
        eq[0] = scale;
        for (i, &zi) in self.iter().enumerate() {
            let (lo, hi) = eq.split_at_mut(1 << i);
            lo.iter_mut().zip(hi.iter_mut()).for_each(|(a0, a1)| {
                *a1 = *a0 * zi;
                *a0 -= *a1;
            });
        }
        eq.into()
    }

    pub fn eq_packed<BaseField: Field>(&self, scale: F) -> Poly<F::ExtensionPacking>
    where
        F: ExtensionField<BaseField>,
    {
        let k = self.len();
        let k_pack = log2_strict_usize(BaseField::Packing::WIDTH);
        assert!(k >= k_pack);
        assert_ne!(scale, F::ZERO);

        let mut acc_init: Vec<F> = F::zero_vec(1 << k_pack);
        acc_init[0] = scale;
        for i in 0..k_pack {
            let (lo, hi) = acc_init.split_at_mut(1 << i);
            let var = self[i];
            lo.iter_mut().zip(hi.iter_mut()).for_each(|(lo, hi)| {
                *hi = *lo * var;
                *lo -= *hi;
            });
        }

        let mut acc = F::ExtensionPacking::zero_vec(1 << (k - k_pack));
        acc[0] = F::ExtensionPacking::from_ext_slice(&acc_init);

        for i in 0..k - k_pack {
            let (lo, hi) = acc.split_at_mut(1 << i);
            let var = self[i + k_pack];
            lo.iter_mut().zip(hi.iter_mut()).for_each(|(lo, hi)| {
                *hi = *lo * var;
                *lo -= *hi;
            });
        }
        acc.into()
    }

    pub fn transpose(points: &[Self]) -> RowMajorMatrix<F> {
        let k = points.iter().map(Point::k).all_equal_value().unwrap();
        let n = points.len();
        let mut flat_points = F::zero_vec(k * n);
        points.iter().enumerate().for_each(|(i, point)| {
            point
                .iter()
                .enumerate()
                .for_each(|(j, &cur)| flat_points[j * n + i] = cur);
        });
        RowMajorMatrix::new(flat_points, n)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Poly<F>(Vec<F>);

impl<F: Clone + Copy + Default + Send + Sync, I: IntoIterator<Item = F>> From<I> for Poly<F> {
    fn from(values: I) -> Self {
        Self::new(values.into_iter().collect::<Vec<_>>())
    }
}

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

impl<F: Clone + Copy + Default + Send + Sync> Poly<F> {
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

    pub fn zero(k: usize) -> Self
    where
        F: PrimeCharacteristicRing,
    {
        Self(F::zero_vec(1 << k))
    }

    pub fn k(&self) -> usize {
        log2_strict_usize(self.len())
    }

    pub fn constant(&self) -> Option<F> {
        (self.len() == 1).then_some(*self.first().unwrap())
    }

    pub fn reverse_index_bits(self) -> Self {
        let mut mat = DenseMatrix::new(self.0, 1);
        reverse_matrix_index_bits(&mut mat);
        mat.values.into()
    }

    pub fn mat(&self, width: usize) -> RowMajorMatrixView<'_, F> {
        RowMajorMatrixView::new(&self.0, width)
    }
}

impl<A: Clone + Copy + Default + Send + Sync> Poly<A> {
    pub fn eval_packed<F, Ext>(&self, point: &Point<Ext>) -> Ext
    where
        F: Field,
        Ext: ExtensionField<F, ExtensionPacking = A>,
        A: PackedFieldExtension<F, Ext>,
    {
        let k = point.len();
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        assert_eq!(self.k() + k_pack, k);
        assert!(k >= 2 * k_pack);

        let (left, right) = point.split_at(point.len() / 2);
        let left = left.eq_packed(Ext::ONE);
        let right = right.eq(Ext::ONE);

        let packed = self
            .par_chunks(left.len())
            .zip_eq(right.par_iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<A>()
                    * c
            })
            .sum::<A>();
        Ext::ExtensionPacking::to_ext_iter([packed]).sum::<Ext>()
    }

    pub fn eval_univariate_packed<F, Ext>(&self, point: Ext) -> Ext
    where
        F: Field,
        Ext: ExtensionField<F, ExtensionPacking = A>,
        A: PackedFieldExtension<F, Ext>,
    {
        let exp = point.exp_u64(F::Packing::WIDTH as u64);
        let packed_acc = self.iter().rfold(A::ZERO, |acc, &next| acc * exp + next);
        let unpacked = A::to_ext_iter([packed_acc]).collect::<Vec<_>>();
        unpacked
            .iter()
            .rfold(Ext::ZERO, |acc, &next| acc * point + next)
    }

    pub fn eval_univariate<Ext>(&self, point: Ext) -> Ext
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        crate::utils::par_horner(self, point)
    }

    pub fn eval<Ext>(&self, point: &Point<Ext>) -> Ext
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        assert_eq!(self.k(), point.k());
        let constant = (self.len() == 1).then_some(*self.first().unwrap());
        if let Some(constant) = constant {
            return constant.into();
        }

        let (left, right) = point.split_at(point.len() / 2);
        let left = left.eq(Ext::ONE);
        let right = right.eq(Ext::ONE);

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

    pub fn eval_base<Ext>(&self, point: &Point<Ext>) -> Ext
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        assert_eq!(self.k(), point.k());
        let constant = (self.len() == 1).then_some(*self.first().unwrap());
        if let Some(constant) = constant {
            return constant.into();
        }
        let poly = A::Packing::pack_slice(self);

        let (left, right) = point.split_at(point.len() / 2);
        let left = left.eq_packed(Ext::ONE);
        let right = right.eq(Ext::ONE);

        let sum = poly
            .par_chunks(left.len())
            .zip_eq(right.par_iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<Ext::ExtensionPacking>()
                    * c
            })
            .sum::<Ext::ExtensionPacking>();
        unpack(&[sum]).into_iter().sum::<Ext>()
    }

    pub fn fix_var_mut<F: Clone + Copy + Default + Send + Sync>(&mut self, zi: F)
    where
        A: Algebra<F>,
    {
        let mid = self.len() / 2;
        let (p0, p1) = self.split_at_mut(mid);
        p0.par_iter_mut()
            .zip(p1.par_iter())
            .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * zi);
        self.truncate(mid);
    }

    pub fn fix_var_packed<Ext>(&self, zi: Ext) -> Poly<Ext::ExtensionPacking>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        let zi = Ext::ExtensionPacking::from_ext_slice(&vec![zi; A::Packing::WIDTH]);
        let poly = A::Packing::pack_slice(self);

        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at(mid);
        p0.par_iter()
            .zip(p1.par_iter())
            .map(|(&a0, &a1)| zi * (a1 - a0) + a0)
            .collect::<Vec<_>>()
            .into()
    }

    pub fn fix_var<Ext>(&self, zi: Ext) -> Poly<Ext>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        let (p0, p1) = self.split_at(self.len() / 2);
        p0.par_iter()
            .zip(p1.par_iter())
            .map(|(&a0, &a1)| zi * (a1 - a0) + a0)
            .collect::<Vec<_>>()
            .into()
    }

    #[cfg(test)]
    pub(crate) fn pack<F>(&self) -> Poly<A::ExtensionPacking>
    where
        F: Field,
        A: ExtensionField<F>,
    {
        crate::utils::pack(self).into()
    }

    pub fn unpack<F, Ext>(&self) -> Poly<Ext>
    where
        F: Field,
        Ext: ExtensionField<F, ExtensionPacking = A>,
        A: PackedFieldExtension<F, Ext>,
    {
        crate::utils::unpack(self).into()
    }
}

#[cfg(test)]
mod test {
    use crate::poly::{Point, Poly};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
    type F = p3_koala_bear::KoalaBear;
    type Ext = BinomialExtensionField<F, 4>;
    use rand::Rng;

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
    fn test_univariate() {
        let mut rng = crate::test::rng(1);

        for k in 4..=10 {
            let poly = Poly::<Ext>::rand(&mut rng, k);
            let var: Ext = rng.random();
            let e0 = poly.eval_univariate(var);
            let poly_packed = poly.pack::<F>();
            let e1 = poly_packed.eval_univariate_packed::<F, Ext>(var);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_multilinear() {
        fn eval_ref<F: Field, Ext: ExtensionField<F>>(poly: &Poly<F>, point: &Point<Ext>) -> Ext {
            poly.iter()
                .zip(point.eq(Ext::ONE).iter())
                .map(|(&coeff, &eq)| eq * coeff)
                .sum::<Ext>()
        }

        let mut rng = crate::test::rng(1);

        for k in 1..=10 {
            let poly = Poly::<Ext>::rand(&mut rng, k);

            {
                let point = Point::<Ext>::rand(&mut rng, k);
                let e0 = eval_ref(&poly, &point);
                let e1 = poly.eval(&point);
                assert_eq!(e0, e1);

                if k >= 4 {
                    let poly_packed = poly.pack::<F>();
                    let e1 = poly_packed.eval_packed::<F, Ext>(&point);
                    assert_eq!(e0, e1);
                }

                {
                    let mut poly = poly.fix_var(*point.last().unwrap());
                    point.iter().rev().skip(1).for_each(|&var| {
                        let input_poly = poly.clone();
                        poly.fix_var_mut(var);
                        if input_poly.k() >= 4 {
                            let mut poly_packed = input_poly.pack::<F>();
                            poly_packed.fix_var_mut(var);
                            assert_eq!(poly_packed.unpack(), poly);
                        }
                    });
                    assert_eq!(e0, poly.constant().unwrap());
                }
            }
        }
    }
}
