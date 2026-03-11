use std::ops::RangeBounds;
use std::slice::SliceIndex;

use crate::utils::n_rand;
use crate::{Poly, p3_field_prelude::*};
use itertools::Itertools;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use rand::distr::{Distribution, StandardUniform};

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
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

    pub fn empty() -> Self {
        Self(vec![])
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

    pub fn concat(&self, other: Point<F>) -> Point<F> {
        self.iter()
            .chain(other.iter())
            .copied()
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
            return vec![scale].into();
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
