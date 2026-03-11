use crate::eq::SplitEq;
use crate::utils::n_rand;
use crate::{Point, p3_field_prelude::*};
use p3_matrix::dense::RowMajorMatrixView;
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Poly<F>(pub(crate) Vec<F>);
pub type PackedPoly<F> = Poly<<F as Field>::Packing>;
pub type PackedExtPoly<F, Ext> = Poly<<Ext as ExtensionField<F>>::ExtensionPacking>;

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

    pub fn reverse_vars(self) -> Self {
        let mut mat = DenseMatrix::new(self.0, 1);
        reverse_matrix_index_bits(&mut mat);
        mat.values.into()
    }

    pub fn mat(&self, width: usize) -> RowMajorMatrixView<'_, F> {
        RowMajorMatrixView::new(&self.0, width)
    }
}

impl<A: Clone + Copy + Default + Send + Sync> Poly<A> {
    pub fn eval_base<Ext>(&self, point: &Point<Ext>) -> Ext
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        SplitEq::<A, Ext>::new(point, Ext::ONE).eval_base(self)
    }

    pub fn eval_ext<F>(&self, point: &Point<A>) -> A
    where
        F: Field,
        A: ExtensionField<F>,
    {
        SplitEq::<F, A>::new(point, A::ONE).eval_ext(self)
    }

    pub fn eval_packed<F, Ext>(&self, point: &Point<Ext>) -> Ext
    where
        F: Field,
        Ext: ExtensionField<F, ExtensionPacking = A>,
        A: PackedFieldExtension<F, Ext>,
    {
        SplitEq::<F, Ext>::new(point, Ext::ONE).eval_packed(self)
    }

    pub fn fix_hi_var<Ext>(&self, zi: Ext) -> Poly<Ext>
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

    pub fn fix_hi_var_mut<F: Clone + Copy + Default + Send + Sync>(&mut self, zi: F)
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

    pub fn fix_lo_var_mut<F: Clone + Copy + Default + Send + Sync>(&mut self, zi: F)
    where
        A: Algebra<F>,
    {
        let src = self
            .par_chunks(2)
            .map(|a| (a[1] - a[0]) * zi + a[0])
            .collect::<Vec<_>>();
        let mid = self.len() / 2;
        self.truncate(mid);
        self.copy_from_slice(&src);
    }

    pub fn fix_hi_var_to_packed<Ext>(&self, zi: Ext) -> Poly<Ext::ExtensionPacking>
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

    pub fn fix_lo_var<Ext>(&self, zi: Ext) -> Poly<Ext>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        self.par_chunks(2)
            .map(|a| zi * (a[1] - a[0]) + a[0])
            .collect::<Vec<_>>()
            .into()
    }

    pub fn compress_lo<Ext>(&self, point: &Point<Ext>, scale: Ext) -> Poly<Ext>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        SplitEq::<A, Ext>::new(point, scale).compress_lo(self)
    }

    pub fn compress_hi<Ext>(&self, point: &Point<Ext>, scale: Ext) -> Poly<Ext>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        SplitEq::<A, Ext>::new(point, scale).compress_hi(self)
    }

    pub fn compress_lo_to_packed<Ext>(&self, _point: &Point<Ext>) -> Poly<Ext::ExtensionPacking>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        todo!()
        // assert!(point.len() <= self.k());
        // let eq: Poly<Ext> = point.eq(Ext::ONE);
        // let chunk_size = self.len() / eq.len();
        // let mut out = Ext::ExtensionPacking::zero_vec(chunk_size / A::Packing::WIDTH);
        // self.chunks(chunk_size)
        //     .zip_eq(eq.iter())
        //     .for_each(|(chunk, &w)| {
        //         let w = Ext::ExtensionPacking::from(w);
        //         let chunk = A::Packing::pack_slice(chunk);
        //         out.par_iter_mut()
        //             .zip_eq(chunk.par_iter())
        //             .for_each(|(acc, &f)| *acc += w * f);
        //     });
        // out.into()
    }

    pub fn compress_hi_to_packed<Ext>(&self, _point: &Point<Ext>) -> Poly<Ext::ExtensionPacking>
    where
        A: Field,
        Ext: ExtensionField<A>,
    {
        todo!()
        // assert!(self.k() - point.len() >= log2_strict_usize(A::Packing::WIDTH));

        // let eq: Poly<Ext> = point.eq(Ext::ONE);
        // let chunk_size = eq.len();
        // self.par_chunks(chunk_size * A::Packing::WIDTH)
        //     .map(|chunk| {
        //         (0..chunk_size)
        //             .map(|i| {
        //                 Ext::ExtensionPacking::from(eq[i])
        //                     * A::Packing::from_fn(|j| chunk[j * chunk_size + i])
        //             })
        //             .sum::<Ext::ExtensionPacking>()
        //     })
        //     .collect::<Vec<_>>()
        //     .into()
    }

    pub fn pack<F>(&self) -> Poly<A::ExtensionPacking>
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
}
