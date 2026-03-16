pub mod eq;
pub mod point;
pub mod poly;
#[cfg(test)]
mod test;
pub mod utils;

use crate::p3_field_prelude::*;
pub use point::*;
pub use poly::*;
pub use utils::*;

pub mod prelude {
    pub use crate::{Point, Poly};
}

pub fn eval_eq_xy<F: Field, Ext: ExtensionField<F>>(x: &[F], y: &[Ext]) -> Ext {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi * xi).double() - xi - yi + F::ONE)
        .product()
}

pub fn eval_pow_xy<F: Field, Ext: ExtensionField<F>>(x: &Point<F>, y: &Point<Ext>) -> Ext {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi * xi) - yi + F::ONE)
        .product()
}

pub fn eval_poly_reference<F: Field, Ext: ExtensionField<F>>(
    poly: &[F],
    point: &Point<Ext>,
) -> Ext {
    poly.iter()
        .zip(point.eq(Ext::ONE).iter())
        .map(|(&coeff, &eq)| eq * coeff)
        .sum::<Ext>()
}

pub mod p3_field_prelude {
    pub use p3_field::{
        Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
        PrimeCharacteristicRing, dot_product, extension::BinomialExtensionField,
        extension::PackedBinomialExtensionField,
    };
}
