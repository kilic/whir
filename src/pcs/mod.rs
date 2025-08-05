use p3_field::{ExtensionField, Field};

use crate::{
    poly::{Point, Poly},
    transcript::{Challenge, Reader, Writer},
};

pub mod params;
pub mod stir;
pub mod sumcheck;
pub mod whir;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Claim<F: Field> {
    pub(crate) point: Point<F>,
    pub(crate) eval: F,
}

impl<F: Field> Claim<F> {
    pub fn new(point: Point<F>, eval: F) -> Self {
        Self { point, eval }
    }

    pub fn k(&self) -> usize {
        self.point.len()
    }
}

pub(crate) fn make_claims<
    Transcript,
    F: Field,
    Ex0: ExtensionField<F>,
    Ex1: ExtensionField<Ex0> + ExtensionField<F>,
>(
    transcript: &mut Transcript,
    num_points: usize,
    poly: &Poly<Ex0>,
) -> Result<Vec<Claim<Ex1>>, crate::Error>
where
    Transcript: Challenge<F, Ex1> + Writer<Ex1>,
{
    (0..num_points)
        .map(|_| {
            let point = Point::expand(poly.k(), transcript.draw());
            let eval = poly.eval_lagrange(&point);
            let claim = Claim { point, eval };
            transcript.write(claim.eval)?;
            Ok(claim)
        })
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn get_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    k: usize,
    n: usize,
) -> Result<Vec<Claim<Ext>>, crate::Error>
where
    Transcript: Reader<Ext> + Challenge<F, Ext>,
{
    (0..n)
        .map(|_| {
            let var: Ext = transcript.draw();
            let point = Point::expand(k, var);
            let eval = transcript.read()?;
            Ok(Claim::new(point, eval))
        })
        .collect::<Result<Vec<_>, _>>()
}

#[cfg(test)]
mod test {
    use crate::poly::{Point, Poly};
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::Rng;

    #[test]
    fn test_whir_final_folding() {
        type F = p3_goldilocks::Goldilocks;
        use p3_field::PrimeCharacteristicRing;
        use p3_field::TwoAdicField;

        let k = 7usize;
        let mut rng = &mut crate::test::rng(1);
        let poly_in_coeffs: Poly<F> = Poly::rand(&mut rng, k);
        let poly_in_lagrange: Poly<F> = poly_in_coeffs.clone().to_lagrange();
        let size = 1 << k;

        for folding in 0..k {
            let width = 1 << folding;

            // rearrange the coefficients since we will fix variables backwards
            let mut transposed = vec![F::ZERO; size];
            transpose::transpose(&poly_in_coeffs, &mut transposed, size / width, width);

            for extend in 0..5 {
                let rate = k + extend;

                // pad as rate
                let pad_size = 1 << rate;
                let mut padded: Vec<F> = crate::utils::unsafe_allocate_zero_vec(pad_size);
                padded[..poly_in_coeffs.len()].copy_from_slice(&transposed);

                // encode and find the leaves that would be committed to
                let leaves = Radix2DitParallel::default()
                    .dft_algebra_batch(RowMajorMatrix::new(padded, width));

                // simulate sumcheck folding
                let mut poly_in_lagrange = poly_in_lagrange.clone();
                let rs: Point<F> = (0..folding)
                    .map(|_| {
                        let r: F = rng.random();
                        poly_in_lagrange.fix_var(r);
                        r
                    })
                    .collect::<Vec<_>>()
                    .into();

                // poly to send to verifier
                let poly_in_coeffs = poly_in_lagrange.to_coeffs();

                // find the domain generator
                let domain_size = rate - folding;
                let omega = F::two_adic_generator(domain_size);

                // check for each index
                for (index, row) in leaves.rows().enumerate() {
                    let wi = omega.exp_u64(index as u64);
                    let point = Point::<F>::expand(poly_in_coeffs.k(), wi);
                    let e0 = poly_in_coeffs.eval_coeffs(&point);
                    let e1 = poly_in_coeffs.eval_univariate(wi);
                    assert_eq!(e0, e1);

                    let row: Poly<F> = row.collect::<Vec<_>>().into();
                    let e1 = row.eval_coeffs(&rs.reversed());
                    assert_eq!(e0, e1);
                }
            }
        }
    }
}
