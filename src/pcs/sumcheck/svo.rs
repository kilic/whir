use p3_util::log2_strict_usize;

use crate::p3_field_prelude::*;
use crate::pcs::sumcheck::lagrange_weights_012;
use crate::pcs::sumcheck::split::log3_strict_usize;
use crate::poly::Point;

pub(crate) type SvoAccumulators<A> = Vec<[Vec<A>; 2]>;

#[tracing::instrument(skip_all)]
pub(crate) fn calculate_accumulators<F: Field, Ext: ExtensionField<F>>(
    partial_evals: &[Ext],
    suffix: &Point<Ext>,
) -> SvoAccumulators<Ext> {
    (1..=suffix.k())
        .map(|i| {
            let us = points_012::<F>(i);
            [
                calculate_accumulators_inner(&us[0], partial_evals, suffix),
                calculate_accumulators_inner(&us[1], partial_evals, suffix),
            ]
        })
        .collect::<Vec<_>>()
}

pub(crate) fn points_012<F: Field>(l: usize) -> [Vec<Point<F>>; 2] {
    fn expand<F: Field>(pts: &[Point<F>], values: &[usize]) -> Vec<Point<F>> {
        values
            .iter()
            .flat_map(|&v| {
                pts.iter().cloned().map(move |mut p| {
                    p.push(F::from_u32(v as u32));
                    p
                })
            })
            .collect::<Vec<_>>()
    }

    assert!(l > 0);
    let mut pts = vec![Point::new(vec![])];
    for _ in 0..l - 1 {
        pts = expand(&pts, &[0, 1, 2]);
    }
    let mut pts0 = expand(&pts, &[0]);
    let mut pts2 = expand(&pts, &[2]);
    pts0.iter_mut().for_each(Point::reverse);
    pts2.iter_mut().for_each(Point::reverse);
    [pts0, pts2]
}

fn calculate_accumulators_inner<F: Field, Ext: ExtensionField<F>>(
    us: &[Point<F>],
    partial_evals: &[Ext],
    z_svo: &Point<Ext>,
) -> Vec<Ext> {
    let l0 = log2_strict_usize(partial_evals.len());
    assert_eq!(l0, z_svo.len());
    let off = l0 - log3_strict_usize(us.len()) - 1;

    let (z1, z0) = z_svo.split_at(off);
    let eq0 = z0.eq(Ext::ONE);
    let eq1 = z1.eq(Ext::ONE);

    let partial_evals: Vec<Ext> = partial_evals
        .chunks(eq1.len())
        .map(|chunk| dot_product::<Ext, _, _>(eq1.iter().copied(), chunk.iter().copied()))
        .collect();

    us.iter()
        .map(|u| {
            let coeffs = u.eq(F::ONE);
            dot_product::<Ext, _, _>(eq0.iter().copied(), coeffs.iter().copied())
                * dot_product::<Ext, _, _>(partial_evals.iter().copied(), coeffs.iter().copied())
        })
        .collect()
}

pub(super) fn lagrange_weights_012_multi<F: Field>(rs: &[F]) -> Vec<F> {
    let mut weights = vec![F::ONE];
    rs.iter().for_each(|&r| {
        weights = lagrange_weights_012(r)
            .iter()
            .flat_map(|li| weights.iter().map(|&w| w * *li))
            .collect();
    });
    weights
}

#[cfg(test)]
mod test {
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use p3_koala_bear::KoalaBear;

    use crate::{
        pcs::sumcheck::{split::SplitPoint, svo::points_012},
        poly::{Point, Poly},
    };

    #[test]
    fn test_accumulators() {
        type F = KoalaBear;
        type Ext = F;
        let k = 12;
        let mut rng = crate::test::rng(1);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<Ext>::rand(&mut rng, k);
        let eq: Poly<Ext> = point.eq(Ext::ONE);
        let e0 = poly.eval(&point);

        for l0 in 0..k / 2 {
            let split_point = SplitPoint::<F, Ext>::new(point.clone(), Some(l0));
            let (accumulators, e1) = split_point.eval(&poly);
            assert_eq!(e0, e1);

            for (i, accumulator) in accumulators.iter().enumerate() {
                let us = points_012::<F>(i + 1);
                for (us, accs) in us.iter().zip(accumulator.iter()) {
                    us.iter().zip(accs.iter()).for_each(|(u, &acc)| {
                        let poly = poly.partial_eval(u);
                        let eq = eq.partial_eval(u);
                        assert_eq!(acc, dot_product(poly.iter().copied(), eq.iter().copied()));
                    });
                }
            }
        }
    }
}
