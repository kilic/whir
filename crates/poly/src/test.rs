use crate::eq::{SplitEq, compress_hi_reference, compress_lo_reference};
use crate::{Point, Poly, eval_eq_xy, eval_poly_reference};
use p3_field::PackedValue;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_util::log2_strict_usize;
type F = p3_koala_bear::KoalaBear;
type Ext = BinomialExtensionField<F, 4>;
use rand::RngExt;

fn combine_into_reference(out: &mut [Ext], eq: &[Ext], alpha: Option<Ext>) {
    match alpha {
        Some(alpha) => out
            .iter_mut()
            .zip(eq.iter())
            .for_each(|(out, &eq)| *out += eq * alpha),
        None => out
            .iter_mut()
            .zip(eq.iter())
            .for_each(|(out, &eq)| *out += eq),
    }
}

#[test]
fn test_eq() {
    let mut rng = &mut common::test::rng(1);
    for k in 1..10 {
        let x = Point::<F>::rand(&mut rng, k);
        let y = Point::<Ext>::rand(&mut rng, k);
        let e0 = eval_eq_xy(&x, &y);
        let e1 = x.eq(F::ONE).eval_base(&y);
        assert_eq!(e0, e1);
    }
}

#[test]
fn test_univariate() {
    let mut rng = common::test::rng(1);
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
fn test_poly() {
    let mut rng = common::test::rng(1);

    // base field
    for k in 1..=10 {
        let poly = Poly::<F>::rand(&mut rng, k);
        let poly_rev = poly.clone().reverse_vars();
        let point: Point<Ext> = Point::rand(&mut rng, k);

        let e0 = eval_poly_reference(&poly, &point);
        assert_eq!(e0, poly.eval_base(&point));
        assert_eq!(e0, poly_rev.eval_base(&point.reversed()));
        assert_eq!(e0, poly.compress_lo(&point, Ext::ONE).constant().unwrap());
        assert_eq!(e0, poly.compress_hi(&point, Ext::ONE).constant().unwrap());
        for d in 1..k {
            let (z0, z1) = point.split_at(d);
            assert_eq!(e0, poly.compress_lo(&z0, Ext::ONE).eval_ext::<F>(&z1));
            assert_eq!(e0, poly.compress_hi(&z1, Ext::ONE).eval_ext::<F>(&z0));

            let mut fixed: Poly<Ext> = poly.fix_hi_var(z0[0]);
            assert_eq!(fixed, poly.compress_hi(&z0.range(..1), Ext::ONE));
            assert_eq!(fixed, poly_rev.fix_lo_var(z0[0]).reverse_vars());
            for i in 1..d {
                fixed.fix_hi_var_mut(z0[i]);
                assert_eq!(
                    fixed,
                    poly.compress_hi(&z0.range(..=i).reversed(), Ext::ONE)
                );
                assert_eq!(
                    fixed,
                    poly_rev
                        .compress_lo(&z0.range(..=i), Ext::ONE)
                        .reverse_vars()
                );
            }
        }
    }

    // ext field
    for k in 1..=10 {
        let poly = Poly::<Ext>::rand(&mut rng, k);
        let poly_rev = poly.clone().reverse_vars();
        let point: Point<Ext> = Point::rand(&mut rng, k);

        let e0 = eval_poly_reference(&poly, &point);
        assert_eq!(e0, poly.eval_ext::<F>(&point));
        assert_eq!(e0, poly.compress_lo(&point, Ext::ONE).constant().unwrap());
        assert_eq!(e0, poly.compress_hi(&point, Ext::ONE).constant().unwrap());
        if SplitEq::<F, Ext>::can_pack(k / 2) {
            assert_eq!(e0, poly.pack::<F>().eval_packed::<F, Ext>(&point));
        }

        for d in 1..k {
            let (z0, z1) = point.split_at(d);
            assert_eq!(e0, poly.compress_lo(&z0, Ext::ONE).eval_ext::<F>(&z1));
            assert_eq!(e0, poly.compress_hi(&z1, Ext::ONE).eval_ext::<F>(&z0));
            let mut fixed: Poly<Ext> = poly.fix_hi_var(z0[0]);
            assert_eq!(fixed, poly.compress_hi(&z0.range(..1), Ext::ONE));
            assert_eq!(fixed, poly_rev.fix_lo_var(z0[0]).reverse_vars());
            for i in 1..d {
                fixed.fix_hi_var_mut(z0[i]);
                assert_eq!(
                    fixed,
                    poly.compress_hi(&z0.range(..=i).reversed(), Ext::ONE)
                );
                assert_eq!(
                    fixed,
                    poly_rev
                        .compress_lo(&z0.range(..=i), Ext::ONE)
                        .reverse_vars()
                );
            }
        }
    }
}

#[test]
fn test_eq_compress_hi() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;

    for poly_k in [1, 2, 3, 4, 5, 10, 15, 20] {
        println!();
        println!("k = {poly_k}");
        for point_k in 1..=poly_k {
            println!();
            println!("point_k = {point_k}");

            let poly = Poly::<F>::rand(&mut rng, poly_k);
            let point: Point<Ext> = Point::rand(&mut rng, point_k);
            let out0 = compress_hi_reference(&poly, &point);

            common::test::bench(
                "partial_eval",
                n_iter,
                || {},
                |_| {},
                |_| {},
                |_, _, _| compress_hi_reference(&poly, &point),
                |_, out1| assert_eq!(out0, out1),
            );

            common::test::bench(
                "split unpacked",
                n_iter,
                || {},
                |_| {},
                |_| {},
                |_, _, _| SplitEq::<F, Ext>::new_unpacked(&point).compress_hi(&poly),
                |_, out1| assert_eq!(out0, out1),
            );

            common::test::bench(
                "split packed",
                n_iter,
                || {},
                |_| {},
                |_| {},
                |_, _, _| SplitEq::<F, Ext>::new(&point, Ext::ONE).compress_hi(&poly),
                |_, out1| assert_eq!(out0, out1),
            );
        }
    }
}

#[test]
fn test_eq_compress_lo() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;

    for poly_k in [1, 2, 3, 4, 5, 10, 15, 20] {
        println!();
        println!("k = {poly_k}");
        for point_k in 1..=poly_k {
            println!();
            println!("point_k = {point_k}");

            let poly = Poly::<F>::rand(&mut rng, poly_k);
            let point: Point<Ext> = Point::rand(&mut rng, point_k);
            let out0 = compress_lo_reference(&poly, &point);

            common::test::bench(
                "partial_eval",
                n_iter,
                || {},
                |_| {},
                |_| {},
                |_, _, _| compress_lo_reference(&poly, &point),
                |_, out1| assert_eq!(out0, out1),
            );

            common::test::bench(
                "split unpacked",
                n_iter,
                || {},
                |_| {},
                |_| {},
                |_, _, _| SplitEq::<F, Ext>::new_unpacked(&point).compress_lo(&poly),
                |_, out1| assert_eq!(out0, out1),
            );

            common::test::bench(
                "split packed",
                n_iter,
                || {},
                |_| {},
                |_| {},
                |_, _, _| SplitEq::<F, Ext>::new(&point, Ext::ONE).compress_lo(&poly),
                |_, out1| assert_eq!(out0, out1),
            );
        }
    }
}

#[test]
fn test_eval_base() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;

    for k in 10..20 {
        println!("{k}");
        let poly = Poly::<F>::rand(&mut rng, k);
        let point: Point<Ext> = Point::rand(&mut rng, k);
        let e0 = eval_poly_reference(&poly, &point);
        common::test::bench(
            "eval base packed",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| SplitEq::<F, Ext>::new(&point, Ext::ONE).eval_base(&poly),
            |_, e1| assert_eq!(e0, e1),
        );

        common::test::bench(
            "eval base unpacked",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| SplitEq::<F, Ext>::new_unpacked(&point).eval_base(&poly),
            |_, e1| assert_eq!(e0, e1),
        );
    }
}

#[test]
fn test_eval_ext() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;

    for k in 10..20 {
        println!("{k}");
        let poly = Poly::<Ext>::rand(&mut rng, k);
        let point: Point<Ext> = Point::rand(&mut rng, k);
        let e0 = eval_poly_reference(&poly, &point);
        common::test::bench(
            "eval ext packed",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| SplitEq::<F, Ext>::new(&point, Ext::ONE).eval_ext(&poly),
            |_, e1| assert_eq!(e0, e1),
        );

        common::test::bench(
            "eval ext unpacked",
            n_iter,
            || {},
            |_| {},
            |_| {},
            |_, _, _| SplitEq::<F, Ext>::new_unpacked(&point).eval_ext(&poly),
            |_, e1| assert_eq!(e0, e1),
        );
    }
}

#[test]
fn test_combine_into() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;

    for k in [1, 2, 3, 4, 5, 10, 12, 14, 15, 16, 18, 20] {
        println!();
        println!("k = {k}");

        let point: Point<Ext> = Point::rand(&mut rng, k);
        let alpha = Ext::ONE;
        let eq: Poly<Ext> = point.eq(Ext::ONE);
        let split_unpacked = SplitEq::<F, Ext>::new_unpacked(&point);
        let split_packed = SplitEq::<F, Ext>::new(&point, Ext::ONE);

        let out_init: Vec<Ext> = vec![Ext::ZERO; 1 << k];

        let mut expected_none = out_init.clone();
        combine_into_reference(&mut expected_none, &eq, None);

        let mut expected_alpha = out_init.clone();
        combine_into_reference(&mut expected_alpha, &eq, Some(alpha));

        common::test::bench(
            "combine ref",
            n_iter,
            || {},
            |_| out_init.clone(),
            |_| {},
            |_, _, out| combine_into_reference(out, &eq, None),
            |out, _| assert_eq!(out, expected_none),
        );

        common::test::bench(
            "unpacked",
            n_iter,
            || {},
            |_| out_init.clone(),
            |_| {},
            |_, _, out| split_unpacked.combine_into(out, None),
            |out, _| assert_eq!(out, expected_none),
        );

        common::test::bench(
            "packed",
            n_iter,
            || {},
            |_| out_init.clone(),
            |_| {},
            |_, _, out| split_packed.combine_into(out, None),
            |out, _| assert_eq!(out, expected_none),
        );
    }
}

#[test]
fn test_combine_into_packed() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;
    type PackedF = <F as p3_field::Field>::Packing;
    type PackedExt = <Ext as p3_field::ExtensionField<F>>::ExtensionPacking;
    let k_pack = log2_strict_usize(PackedF::WIDTH);
    let alpha = Ext::ONE;

    for k in [10, 12, 14, 15, 16, 18, 20] {
        println!();
        println!("k = {k}");

        let point: Point<Ext> = Point::rand(&mut rng, k);
        let split_packed = SplitEq::<F, Ext>::new(&point, Ext::ONE);
        let out_init: Vec<PackedExt> = PackedExt::zero_vec(1 << (k - k_pack));

        let expected = point.eq_packed::<F>(Ext::ONE);
        let expected: Vec<PackedExt> = expected.iter().copied().collect();

        common::test::bench(
            "combine into packed",
            n_iter,
            || {},
            |_| out_init.clone(),
            |_| {},
            |_, _, out| split_packed.combine_into_packed(out, alpha),
            |out, _| assert_eq!(out, expected),
        );
    }
}

#[test]
fn test_fix_var_mut_bench() {
    let mut rng = common::test::rng(1);
    let n_iter = 100;

    for k in [8, 10, 12, 14, 16, 18, 20] {
        println!();
        println!("k = {k}");

        let poly = Poly::<F>::rand(&mut rng, k);
        let zi: F = rng.random();

        let expected_hi = poly.fix_hi_var(zi);
        let expected_lo = poly.fix_lo_var(zi);

        common::test::bench(
            "fix_hi_var_mut",
            n_iter,
            || {},
            |_| poly.clone(),
            |_| {},
            |_, _, out| {
                out.fix_hi_var_mut(zi);
            },
            |out, _| assert_eq!(expected_hi, out),
        );

        common::test::bench(
            "fix_lo_var_mut",
            n_iter,
            || {},
            |_| poly.clone(),
            |_| {},
            |_, _, out| {
                out.fix_lo_var_mut(zi);
            },
            |out, _| assert_eq!(expected_lo, out),
        );
    }
}
