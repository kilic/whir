use crate::sumcheck::prover2::ProverStack;
use crate::sumcheck::verifier2::{
    SumcheckVerifier as SumcheckVerifier, VerifierStack as VerifierStack,
};
use crate::sumcheck::{EqClaim, PowClaim};
use common::Error;
use common::field::*;
use poly::{Point, Poly};
use rand::RngExt;
use transcript::{Challenge, Reader, Writer};

use p3_koala_bear::KoalaBear;
use transcript::test_transcript::{TestReader, TestWriter};
type F = KoalaBear;
type Ext = BinomialExtensionField<F, 4>;

pub(crate) fn make_eq_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    n_points: usize,
    poly: &Poly<Ext>,
) -> Result<Vec<EqClaim<Ext>>, Error>
where
    Transcript: Challenge<F, Ext> + Writer<Ext>,
{
    (0..n_points)
        .map(|_| {
            let point = Point::expand(poly.k(), transcript.draw());
            let eval = poly.eval_base(&point);
            let claim = EqClaim { point, eval };
            transcript.write(claim.eval)?;
            Ok(claim)
        })
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn make_pow_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    n_points: usize,
    poly: &Poly<Ext>,
) -> Result<Vec<PowClaim<F, Ext>>, Error>
where
    Transcript: Challenge<F, F> + Writer<Ext>,
{
    let k = poly.k();
    (0..n_points)
        .map(|_| {
            let var = transcript.draw();
            let eval = poly
                .iter()
                .zip(var.powers().take(poly.len()))
                .map(|(&c, p)| c * p)
                .sum();
            let claim = PowClaim { var, eval, k };
            transcript.write(claim.eval)?;
            Ok(claim)
        })
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn read_eq_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    k: usize,
    n: usize,
) -> Result<Vec<EqClaim<Ext>>, Error>
where
    Transcript: Reader<Ext> + Challenge<F, Ext>,
{
    (0..n)
        .map(|_| {
            let var: Ext = transcript.draw();
            let point = Point::expand(k, var);
            let eval = transcript.read()?;
            Ok(EqClaim::new(point, eval))
        })
        .collect::<Result<Vec<_>, _>>()
}

pub(crate) fn read_pow_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    k: usize,
    n: usize,
) -> Result<Vec<PowClaim<F, Ext>>, Error>
where
    Transcript: Reader<Ext> + Challenge<F, F>,
{
    (0..n)
        .map(|_| {
            let var: F = transcript.draw();
            let eval = transcript.read()?;
            Ok(PowClaim::new(var, eval, k))
        })
        .collect::<Result<Vec<_>, _>>()
}

#[test]
fn test_leanvm() {
    let mut rng = common::test::rng(1);

    let d = 7;

    let mut transcript = TestWriter::<Vec<u8>, F>::init();
    let mut stack = ProverStack::<F, Ext>::new();
    let ids_a = (0..20)
        .map(|_| stack.register(Poly::<F>::rand(&mut rng, 20)))
        .collect::<Vec<_>>();
    let ids_b = (0..29)
        .map(|_| stack.register(Poly::<F>::rand(&mut rng, 8)))
        .collect::<Vec<_>>();
    let ids_c = (0..96)
        .map(|_| stack.register(Poly::<F>::rand(&mut rng, 8)))
        .collect::<Vec<_>>();

    tracing::info_span!("evals").in_scope(|| {
        let apply_shared_point = |stack: &mut ProverStack<F, Ext>,
                                  transcript: &mut TestWriter<Vec<u8>, F>,
                                  ids: &[usize],
                                  k: usize,
                                  selected: &[usize]| {
            let point = Point::expand(k, transcript.draw());
            for &i in selected {
                stack.eval(transcript, ids[i], &point).unwrap();
            }
        };

        // Table A: 20 polys, k = 20
        let p0 = vec![
            0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        ];
        let p1 = vec![0, 1];
        let p2 = (0..20).collect::<Vec<_>>();
        apply_shared_point(&mut stack, &mut transcript, &ids_a, 20, &p0);
        apply_shared_point(&mut stack, &mut transcript, &ids_a, 20, &p1);
        apply_shared_point(&mut stack, &mut transcript, &ids_a, 20, &p2);

        // Table B: 29 polys, k = 8
        let p3 = (6..=23).collect::<Vec<_>>();
        let p4 = (0..=7).chain(24..=28).collect::<Vec<_>>();
        let p5 = (0..29).collect::<Vec<_>>();
        apply_shared_point(&mut stack, &mut transcript, &ids_b, 8, &p3);
        apply_shared_point(&mut stack, &mut transcript, &ids_b, 8, &p4);
        apply_shared_point(&mut stack, &mut transcript, &ids_b, 8, &p5);

        // Table C: 96 polys, k = 8
        let p6 = (1..=19).chain(88..=95).collect::<Vec<_>>();
        let p7 = (0..96).collect::<Vec<_>>();
        apply_shared_point(&mut stack, &mut transcript, &ids_c, 8, &p6);
        apply_shared_point(&mut stack, &mut transcript, &ids_c, 8, &p7);
    });

    let mut layout = stack.layout();
    assert_eq!(layout.k(), 25);
    layout.write_virtual_eval(&mut transcript).unwrap();
    layout.write_virtual_eval(&mut transcript).unwrap();
    let mut prover = layout.new_prover(&mut transcript, d).unwrap();

    let eq_claims = make_eq_claims(&mut transcript, 2, prover.poly()).unwrap();
    let pow_claims = make_pow_claims(&mut transcript, 100, prover.poly()).unwrap();
    prover
        .fold(&mut transcript, 5, &eq_claims, &pow_claims)
        .unwrap();

    prover.write_poly(&mut transcript).unwrap();
    let checkpoint_prover: F = transcript.draw();
    let proof = transcript.finalize();

    // Verifier
    let mut transcript = TestReader::<&[u8], F>::init(&proof);
    let mut stack = VerifierStack::<F, Ext>::new();
    let ids_a_v = (0..20).map(|_| stack.register(20)).collect::<Vec<_>>();
    let ids_b_v = (0..29).map(|_| stack.register(8)).collect::<Vec<_>>();
    let ids_c_v = (0..96).map(|_| stack.register(8)).collect::<Vec<_>>();

    let read_shared_point = |stack: &mut VerifierStack<F, Ext>,
                             transcript: &mut TestReader<&[u8], F>,
                             ids: &[usize],
                             k: usize,
                             selected: &[usize]| {
        let point = Point::expand(k, transcript.draw());
        for &i in selected {
            let _ = stack.read_eval(transcript, ids[i], &point).unwrap();
        }
    };

    // Table A: 20 polys, k = 20
    let p0 = vec![
        0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    ];
    let p1 = vec![0, 1];
    let p2 = (0..20).collect::<Vec<_>>();
    read_shared_point(&mut stack, &mut transcript, &ids_a_v, 20, &p0);
    read_shared_point(&mut stack, &mut transcript, &ids_a_v, 20, &p1);
    read_shared_point(&mut stack, &mut transcript, &ids_a_v, 20, &p2);

    // Table B: 29 polys, k = 8
    let p3 = (6..=23).collect::<Vec<_>>();
    let p4 = (0..=7).chain(24..=28).collect::<Vec<_>>();
    let p5 = (0..29).collect::<Vec<_>>();
    read_shared_point(&mut stack, &mut transcript, &ids_b_v, 8, &p3);
    read_shared_point(&mut stack, &mut transcript, &ids_b_v, 8, &p4);
    read_shared_point(&mut stack, &mut transcript, &ids_b_v, 8, &p5);

    // Table C: 96 polys, k = 8
    let p6 = (1..=19).chain(88..=95).collect::<Vec<_>>();
    let p7 = (0..96).collect::<Vec<_>>();
    read_shared_point(&mut stack, &mut transcript, &ids_c_v, 8, &p6);
    read_shared_point(&mut stack, &mut transcript, &ids_c_v, 8, &p7);

    let mut layout = stack.layout();
    let _ = layout.read_virtual_eval(&mut transcript).unwrap();
    let _ = layout.read_virtual_eval(&mut transcript).unwrap();
    let (mut verifier, _) = SumcheckVerifier::<F, Ext>::new(&mut transcript, layout, d).unwrap();

    let eq_claims = read_eq_claims(&mut transcript, verifier.k(), 2).unwrap();
    let pow_claims = read_pow_claims(&mut transcript, verifier.k(), 100).unwrap();
    verifier
        .fold(&mut transcript, 5, &eq_claims, &pow_claims)
        .unwrap();

    verifier.finalize(&mut transcript, None).unwrap();
    let checkpoint_verifier: F = transcript.draw();
    assert_eq!(checkpoint_prover, checkpoint_verifier);
}

#[test]
fn test_batch_sumcheck1() {
    let mut rng = common::test::rng(1);

    let d = 4;
    let k0 = 10;

    let f0 = Poly::<F>::rand(&mut rng, k0);

    let mut transcript = TestWriter::<Vec<u8>, F>::init();
    let mut stack = ProverStack::<F, Ext>::new();
    let id0 = stack.register(f0);

    let point = Point::expand(stack.k_poly(id0), transcript.draw());
    stack.eval(&mut transcript, id0, &point).unwrap();
    let mut layout = stack.layout();
    layout.write_virtual_eval(&mut transcript).unwrap();
    layout.write_virtual_eval(&mut transcript).unwrap();
    let mut prover = layout.new_prover(&mut transcript, d).unwrap();
    let eq_claims = make_eq_claims(&mut transcript, 2, prover.poly()).unwrap();
    let pow_claims = make_pow_claims(&mut transcript, 10, prover.poly()).unwrap();
    prover
        .fold(&mut transcript, 2, &eq_claims, &pow_claims)
        .unwrap();

    prover.write_poly(&mut transcript).unwrap();
    let checkpoint_prover: F = transcript.draw();
    let proof = transcript.finalize();

    let mut transcript = TestReader::<&[u8], F>::init(&proof);
    let mut stack = VerifierStack::<F, Ext>::new();
    let id0 = stack.register(k0);
    let point0 = Point::expand(stack.k_poly(id0), transcript.draw());
    let _ = stack.read_eval(&mut transcript, id0, &point0).unwrap();
    let mut layout = stack.layout();
    let _ = layout.read_virtual_eval(&mut transcript).unwrap();
    let _ = layout.read_virtual_eval(&mut transcript).unwrap();
    let (mut verifier, _) = SumcheckVerifier::<F, Ext>::new(&mut transcript, layout, d).unwrap();
    let eq_claims = read_eq_claims(&mut transcript, verifier.k(), 2).unwrap();
    let pow_claims = read_pow_claims(&mut transcript, verifier.k(), 10).unwrap();
    verifier
        .fold(&mut transcript, 2, &eq_claims, &pow_claims)
        .unwrap();

    verifier.finalize(&mut transcript, None).unwrap();
    let checkpoint_verifier: F = transcript.draw();
    assert_eq!(checkpoint_prover, checkpoint_verifier);
}

#[test]
fn test_batch_sumcheck2() {
    let mut rng = common::test::rng(1);

    let d = 4;
    let k0 = 8;
    let k1 = 7;
    let k2 = 6;
    let k3 = 4;

    let f0 = Poly::<F>::rand(&mut rng, k0);
    let f1 = Poly::<F>::rand(&mut rng, k1);
    let f2 = Poly::<F>::rand(&mut rng, k2);
    let f3 = Poly::<F>::rand(&mut rng, k3);

    let mut transcript = TestWriter::<Vec<u8>, F>::init();
    let mut stack = ProverStack::<F, Ext>::new();
    let id0 = stack.register(f0);
    let id1 = stack.register(f1);
    let id2 = stack.register(f2);
    let id3 = stack.register(f3);

    let point0 = Point::expand(stack.k_poly(id0), transcript.draw());
    let point1 = Point::expand(stack.k_poly(id1), transcript.draw());
    let point2 = Point::expand(stack.k_poly(id2), transcript.draw());
    let point3 = Point::expand(stack.k_poly(id3), transcript.draw());
    stack.eval(&mut transcript, id0, &point0).unwrap();
    stack.eval(&mut transcript, id1, &point1).unwrap();
    stack.eval(&mut transcript, id2, &point2).unwrap();
    stack.eval(&mut transcript, id3, &point3).unwrap();
    let mut layout = stack.layout();
    layout.write_virtual_eval(&mut transcript).unwrap();
    layout.write_virtual_eval(&mut transcript).unwrap();
    let mut prover = layout.new_prover(&mut transcript, d).unwrap();
    let eq_claims = make_eq_claims(&mut transcript, 2, prover.poly()).unwrap();
    let pow_claims = make_pow_claims(&mut transcript, 10, prover.poly()).unwrap();
    prover
        .fold(&mut transcript, 2, &eq_claims, &pow_claims)
        .unwrap();

    prover.write_poly(&mut transcript).unwrap();
    let checkpoint_prover: F = transcript.draw();
    let proof = transcript.finalize();

    let mut transcript = TestReader::<&[u8], F>::init(&proof);
    let mut stack = VerifierStack::<F, Ext>::new();
    let id0 = stack.register(k0);
    let id1 = stack.register(k1);
    let id2 = stack.register(k2);
    let id3 = stack.register(k3);
    let point0 = Point::expand(stack.k_poly(id0), transcript.draw());
    let point1 = Point::expand(stack.k_poly(id1), transcript.draw());
    let point2 = Point::expand(stack.k_poly(id2), transcript.draw());
    let point3 = Point::expand(stack.k_poly(id3), transcript.draw());
    let _ = stack.read_eval(&mut transcript, id0, &point0).unwrap();
    let _ = stack.read_eval(&mut transcript, id1, &point1).unwrap();
    let _ = stack.read_eval(&mut transcript, id2, &point2).unwrap();
    let _ = stack.read_eval(&mut transcript, id3, &point3).unwrap();
    let mut layout = stack.layout();
    let _ = layout.read_virtual_eval(&mut transcript).unwrap();
    let _ = layout.read_virtual_eval(&mut transcript).unwrap();
    let (mut verifier, _) = SumcheckVerifier::<F, Ext>::new(&mut transcript, layout, d).unwrap();
    let eq_claims = read_eq_claims(&mut transcript, verifier.k(), 2).unwrap();
    let pow_claims = read_pow_claims(&mut transcript, verifier.k(), 10).unwrap();
    verifier
        .fold(&mut transcript, 2, &eq_claims, &pow_claims)
        .unwrap();

    verifier.finalize(&mut transcript, None).unwrap();
    let checkpoint_verifier: F = transcript.draw();
    assert_eq!(checkpoint_prover, checkpoint_verifier);
}

#[test]
fn test_batch_sumcheck3() {
    let d = 4;
    let min_poly_k = 4;
    let max_poly_k = 10;
    let n_polys = 10;

    let mut rng = common::test::rng(1);

    for _ in 0..1 {
        let mut poly_ks = vec![max_poly_k];
        let mut n_eqs0 = vec![rng.random_range::<usize, _>(1..=5)];
        for _ in 0..n_polys - 1 {
            poly_ks.push(rng.random_range::<usize, _>(min_poly_k..=max_poly_k));
            n_eqs0.push(rng.random_range::<usize, _>(1..=5));
        }

        // Prover
        let mut transcript = TestWriter::<Vec<u8>, F>::init();
        let mut stack = ProverStack::<F, Ext>::new();
        let _ = poly_ks
            .iter()
            .map(|&k| stack.register(Poly::<F>::rand(&mut rng, k)))
            .collect::<Vec<_>>();

        n_eqs0.iter().enumerate().for_each(|(id, &n_claim)| {
            for _ in 0..n_claim {
                let point = Point::expand(poly_ks[id], transcript.draw());
                stack.eval(&mut transcript, id, &point).unwrap();
            }
        });

        let mut prover = stack.layout().new_prover(&mut transcript, d).unwrap();

        let n_fold = 2;
        let mut n_eqs = vec![];
        let mut n_pows = vec![];
        let mut ds = vec![];
        for _ in 0..n_fold {
            n_eqs.push(rng.random_range::<usize, _>(0..=3));
            n_pows.push(rng.random_range::<usize, _>(0..=3));
            ds.push(2)
        }

        ds.iter()
            .zip(n_eqs.iter())
            .zip(n_pows.iter())
            .for_each(|((&d, &n_eq), &n_pow)| {
                let eq_claims = make_eq_claims(&mut transcript, n_eq, prover.poly()).unwrap();
                let pow_claims = make_pow_claims(&mut transcript, n_pow, prover.poly()).unwrap();
                prover
                    .fold(&mut transcript, d, &eq_claims, &pow_claims)
                    .unwrap();
            });

        prover.write_poly(&mut transcript).unwrap();
        let checkpoint_prover: F = transcript.draw();
        let proof = transcript.finalize();

        // Verifier
        let mut transcript = TestReader::<&[u8], F>::init(&proof);
        let mut stack = VerifierStack::<F, Ext>::new();
        poly_ks.iter().for_each(|&k| {
            stack.register(k);
        });

        n_eqs0.iter().enumerate().for_each(|(id, &n_claim)| {
            for _ in 0..n_claim {
                let point = Point::expand(stack.k_poly(id), transcript.draw());
                let _ = stack.read_eval(&mut transcript, id, &point).unwrap();
            }
        });
        let (mut verifier, _) = SumcheckVerifier::new(&mut transcript, stack.layout(), d).unwrap();
        ds.iter()
            .zip(n_eqs.iter())
            .zip(n_pows.iter())
            .for_each(|((&d, &n_eq), &n_pow)| {
                let eq_claims = read_eq_claims(&mut transcript, verifier.k(), n_eq).unwrap();
                let pow_claims = read_pow_claims(&mut transcript, verifier.k(), n_pow).unwrap();
                verifier
                    .fold(&mut transcript, d, &eq_claims, &pow_claims)
                    .unwrap();
            });

        verifier.finalize(&mut transcript, None).unwrap();
        let checkpoint_verifier: F = transcript.draw();
        assert_eq!(checkpoint_prover, checkpoint_verifier);
    }
}
