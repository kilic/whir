use crate::p3_field_prelude::*;

use crate::pcs::sumcheck::prover::Sumcheck;
use crate::pcs::sumcheck::split::PolyEvaluator;
use crate::pcs::sumcheck::verifier::{MultiRound, SumcheckVerifier};
use crate::pcs::sumcheck::{EqClaim, PowClaim};
use crate::pcs::test::{make_eq_claims, make_pow_claims};
use crate::transcript::test_transcript::{TestReader, TestWriter};
use crate::{
    pcs::test::{read_eq_claims, read_pow_claims},
    poly::Poly,
    transcript::{Challenge, Writer},
};
use rand::Rng;

type F = p3_koala_bear::KoalaBear;
type Ext = BinomialExtensionField<F, 4>;

#[test]
fn test_sumcheck_single_fold() {
    for k in 1..=10 {
        let mut rng = crate::test::rng(1);
        let poly = Poly::<F>::rand(&mut rng, k);
        for d in 1..=k {
            for l0 in 0..=d {
                let n_eqs = 10;
                let poly = poly.clone();
                let (proof, checkpoint_prover) = {
                    let mut transcript = TestWriter::<Vec<u8>, F>::init();
                    let mut ev = PolyEvaluator::<F, Ext>::new(poly, l0);
                    (0..n_eqs).for_each(|_| ev.add_claim(&mut transcript).unwrap());
                    let sc = Sumcheck::<F, Ext>::new(&mut transcript, ev.clone(), d).unwrap();
                    assert_eq!(sc.k(), k - d);
                    {
                        let claims = ev.clone().normalize();
                        let round = MultiRound::<F, Ext>::new(
                            claims.iter().map(EqClaim::point).collect::<Vec<_>>(),
                            vec![],
                            sc.rs(),
                            sc.alpha(),
                        );
                        assert_eq!(round.eval(sc.poly_unpacked()), sc.sum());
                    }

                    transcript.write_many(sc.poly_unpacked()).unwrap();
                    let checkpoint: F = transcript.draw();
                    let proof = transcript.finalize();
                    (proof, checkpoint)
                };

                let checkpoint_verifier = {
                    let mut transcript = TestReader::<&[u8], F>::init(&proof);
                    let eq_claims_ext =
                        read_eq_claims::<_, F, Ext>(&mut transcript, k, n_eqs).unwrap();
                    let mut verifier = SumcheckVerifier::<F, Ext>::new(k);
                    verifier
                        .fold(&mut transcript, d, &eq_claims_ext, &[])
                        .unwrap();
                    verifier.finalize(&mut transcript, None).unwrap();
                    transcript.draw()
                };
                assert_eq!(checkpoint_prover, checkpoint_verifier);
            }
        }
    }
}

#[test]
fn test_sumcheck_multi_fold() {
    let mut rng = crate::test::rng(1);
    for n_fold in 1..5 {
        for _ in 0..100 {
            let test_suite: Vec<(usize, usize, usize)> = (0..n_fold)
                .map(|i| {
                    (
                        rng.random_range(0..4),
                        if i == 0 { 0 } else { rng.random_range(0..4) },
                        rng.random_range(1..4),
                    )
                })
                .collect::<Vec<_>>();
            let l0 = rng.random_range(0..=test_suite[0].2);
            let k = test_suite.iter().map(|&(_, _, d)| d).sum::<usize>();

            let poly = Poly::<F>::rand(&mut rng, k);
            let (proof, checkpoint_prover) = {
                let mut transcript = TestWriter::<Vec<u8>, F>::init();
                let mut k_folding = k;
                let mut round = 0;

                let (mut sc, mut rounds) = {
                    let (n_eqs, n_pows, d) = test_suite[0];
                    assert_eq!(n_pows, 0);

                    let mut ev = PolyEvaluator::<F, Ext>::new(poly, l0);

                    (0..n_eqs).for_each(|_| ev.add_claim(&mut transcript).unwrap());

                    k_folding -= d;
                    round += d;

                    let sc = Sumcheck::new(&mut transcript, ev.clone(), d).unwrap();

                    assert_eq!(sc.k(), k_folding);
                    assert_eq!(k - sc.k(), round);

                    let claims = ev.clone().normalize();
                    let round = MultiRound::<F, Ext>::new(
                        claims.iter().map(EqClaim::point).collect::<Vec<_>>(),
                        vec![],
                        sc.rs(),
                        sc.alpha(),
                    );
                    assert_eq!(round.eval(sc.poly_unpacked()), sc.sum());

                    (sc, vec![round])
                };

                test_suite.iter().skip(1).for_each(|&(n_eqs, n_pows, d)| {
                    k_folding -= d;
                    round += d;

                    let eq_claims =
                        make_eq_claims::<_, F, Ext>(&mut transcript, n_eqs, sc.poly_unpacked())
                            .unwrap();

                    let pow_claims =
                        make_pow_claims::<_, F, Ext>(&mut transcript, n_pows, sc.poly_unpacked())
                            .unwrap();

                    let rs = sc
                        .fold(&mut transcript, d, &eq_claims, &pow_claims)
                        .unwrap();

                    assert_eq!(sc.k(), k_folding);
                    assert_eq!(k - sc.k(), round);

                    {
                        rounds.iter_mut().for_each(|round| round.extend(&rs));
                        let eq_points = eq_claims.iter().map(EqClaim::point).collect();
                        let pow_points = pow_claims.iter().map(PowClaim::point).collect();
                        rounds.push(MultiRound::<F, Ext>::new(
                            eq_points,
                            pow_points,
                            rs,
                            sc.alpha(),
                        ));
                    }

                    assert_eq!(
                        rounds
                            .iter()
                            .map(|round| round.eval(sc.poly_unpacked()))
                            .sum::<Ext>(),
                        sc.sum()
                    );
                });

                assert_eq!(round, k);
                assert_eq!(k_folding, 0);

                transcript.write_many(sc.poly_unpacked()).unwrap();
                let checkpoint: F = transcript.draw();
                let proof = transcript.finalize();
                (proof, checkpoint)
            };

            let checkpoint_verifier = {
                let mut transcript = TestReader::<&[u8], F>::init(&proof);
                let mut verifier = SumcheckVerifier::<F, Ext>::new(k);

                test_suite.iter().for_each(|&(n_eqs, n_pows, d)| {
                    let eq_claims = read_eq_claims(&mut transcript, verifier.k, n_eqs).unwrap();
                    let pow_claims = read_pow_claims(&mut transcript, verifier.k, n_pows).unwrap();
                    verifier
                        .fold(&mut transcript, d, &eq_claims, &pow_claims)
                        .unwrap();
                });

                verifier.finalize(&mut transcript, None).unwrap();
                let checkpoint: F = transcript.draw();
                checkpoint
            };
            assert_eq!(checkpoint_prover, checkpoint_verifier);
        }
    }
}
