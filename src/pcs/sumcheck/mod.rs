use p3_field::{dot_product, ExtensionField, Field};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    pcs::{sumcheck::compress::compress_claims, EqClaim, PowClaim},
    poly::{eval_eq_xy, eval_pow_xy, Eval, Point},
    transcript::{Challenge, Reader, Writer},
    utils::VecOps,
};

mod compress;

type Poly<F> = crate::poly::Poly<F, Eval>;

fn extrapolate<F: Field, EF: ExtensionField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from_usize(i))
        .collect::<Vec<_>>();
    crate::utils::arithmetic::interpolate(&points, evals)
        .iter()
        .horner(target)
}

#[tracing::instrument(skip_all, fields(k = poly.k()))]
fn round<
    Transcript,
    F: Field,
    Ex0: ExtensionField<F>,
    Ex1: ExtensionField<Ex0> + ExtensionField<F>,
>(
    transcript: &mut Transcript,
    sum: &mut Ex1,
    poly: &Poly<Ex0>,
    weights: &mut Poly<Ex1>,
) -> Result<Ex1, crate::Error>
where
    Transcript: Writer<Ex1> + Challenge<F, Ex1>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = weights.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip(phi.par_iter())
        .zip(elo.par_iter().zip(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ex1::ZERO, Ex1::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    transcript.write(v0)?;
    transcript.write(v2)?;
    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ex1>::draw(transcript);
    *sum = extrapolate(&round_poly, r);
    weights.fix_var_mut(r);
    Ok(r)
}

pub struct Sumcheck<F: Field, Ext: ExtensionField<F>> {
    pub sum: Ext,
    pub rs: Point<Ext>,
    pub weights: Poly<Ext>,
    pub poly: Poly<Ext>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> Sumcheck<F, Ext> {
    pub fn k(&self) -> usize {
        self.poly.k()
    }

    #[tracing::instrument(skip_all, fields(k = poly.k(), d = d, eq_base, eq_ext = eq_claims_ext.len(), pow_ext = pow_claims_ext.len()))]
    pub fn new<Transcript>(
        transcript: &mut Transcript,
        d: usize,
        alpha: Ext,

        eq_claims_ext: &[EqClaim<Ext, Ext>],
        pow_claims_ext: &[PowClaim<Ext, Ext>],

        poly: &Poly<F>,
    ) -> Result<Self, crate::Error>
    where
        F: ExtensionField<F>,
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let k = poly.k();
        assert!(d <= k);
        assert!(d > 0);

        let mut weights: Poly<Ext> = Poly::zero(k);
        let mut sum = Ext::ZERO;
        compress_claims(
            (&mut weights, &mut sum, false),
            alpha,
            &[],
            eq_claims_ext,
            &[],
            pow_claims_ext,
        );

        // first round
        let r = round(transcript, &mut sum, poly, &mut weights)?;
        // fix the first variable and get fixed new polynomial in extension field
        let mut poly = poly.fix_var(r);
        let mut rs: Point<Ext> = vec![r].into();

        // debug if we applied round correctly
        debug_assert_eq!(
            sum,
            dot_product(weights.iter().cloned(), poly.iter().cloned())
        );

        // rest
        (1..d).try_for_each(|_| {
            let r = round::<_, F, Ext, Ext>(transcript, &mut sum, &poly, &mut weights)?;
            // fix the next variable using same space
            poly.fix_var_mut(r);
            rs.push(r);

            // debug if we applied round correctly
            debug_assert_eq!(
                sum,
                dot_product(weights.iter().cloned(), poly.iter().cloned())
            );

            Ok(())
        })?;

        Ok(Self {
            sum,
            rs,
            weights,
            poly,
            _marker: std::marker::PhantomData,
        })
    }

    #[tracing::instrument(skip_all, fields(k = self.k(), d = d, eq_base = eq_claims_base.len(),eq_ext = eq_claims_ext.len(),pow_base = pow_claims_base.len(),pow_ext = pow_claims_ext.len()))]
    #[allow(clippy::too_many_arguments)]
    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        alpha: Ext,
        eq_claims_base: &[EqClaim<F, Ext>],
        eq_claims_ext: &[EqClaim<Ext, Ext>],
        pow_claims_base: &[PowClaim<F, Ext>],
        pow_claims_ext: &[PowClaim<Ext, Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        compress_claims(
            (&mut self.weights, &mut self.sum, true),
            alpha,
            eq_claims_base,
            eq_claims_ext,
            pow_claims_base,
            pow_claims_ext,
        );

        // run sumcheck rounds
        let rs = (0..d)
            .map(|_| {
                let r = round::<_, F, Ext, Ext>(
                    transcript,
                    &mut self.sum,
                    &self.poly,
                    &mut self.weights,
                )?;
                self.poly.fix_var_mut(r);
                Ok(r)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // update round challenges
        self.rs.extend(rs.iter());

        // return round point
        Ok(rs.into())
    }
}

#[derive(Debug, Clone)]
struct MultiRound<F: Field, Ext: ExtensionField<F>> {
    points_base: Vec<Point<F>>,
    points_ext: Vec<Point<Ext>>,
    vars_base: Vec<Point<F>>,
    vars_ext: Vec<Point<Ext>>,
    rs: Point<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> MultiRound<F, Ext> {
    fn new(
        points_base: Vec<Point<F>>,
        points_ext: Vec<Point<Ext>>,
        vars_base: Vec<Point<F>>,
        vars_ext: Vec<Point<Ext>>,
        rs: Point<Ext>,
        alpha: Ext,
    ) -> Self {
        Self {
            points_base,
            points_ext,
            vars_base,
            vars_ext,
            rs,
            alpha,
        }
    }

    fn extend(&mut self, rs: &Point<Ext>) {
        self.rs.extend(rs.iter());
    }

    fn weights(&self, poly: &Poly<Ext>) -> Ext {
        let rs = &self.rs.reversed();
        let eqs = self
            .points_base
            .iter()
            .map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);
                eval_eq_xy(&zs1, rs) * poly.eval(&zs0.as_ext::<Ext>())
            })
            .chain(self.vars_base.iter().map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);

                if off == 0 {
                    eval_pow_xy(&zs1, rs) * poly.constant().unwrap()
                } else {
                    eval_pow_xy(&zs1, rs) * poly.eval_univariate(Ext::from(*zs0.first().unwrap()))
                }
            }))
            .chain(self.points_ext.iter().map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);
                eval_eq_xy(&zs1, rs) * poly.eval(&zs0)
            }))
            .chain(self.vars_ext.iter().map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);

                if off == 0 {
                    eval_pow_xy(&zs1, rs) * poly.constant().unwrap()
                } else {
                    eval_pow_xy(&zs1, rs) * poly.eval_univariate(*zs0.first().unwrap())
                }
            }))
            .collect::<Vec<_>>();
        eqs.iter().horner_shifted(self.alpha, self.alpha)
    }
}

fn reduce<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
) -> Result<Ext, crate::Error>
where
    Transcript: Reader<Ext> + Challenge<F, Ext>,
{
    let v0: Ext = transcript.read()?;
    let v2 = transcript.read()?;
    let sumcheck_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&sumcheck_poly, r);
    Ok(r)
}

pub struct SumcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    pub k: usize,
    sum: Ext,
    multi_rounds: Vec<MultiRound<F, Ext>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            sum: Ext::ZERO,
            multi_rounds: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        alpha: Ext,
        eq_claims_base: &[EqClaim<F, Ext>],
        eq_claims_ext: &[EqClaim<Ext, Ext>],
        pow_claims_base: &[PowClaim<F, Ext>],
        pow_claims_ext: &[PowClaim<Ext, Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        eq_claims_ext.iter().for_each(|o| assert_eq!(o.k(), self.k));
        eq_claims_base
            .iter()
            .for_each(|o| assert_eq!(o.k(), self.k));

        // update sum
        self.sum = std::iter::once(&self.sum)
            .chain(eq_claims_base.iter().map(EqClaim::<F, Ext>::eval))
            .chain(pow_claims_base.iter().map(PowClaim::<F, Ext>::eval))
            .chain(eq_claims_ext.iter().map(EqClaim::<Ext, Ext>::eval))
            .chain(pow_claims_ext.iter().map(PowClaim::<Ext, Ext>::eval))
            .horner(alpha);

        // run sumcheck rounds
        let round_rs: Point<Ext> = (0..d)
            .map(|_| reduce::<_, F, Ext>(transcript, &mut self.sum))
            .collect::<Result<Vec<_>, _>>()?
            .into();

        // update previous multi rounds
        self.multi_rounds
            .iter_mut()
            .for_each(|round| round.extend(&round_rs));

        // add current multi rounds
        let eq_points_base = eq_claims_base
            .iter()
            .map(EqClaim::point)
            .collect::<Vec<_>>();
        let eq_points_ext = eq_claims_ext.iter().map(EqClaim::point).collect::<Vec<_>>();
        let pow_points_base = pow_claims_base
            .iter()
            .map(PowClaim::point)
            .collect::<Vec<_>>();
        let pow_points_ext = pow_claims_ext
            .iter()
            .map(PowClaim::point)
            .collect::<Vec<_>>();

        self.multi_rounds.push(MultiRound::new(
            eq_points_base,
            eq_points_ext,
            pow_points_base,
            pow_points_ext,
            round_rs.clone(),
            alpha,
        ));

        self.k -= d;

        Ok(round_rs)
    }

    pub fn finalize<Transcript>(
        self,
        transcript: &mut Transcript,
        poly: Option<Poly<Ext>>,
    ) -> Result<(), crate::Error>
    where
        Transcript: Reader<Ext>,
    {
        let poly = poly.map_or_else(|| Ok(Poly::new(transcript.read_many(1 << self.k)?)), Ok)?;
        let weigths = self
            .multi_rounds
            .iter()
            .map(|round| round.weights(&poly))
            .sum::<Ext>();
        (self.sum == weigths)
            .then_some(())
            .ok_or(crate::Error::Verify)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        pcs::{
            test::{
                get_eq_claims_base, get_eq_claims_ext, get_pow_claims_base, get_pow_claims_ext,
                make_eq_claims_base, make_eq_claims_ext, make_pow_claims_base, make_pow_claims_ext,
            },
            EqClaim, PowClaim,
        },
        poly::Poly,
        transcript::{
            rust_crypto::{RustCryptoReader, RustCryptoWriter},
            Challenge, Writer,
        },
    };
    use p3_field::extension::BinomialExtensionField;
    use rand::Rng;

    #[test]
    fn test_sumcheck() {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;
        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        {
            let k = 5;

            let mut rng = crate::test::rng(1);
            let poly: super::Poly<F> = Poly::rand(&mut rng, k);

            for d in 1..=5 {
                let (proof, checkpoint_prover) = {
                    let mut transcript = Writer::init("");

                    let eq_claims_ext =
                        make_eq_claims_ext::<_, F, F, Ext>(&mut transcript, 4, &poly).unwrap();
                    let pow_claims_ext =
                        make_pow_claims_ext::<_, F, F, Ext>(&mut transcript, 3, &poly).unwrap();

                    let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                    let sc = super::Sumcheck::<F, Ext>::new(
                        &mut transcript,
                        d,
                        alpha,
                        &eq_claims_ext[..],
                        &pow_claims_ext[..],
                        &poly,
                    )
                    .unwrap();

                    assert_eq!(sc.k(), k - d);

                    {
                        let points_ext =
                            eq_claims_ext.iter().map(EqClaim::point).collect::<Vec<_>>();
                        let vars_ext = pow_claims_ext
                            .iter()
                            .map(PowClaim::point)
                            .collect::<Vec<_>>();
                        let round = super::MultiRound::<F, Ext>::new(
                            vec![],
                            points_ext,
                            vec![],
                            vars_ext,
                            sc.rs.clone(),
                            alpha,
                        );
                        assert_eq!(round.weights(&sc.poly), sc.sum);
                    }

                    transcript.write_many(&sc.poly).unwrap();
                    let checkpoint: F = transcript.draw();
                    let proof = transcript.finalize();
                    (proof, checkpoint)
                };

                let checkpoint_verifier = {
                    let mut transcript = Reader::init(&proof, "");
                    let eq_claims_ext =
                        get_eq_claims_ext::<_, F, Ext>(&mut transcript, k, 4).unwrap();
                    let pow_claims_ext =
                        get_pow_claims_ext::<_, F, Ext>(&mut transcript, k, 3).unwrap();
                    let mut verifier = super::SumcheckVerifier::<F, Ext>::new(k);
                    let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                    verifier
                        .fold(
                            &mut transcript,
                            d,
                            alpha,
                            &[],
                            &eq_claims_ext,
                            &[],
                            &pow_claims_ext,
                        )
                        .unwrap();
                    verifier.finalize(&mut transcript, None).unwrap();
                    transcript.draw()
                };
                assert_eq!(checkpoint_prover, checkpoint_verifier);
            }
        }

        {
            let mut rng = crate::test::rng(1);
            for n_fold in 2..4 {
                for _ in 0..100 {
                    let n_base_eqs = (0..n_fold)
                        .map(|i| if i == 0 { 0 } else { rng.random_range(1..4) })
                        .collect::<Vec<usize>>();

                    let n_ext_eqs = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();

                    let n_base_pows = (0..n_fold)
                        .map(|i| if i == 0 { 0 } else { rng.random_range(1..4) })
                        .collect::<Vec<usize>>();

                    let n_ext_pows = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();

                    let ds = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();

                    let k = ds.iter().sum::<usize>();

                    let poly: super::Poly<F> = Poly::rand(&mut rng, k);
                    let (proof, checkpoint_prover) = {
                        let mut transcript = Writer::init("");

                        let mut k_folding = k;
                        let mut round = 0;

                        let (mut sc, mut rounds) = {
                            let n_eq_ext = n_ext_eqs[0];
                            let n_pow_ext = n_ext_pows[0];
                            let d = ds[0];

                            let eq_claims_ext =
                                make_eq_claims_ext(&mut transcript, n_eq_ext, &poly).unwrap();

                            let pow_claims_ext =
                                make_pow_claims_ext(&mut transcript, n_pow_ext, &poly).unwrap();

                            k_folding -= d;
                            round += d;

                            let alpha = Challenge::<F, Ext>::draw(&mut transcript);

                            let sc = super::Sumcheck::new(
                                &mut transcript,
                                d,
                                alpha,
                                &eq_claims_ext,
                                &pow_claims_ext,
                                &poly,
                            )
                            .unwrap();

                            assert_eq!(sc.k(), k_folding);
                            assert_eq!(k - sc.k(), round);

                            let points_ext =
                                eq_claims_ext.iter().map(EqClaim::point).collect::<Vec<_>>();
                            let var_ext = pow_claims_ext
                                .iter()
                                .map(PowClaim::point)
                                .collect::<Vec<_>>();
                            let round = super::MultiRound::<F, Ext>::new(
                                vec![],
                                points_ext,
                                vec![],
                                var_ext,
                                sc.rs.clone(),
                                alpha,
                            );

                            assert_eq!(round.weights(&sc.poly), sc.sum);
                            (sc, vec![round])
                        };

                        n_ext_eqs
                            .iter()
                            .zip(n_base_eqs.iter())
                            .zip(n_ext_pows.iter())
                            .zip(n_base_pows.iter())
                            .zip(ds.iter())
                            .skip(1)
                            .for_each(
                                |((((&n_eq_ext, &n_eq_base), &n_pow_ext), &n_pow_base), &d)| {
                                    k_folding -= d;
                                    round += d;

                                    let eq_claims_base = make_eq_claims_base::<_, F, Ext>(
                                        &mut transcript,
                                        n_eq_base,
                                        &sc.poly,
                                    )
                                    .unwrap();

                                    let eq_claims_ext = make_eq_claims_ext::<_, F, Ext, Ext>(
                                        &mut transcript,
                                        n_eq_ext,
                                        &sc.poly,
                                    )
                                    .unwrap();

                                    let pow_claims_base = make_pow_claims_base::<_, F, Ext>(
                                        &mut transcript,
                                        n_pow_base,
                                        &sc.poly,
                                    )
                                    .unwrap();

                                    let pow_claims_ext = make_pow_claims_ext::<_, F, Ext, Ext>(
                                        &mut transcript,
                                        n_pow_ext,
                                        &sc.poly,
                                    )
                                    .unwrap();

                                    let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                                    let rs = sc
                                        .fold(
                                            &mut transcript,
                                            d,
                                            alpha,
                                            &eq_claims_base,
                                            &eq_claims_ext,
                                            &pow_claims_base,
                                            &pow_claims_ext,
                                        )
                                        .unwrap();

                                    assert_eq!(sc.k(), k_folding);
                                    assert_eq!(k - sc.k(), round);

                                    {
                                        rounds.iter_mut().for_each(|round| round.extend(&rs));

                                        let points_base = eq_claims_base
                                            .iter()
                                            .map(EqClaim::point)
                                            .collect::<Vec<_>>();

                                        let points_ext = eq_claims_ext
                                            .iter()
                                            .map(EqClaim::point)
                                            .collect::<Vec<_>>();

                                        let vars_base = pow_claims_base
                                            .iter()
                                            .map(|claim| claim.point())
                                            .collect::<Vec<_>>();

                                        let vars_ext = pow_claims_ext
                                            .iter()
                                            .map(|claim| claim.point())
                                            .collect::<Vec<_>>();

                                        rounds.push(super::MultiRound::<F, Ext>::new(
                                            points_base,
                                            points_ext,
                                            vars_base,
                                            vars_ext,
                                            rs,
                                            alpha,
                                        ));
                                    }

                                    assert_eq!(
                                        rounds
                                            .iter()
                                            .map(|round| round.weights(&sc.poly))
                                            .sum::<Ext>(),
                                        sc.sum
                                    );
                                },
                            );

                        assert_eq!(round, k);
                        assert_eq!(k_folding, 0);

                        transcript.write_many(&sc.poly).unwrap();
                        let checkpoint: F = transcript.draw();
                        let proof = transcript.finalize();
                        (proof, checkpoint)
                    };

                    let checkpoint_verifier = {
                        let mut transcript = Reader::init(&proof, "");
                        let mut verifier = super::SumcheckVerifier::<F, Ext>::new(k);

                        n_ext_eqs
                            .iter()
                            .zip(n_base_eqs.iter())
                            .zip(n_ext_pows.iter())
                            .zip(n_base_pows.iter())
                            .zip(ds.iter())
                            .for_each(
                                |((((&n_eq_ext, &n_eq_base), &n_pow_ext), &n_pow_base), &d)| {
                                    let eq_claims_base: Vec<EqClaim<F, Ext>> =
                                        get_eq_claims_base::<_, F, Ext>(
                                            &mut transcript,
                                            verifier.k,
                                            n_eq_base,
                                        )
                                        .unwrap();

                                    let eq_claims_ext: Vec<EqClaim<Ext, Ext>> =
                                        get_eq_claims_ext::<_, F, Ext>(
                                            &mut transcript,
                                            verifier.k,
                                            n_eq_ext,
                                        )
                                        .unwrap();

                                    let pow_claims_base: Vec<PowClaim<F, Ext>> =
                                        get_pow_claims_base::<_, F, Ext>(
                                            &mut transcript,
                                            verifier.k,
                                            n_pow_base,
                                        )
                                        .unwrap();

                                    let pow_claims_ext: Vec<PowClaim<Ext, Ext>> =
                                        get_pow_claims_ext::<_, F, Ext>(
                                            &mut transcript,
                                            verifier.k,
                                            n_pow_ext,
                                        )
                                        .unwrap();

                                    let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                                    verifier
                                        .fold(
                                            &mut transcript,
                                            d,
                                            alpha,
                                            &eq_claims_base,
                                            &eq_claims_ext,
                                            &pow_claims_base,
                                            &pow_claims_ext,
                                        )
                                        .unwrap();
                                },
                            );

                        verifier.finalize(&mut transcript, None).unwrap();
                        let checkpoint: F = transcript.draw();
                        checkpoint
                    };
                    assert_eq!(checkpoint_prover, checkpoint_verifier);
                }
            }
        }
    }
}
