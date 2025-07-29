use p3_field::{ExtensionField, Field};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    pcs::Claim,
    poly::{Point, Poly},
    transcript::{Challenge, Reader, Writer},
    utils::VecOps,
};

fn extrapolate<F: Field, EF: ExtensionField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from_usize(i))
        .collect::<Vec<_>>();
    let mut poly = crate::utils::arithmetic::interpolate(&points, evals);
    poly.reverse();
    poly.horner(target)
}

fn eval_eq_xy<F: Field>(x: &[F], y: &[F]) -> F {
    assert_eq!(x.len(), y.len());
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| (xi * yi).double() - xi - yi + F::ONE)
        .product()
}

fn round_inner_ext<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
    poly: &Poly<F>,
    eq: &mut Poly<Ext>,
) -> Result<(Poly<Ext>, Ext), crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = eq.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip(phi.par_iter())
        .zip(elo.par_iter().zip(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ext::ZERO, Ext::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    transcript.write(v0)?;
    transcript.write(v2)?;
    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&round_poly, r);
    eq.fix_var(r);
    let poly = poly.fix_var_ext(r);
    Ok((poly, r))
}

fn round_inner<Transcript, F: Field, Ext: ExtensionField<F>>(
    transcript: &mut Transcript,
    sum: &mut Ext,
    poly: &mut Poly<Ext>,
    eq: &mut Poly<Ext>,
) -> Result<Ext, crate::Error>
where
    Transcript: Writer<Ext> + Challenge<F, Ext>,
{
    let mid = poly.len() / 2;
    let (plo, phi) = poly.split_at(mid);
    let (elo, ehi) = eq.split_at(mid);

    let (v0, v2) = plo
        .par_iter()
        .zip(phi.par_iter())
        .zip(elo.par_iter().zip(ehi.par_iter()))
        .map(|((&p0, &p1), (&e0, &e1))| (e0 * p0, (e1.double() - e0) * (p1.double() - p0)))
        .reduce(
            || (Ext::ZERO, Ext::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    transcript.write(v0)?;
    transcript.write(v2)?;
    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&round_poly, r);
    eq.fix_var(r);
    poly.fix_var(r);

    Ok(r)
}

pub struct Sumcheck<F: Field, Ext: ExtensionField<F>> {
    k: usize,
    ch: Ext,
    sum: Ext,
    rs: Point<Ext>,
    eq: Poly<Ext>,
    poly: Poly<Ext>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> Sumcheck<F, Ext> {
    pub fn k(&self) -> usize {
        self.poly.k()
    }

    pub fn round(&self) -> usize {
        self.k - self.poly.k()
    }

    pub fn poly(&self) -> &Poly<Ext> {
        &self.poly
    }

    pub fn sum(&self) -> Ext {
        self.sum
    }

    pub fn rs(&self) -> &Point<Ext> {
        &self.rs
    }

    pub fn write_poly<Transcript>(&self, transcript: &mut Transcript) -> Result<(), crate::Error>
    where
        Transcript: Writer<Ext>,
    {
        transcript.write_many(self.poly())
    }

    pub fn new<Transcript>(
        transcript: &mut Transcript,
        k: usize,
        d: usize,
        ch: Ext,
        claims: &[Claim<Ext>],
        poly: &Poly<F>,
    ) -> Result<Self, crate::Error>
    where
        F: ExtensionField<F>,
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        assert_eq!(poly.k(), k);
        assert!(claims.iter().all(|o| o.k() == k));
        assert!(d > 0);

        let mut sum = claims
            .iter()
            .map(|o| o.eval())
            .collect::<Vec<_>>()
            .horner(ch);

        let eqs = claims
            .iter()
            .map(|o| o.point().eq(Ext::ONE))
            .collect::<Vec<_>>();

        let mut eq: Poly<_> = (0..1 << k)
            .map(|row| eqs.iter().map(|eq| eq[row]).collect::<Vec<_>>().horner(ch))
            .collect::<Vec<_>>()
            .into();

        // first round
        let (mut poly, r) = round_inner_ext(transcript, &mut sum, poly, &mut eq)?;

        let mut rs: Point<Ext> = vec![r].into();

        #[cfg(debug_assertions)]
        assert_eq!(
            eq.iter()
                .zip(poly.iter())
                .fold(Ext::ZERO, |acc, (&c0, &c1)| acc + c0 * c1),
            sum
        );

        // rest
        (1..d).try_for_each(|_| {
            let r = round_inner(transcript, &mut sum, &mut poly, &mut eq)?;
            rs.push(r);

            #[cfg(debug_assertions)]
            assert_eq!(
                eq.iter()
                    .zip(poly.iter())
                    .fold(Ext::ZERO, |acc, (&c0, &c1)| acc + c0 * c1),
                sum
            );
            Ok(())
        })?;

        Ok(Self {
            k,
            ch,
            sum,
            rs,
            eq,
            poly,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        claims: &[Claim<Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        assert!(claims.iter().all(|o| o.k() == self.k()));

        self.sum = std::iter::once(self.sum)
            .chain(claims.iter().map(|o| o.eval()))
            .collect::<Vec<_>>()
            .horner(self.ch);

        let eqs = claims
            .iter()
            .map(|o| o.point().eq(Ext::ONE))
            .collect::<Vec<_>>();

        self.eq.iter_mut().enumerate().for_each(|(i, cur)| {
            *cur = std::iter::once(*cur)
                .chain(eqs.iter().map(|eq| eq[i]).collect::<Vec<_>>())
                .collect::<Vec<_>>()
                .horner(self.ch);
        });

        let rs = (0..d)
            .map(|_| round_inner(transcript, &mut self.sum, &mut self.poly, &mut self.eq))
            .collect::<Result<Vec<_>, _>>()?;

        self.rs.extend(rs.iter());

        Ok(rs.into())
    }
}

#[derive(Debug, Clone)]
struct MultiRound<F: Field> {
    points: Vec<Point<F>>,
    rs: Point<F>,
}

impl<F: Field> MultiRound<F> {
    fn new(points: Vec<Point<F>>, rs: &Point<F>) -> Self {
        Self {
            points,
            rs: rs.clone(),
        }
    }

    fn extend(&mut self, rs: &Point<F>) {
        self.rs.extend(rs.iter());
    }

    fn eqs(&self, poly: &Poly<F>) -> Vec<F> {
        self.points
            .iter()
            .map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - self.rs.len());
                let (zs0, zs1) = zs.split_at(off);
                eval_eq_xy(&zs1, &self.rs.reversed()) * poly.eval_lagrange(&zs0)
            })
            .collect::<Vec<_>>()
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
    let v = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ext>::draw(transcript);
    *sum = extrapolate(&v, r);
    Ok(r)
}

pub struct SumcheckVerifier<F: Field, Ext: ExtensionField<F>> {
    k: usize,
    sum: Ext,
    ch: Ext,
    multi_rounds: Vec<MultiRound<Ext>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field, Ext: ExtensionField<F>> SumcheckVerifier<F, Ext> {
    pub fn new(ch: Ext, k: usize) -> Self {
        Self {
            k,
            ch,
            sum: Ext::ZERO,
            multi_rounds: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        claims: &[Claim<Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        assert!(claims.iter().all(|o| { o.k() == self.k }));
        self.sum = std::iter::once(self.sum)
            .chain(claims.iter().map(|o| {
                assert_eq!(o.k(), self.k);
                o.eval()
            }))
            .collect::<Vec<_>>()
            .horner(self.ch);

        let round_rs: Point<Ext> = (0..d)
            .map(|_| {
                let r = reduce::<_, F, Ext>(transcript, &mut self.sum)?;

                Ok(r)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into();

        self.multi_rounds
            .iter_mut()
            .for_each(|round| round.extend(&round_rs));

        let points = claims
            .iter()
            .map(|o| o.point())
            .cloned()
            .collect::<Vec<_>>();

        self.multi_rounds.push(MultiRound::new(points, &round_rs));
        self.k -= d;

        Ok(round_rs)
    }

    pub fn read_poly<Transcript>(
        &self,
        transcript: &mut Transcript,
    ) -> Result<Poly<Ext>, crate::Error>
    where
        Transcript: Reader<Ext>,
    {
        Ok(transcript.read_many(1 << self.k)?.into())
    }

    pub fn finalize<Transcript>(self, transcript: &mut Transcript) -> Result<(), crate::Error>
    where
        Transcript: Reader<Ext>,
    {
        let poly: Poly<Ext> = self.read_poly(transcript)?;
        let eqs: Vec<_> = self
            .multi_rounds
            .iter()
            .flat_map(|round| round.eqs(&poly))
            .collect::<Vec<_>>();
        (self.sum == eqs.horner(self.ch))
            .then_some(())
            .ok_or(crate::Error::Verify)
    }
}

#[cfg(test)]
mod test {

    use crate::{
        pcs::{
            sumcheck::{eval_eq_xy, MultiRound, SumcheckVerifier},
            Claim,
        },
        poly::{Point, Poly},
        transcript::rust_crypto::{RustCryptoReader, RustCryptoWriter},
        transcript::{Challenge, Reader, Writer},
        utils::VecOps,
    };
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};
    use rand::Rng;

    // fn make_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
    //     transcript: &mut Transcript,
    //     poly: &Poly<F>,
    //     num_points: usize,
    // ) -> Vec<Claim<Ext>>
    // where
    //     Transcript: Challenge<F, Ext> + Writer<Ext>,
    // {
    //     (0..num_points)
    //         .map(|_| {
    //             let point = Point::expand(poly.k(), transcript.draw());
    //             let claim = Claim::evaluate(poly, &point);
    //             transcript.write(claim.eval).unwrap();
    //             claim
    //         })
    //         .collect::<Vec<_>>()
    // }

    fn make_claims<
        Transcript,
        F: Field,
        Ex0: ExtensionField<F>,
        Ex1: ExtensionField<Ex0> + ExtensionField<F>,
    >(
        transcript: &mut Transcript,
        poly: &Poly<Ex0>,
        num_points: usize,
    ) -> Vec<Claim<Ex1>>
    where
        Transcript: Challenge<F, Ex1> + Writer<Ex1>,
    {
        (0..num_points)
            .map(|_| {
                let point = Point::expand(poly.k(), transcript.draw());
                let claim = Claim::evaluate(poly, &point);
                transcript.write(claim.eval).unwrap();
                claim
            })
            .collect::<Vec<_>>()
    }

    fn get_claims<Transcript, F: Field, Ext: ExtensionField<F>>(
        transcript: &mut Transcript,
        k: usize,
        n_points: usize,
    ) -> Vec<Claim<Ext>>
    where
        Transcript: Challenge<F, Ext> + Reader<Ext>,
    {
        (0..n_points)
            .map(|_| {
                let point = Point::expand(k, transcript.draw());
                let eval = transcript.read().unwrap();
                Claim::new(point, eval)
            })
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_sumcheck() {
        type F = p3_koala_bear::KoalaBear;
        type Ext = BinomialExtensionField<F, 4>;

        type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
        type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

        {
            let k = 5;
            let d = 5;
            let mut rng = crate::test::rng(1);

            let poly: Poly<F> = Poly::rand(&mut rng, k);
            let n_points = 1;
            let domain = "";

            let (proof, checkpoint_prover) = {
                let mut transcript = Writer::init(domain);
                let claims = make_claims::<_, F, F, Ext>(&mut transcript, &poly, n_points);
                let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                let sc =
                    super::Sumcheck::<F, Ext>::new(&mut transcript, k, d, alpha, &claims, &poly)
                        .unwrap();

                assert_eq!(sc.k(), k - d);
                {
                    let points = claims
                        .iter()
                        .map(|o| o.point())
                        .cloned()
                        .collect::<Vec<_>>();

                    let round = MultiRound::new(points, sc.rs());
                    let eqs = round.eqs(sc.poly());
                    assert_eq!(eqs.horner(alpha), sc.sum());
                }

                {
                    let z = claims[0].point();
                    let r = sc.rs().reversed();
                    assert_eq!(poly.eval_lagrange(&r), sc.poly.constant().unwrap());
                    assert_eq!(sc.sum(), eval_eq_xy(z, &r) * poly.eval_lagrange(&r));
                    assert_eq!(eval_eq_xy(z, &r), sc.eq.constant().unwrap());
                }

                sc.write_poly(&mut transcript).unwrap();
                let checkpoint: F = transcript.draw();
                let proof = transcript.finalize();
                (proof, checkpoint)
            };

            let checkpoint_verifier = {
                let mut transcript = Reader::init(&proof, "");
                let claims = get_claims::<_, F, Ext>(&mut transcript, k, n_points);
                let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                let mut verifier = SumcheckVerifier::<F, Ext>::new(alpha, k);
                verifier.fold(&mut transcript, d, &claims).unwrap();
                verifier.finalize(&mut transcript).unwrap();
                transcript.draw()
            };
            assert_eq!(checkpoint_prover, checkpoint_verifier);
        }

        {
            let mut rng = crate::test::rng(1);
            for n_fold in 1..4 {
                for _ in 0..100 {
                    let n_points = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();
                    let ds = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();
                    let k = ds.iter().sum::<usize>();

                    let poly: Poly<F> = Poly::rand(&mut rng, k);
                    let (proof, checkpoint_prover) = {
                        let mut transcript = Writer::init("");
                        let alpha = Challenge::<F, Ext>::draw(&mut transcript);

                        let mut k_folding = k;
                        let mut round = 0;

                        let (mut sc, mut rounds) = {
                            let n = n_points[0];
                            let d = ds[0];

                            let claims = make_claims(&mut transcript, &poly, n);

                            k_folding -= d;
                            round += d;

                            let sc =
                                super::Sumcheck::new(&mut transcript, k, d, alpha, &claims, &poly)
                                    .unwrap();

                            assert_eq!(sc.k(), k_folding);
                            assert_eq!(sc.round(), round);

                            let points =
                                claims.iter().map(Claim::point).cloned().collect::<Vec<_>>();
                            let round = MultiRound::new(points, sc.rs());

                            assert_eq!(round.eqs(sc.poly()).horner(alpha), sc.sum());
                            (sc, vec![round])
                        };

                        n_points
                            .iter()
                            .zip(ds.iter())
                            .skip(1)
                            .for_each(|(&num_points, &d)| {
                                k_folding -= d;
                                round += d;

                                let claims = make_claims::<_, F, Ext, Ext>(
                                    &mut transcript,
                                    &sc.poly,
                                    num_points,
                                );

                                let rs = sc.fold(&mut transcript, d, &claims).unwrap();

                                assert_eq!(sc.k(), k_folding);
                                assert_eq!(sc.round(), round);

                                {
                                    rounds.iter_mut().for_each(|round| round.extend(&rs));
                                    let points = claims
                                        .iter()
                                        .map(Claim::point)
                                        .cloned()
                                        .collect::<Vec<_>>();
                                    rounds.push(MultiRound::new(points, &rs));
                                }

                                let eqs: Vec<_> = rounds
                                    .iter()
                                    .flat_map(|round| round.eqs(sc.poly()))
                                    .collect::<Vec<_>>();

                                assert_eq!(eqs.horner(alpha), sc.sum());
                            });

                        assert_eq!(round, k);
                        assert_eq!(k_folding, 0);

                        sc.write_poly(&mut transcript).unwrap();
                        let checkpoint: F = transcript.draw();
                        let proof = transcript.finalize();
                        (proof, checkpoint)
                    };

                    let checkpoint_verifier = {
                        let mut transcript = Reader::init(&proof, "");
                        let alpha = Challenge::<F, Ext>::draw(&mut transcript);

                        let mut verifier = SumcheckVerifier::<F, Ext>::new(alpha, k);

                        n_points.iter().zip(ds.iter()).for_each(|(&n, &d)| {
                            let claims: Vec<Claim<Ext>> =
                                get_claims::<_, F, Ext>(&mut transcript, verifier.k, n);

                            verifier.fold(&mut transcript, d, &claims).unwrap();
                        });

                        verifier.finalize(&mut transcript).unwrap();
                        let checkpoint: F = transcript.draw();
                        checkpoint
                    };
                    assert_eq!(checkpoint_prover, checkpoint_verifier);
                }
            }
        }
    }
}
