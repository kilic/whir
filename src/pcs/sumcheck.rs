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
    crate::utils::arithmetic::interpolate(&points, evals)
        .iter()
        .horner(target)
}

fn eval_eq_xy<F: Field, Ext: ExtensionField<F>>(x: &[F], y: &[Ext]) -> Ext {
    assert_eq!(x.len(), y.len());
    x.par_iter()
        .zip(y.par_iter())
        .map(|(&xi, &yi)| (yi * xi).double() - xi - yi + F::ONE)
        .product()
}

fn round<
    Transcript,
    F: Field,
    Ex0: ExtensionField<F>,
    Ex1: ExtensionField<Ex0> + ExtensionField<F>,
>(
    transcript: &mut Transcript,
    sum: &mut Ex1,
    poly: &Poly<Ex0>,
    eq: &mut Poly<Ex1>,
) -> Result<Ex1, crate::Error>
where
    Transcript: Writer<Ex1> + Challenge<F, Ex1>,
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
            || (Ex1::ZERO, Ex1::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
        );

    transcript.write(v0)?;
    transcript.write(v2)?;
    let round_poly = vec![v0, *sum - v0, v2];
    let r = Challenge::<F, Ex1>::draw(transcript);
    *sum = extrapolate(&round_poly, r);
    eq.fix_var(r);
    Ok(r)
}

pub struct Sumcheck<F: Field, Ext: ExtensionField<F>> {
    k: usize,
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

    pub fn new<Transcript>(
        transcript: &mut Transcript,
        d: usize,
        alpha: Ext,

        claims_ext: &[Claim<Ext, Ext>],
        poly: &Poly<F>,
    ) -> Result<Self, crate::Error>
    where
        F: ExtensionField<F>,
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        let k = poly.k();
        assert!(claims_ext.iter().all(|o| o.k() == poly.k()));
        assert!(k >= d);

        let mut sum = claims_ext
            .iter()
            .map(Claim::<Ext, Ext>::eval)
            .horner_shifted(alpha, alpha);

        let eqs_ext = claims_ext.iter().map(Claim::eq).collect::<Vec<_>>();

        let mut eq: Poly<_> = (0..1 << k)
            .map(|i| eqs_ext.iter().map(|eq| &eq[i]).horner_shifted(alpha, alpha))
            .collect::<Vec<_>>()
            .into();

        // first round
        let r = round(transcript, &mut sum, poly, &mut eq)?;
        let mut poly = poly.fix_var_ext(r);
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
            let r = round::<_, F, Ext, Ext>(transcript, &mut sum, &poly, &mut eq)?;
            poly.fix_var(r);
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
        alpha: Ext,
        claims_base: &[Claim<F, Ext>],
        claims_ext: &[Claim<Ext, Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Writer<F> + Writer<Ext> + Challenge<F, Ext>,
    {
        assert!(claims_base.iter().all(|o| o.k() == self.k()));
        assert!(claims_ext.iter().all(|o| o.k() == self.k()));

        self.sum = std::iter::once(&self.sum)
            .chain(claims_base.iter().map(Claim::<F, Ext>::eval))
            .chain(claims_ext.iter().map(Claim::<Ext, Ext>::eval))
            .horner(alpha);

        let eqs_base = claims_base.iter().map(Claim::eq).collect::<Vec<_>>();
        let eqs_ext = claims_ext.iter().map(Claim::eq).collect::<Vec<_>>();

        let alpha_shift = alpha.exp_u64(eqs_base.len() as u64 + 1);
        self.eq.iter_mut().enumerate().for_each(|(i, cur)| {
            *cur += eqs_base
                .iter()
                .map(|eq| &eq[i])
                .horner_shifted(alpha, alpha)
                + eqs_ext
                    .iter()
                    .map(|eq| &eq[i])
                    .horner_shifted(alpha, alpha_shift)
        });

        let rs = (0..d)
            .map(|_| {
                let r =
                    round::<_, F, Ext, Ext>(transcript, &mut self.sum, &self.poly, &mut self.eq)?;
                self.poly.fix_var(r);
                Ok(r)
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.rs.extend(rs.iter());

        Ok(rs.into())
    }
}

#[derive(Debug, Clone)]
struct MultiRound<F: Field, Ext: ExtensionField<F>> {
    points_base: Vec<Point<F>>,
    points_ext: Vec<Point<Ext>>,
    rs: Point<Ext>,
    alpha: Ext,
}

impl<F: Field, Ext: ExtensionField<F>> MultiRound<F, Ext> {
    fn new(
        points_base: Vec<Point<F>>,
        points_ext: Vec<Point<Ext>>,
        rs: Point<Ext>,
        alpha: Ext,
    ) -> Self {
        Self {
            points_base,
            points_ext,
            rs,
            alpha,
        }
    }

    fn extend(&mut self, rs: &Point<Ext>) {
        self.rs.extend(rs.iter());
    }

    fn eqs(&self, poly: &Poly<Ext>) -> Ext {
        let rs = &self.rs.reversed();
        let eqs = self
            .points_base
            .iter()
            .map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);
                eval_eq_xy(&zs1, rs) * poly.eval_lagrange(&zs0.as_ext::<Ext>())
            })
            .chain(self.points_ext.iter().map(|zs| {
                let off = poly.k();
                assert_eq!(off, zs.len() - rs.len());
                let (zs0, zs1) = zs.split_at(off);
                eval_eq_xy(&zs1, rs) * poly.eval_lagrange(&zs0)
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
    k: usize,
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

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn fold<Transcript>(
        &mut self,
        transcript: &mut Transcript,
        d: usize,
        alpha: Ext,
        claims_base: &[Claim<F, Ext>],
        claims_ext: &[Claim<Ext, Ext>],
    ) -> Result<Point<Ext>, crate::Error>
    where
        Transcript: Reader<Ext> + Challenge<F, Ext>,
    {
        assert!(claims_base.iter().all(|o| { o.k() == self.k }));
        assert!(claims_ext.iter().all(|o| o.k() == self.k));

        // update sum
        self.sum = std::iter::once(&self.sum)
            .chain(claims_base.iter().map(Claim::<F, Ext>::eval))
            .chain(claims_ext.iter().map(Claim::<Ext, Ext>::eval))
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
        let points_base = claims_base.iter().map(Claim::point).collect::<Vec<_>>();
        let points_ext = claims_ext.iter().map(Claim::point).collect::<Vec<_>>();
        self.multi_rounds.push(MultiRound::new(
            points_base,
            points_ext,
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
        let eqs = self
            .multi_rounds
            .iter()
            .map(|round| round.eqs(&poly))
            .sum::<Ext>();
        (self.sum == eqs).then_some(()).ok_or(crate::Error::Verify)
    }
}

#[cfg(test)]
mod test {

    use crate::{
        pcs::{
            sumcheck::{eval_eq_xy, MultiRound, SumcheckVerifier},
            test::{get_claims_base, get_claims_ext, make_claims_base, make_claims_ext},
            Claim,
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
            let d = 5;
            let mut rng = crate::test::rng(1);

            let poly: Poly<F> = Poly::rand(&mut rng, k);
            let n_points = 1;
            let domain = "";

            let (proof, checkpoint_prover) = {
                let mut transcript = Writer::init(domain);

                let claims =
                    make_claims_ext::<_, F, F, Ext>(&mut transcript, n_points, &poly).unwrap();

                let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                let sc = super::Sumcheck::<F, Ext>::new(&mut transcript, d, alpha, &claims, &poly)
                    .unwrap();

                assert_eq!(sc.k(), k - d);
                {
                    let points = claims.iter().map(Claim::point).collect::<Vec<_>>();
                    let round = MultiRound::new(points, vec![], sc.rs().clone(), alpha);
                    assert_eq!(round.eqs(sc.poly()), sc.sum());
                }

                {
                    let z = &claims[0].point;
                    let r = sc.rs().reversed();
                    assert_eq!(poly.eval_lagrange(&r), sc.poly.constant().unwrap());
                    assert_eq!(sc.sum(), alpha * eval_eq_xy(z, &r) * poly.eval_lagrange(&r));
                    assert_eq!(alpha * eval_eq_xy(z, &r), sc.eq.constant().unwrap());
                }

                transcript.write_many(sc.poly()).unwrap();
                let checkpoint: F = transcript.draw();
                let proof = transcript.finalize();
                (proof, checkpoint)
            };

            let checkpoint_verifier = {
                let mut transcript = Reader::init(&proof, "");
                let claims = get_claims_ext::<_, F, Ext>(&mut transcript, k, n_points).unwrap();
                let mut verifier = SumcheckVerifier::<F, Ext>::new(k);
                let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                verifier
                    .fold(&mut transcript, d, alpha, &[], &claims)
                    .unwrap();
                verifier.finalize(&mut transcript, None).unwrap();
                transcript.draw()
            };
            assert_eq!(checkpoint_prover, checkpoint_verifier);
        }

        {
            let mut rng = crate::test::rng(1);
            for n_fold in 1..4 {
                for _ in 0..100 {
                    let n_ext_points = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();

                    let n_base_points = (0..n_fold)
                        .map(|i| if i == 0 { 0 } else { rng.random_range(1..4) })
                        .collect::<Vec<usize>>();

                    let ds = (0..n_fold)
                        .map(|_| rng.random_range(1..4))
                        .collect::<Vec<usize>>();

                    let k = ds.iter().sum::<usize>();

                    let poly: Poly<F> = Poly::rand(&mut rng, k);
                    let (proof, checkpoint_prover) = {
                        let mut transcript = Writer::init("");

                        let mut k_folding = k;
                        let mut round = 0;

                        let (mut sc, mut rounds) = {
                            let n_ext_points = n_ext_points[0];
                            let d = ds[0];

                            let claims =
                                make_claims_ext(&mut transcript, n_ext_points, &poly).unwrap();

                            k_folding -= d;
                            round += d;

                            let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                            let sc =
                                super::Sumcheck::new(&mut transcript, d, alpha, &claims, &poly)
                                    .unwrap();

                            assert_eq!(sc.k(), k_folding);
                            assert_eq!(sc.round(), round);

                            let points = claims.iter().map(Claim::point).collect::<Vec<_>>();
                            let round =
                                MultiRound::<F, Ext>::new(vec![], points, sc.rs().clone(), alpha);

                            assert_eq!(round.eqs(sc.poly()), sc.sum());
                            (sc, vec![round])
                        };

                        n_ext_points
                            .iter()
                            .zip(n_base_points.iter())
                            .zip(ds.iter())
                            .skip(1)
                            .for_each(|((&n_ext_points, &n_base_points), &d)| {
                                k_folding -= d;
                                round += d;

                                let claims_base = make_claims_base::<_, F, Ext>(
                                    &mut transcript,
                                    n_base_points,
                                    &sc.poly,
                                )
                                .unwrap();

                                let claims_ext = make_claims_ext::<_, F, Ext, Ext>(
                                    &mut transcript,
                                    n_ext_points,
                                    &sc.poly,
                                )
                                .unwrap();

                                let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                                let rs = sc
                                    .fold(&mut transcript, d, alpha, &claims_base, &claims_ext)
                                    .unwrap();

                                assert_eq!(sc.k(), k_folding);
                                assert_eq!(sc.round(), round);

                                {
                                    rounds.iter_mut().for_each(|round| round.extend(&rs));

                                    let points_base =
                                        claims_base.iter().map(Claim::point).collect::<Vec<_>>();
                                    let points_ext =
                                        claims_ext.iter().map(Claim::point).collect::<Vec<_>>();
                                    rounds.push(MultiRound::<F, Ext>::new(
                                        points_base,
                                        points_ext,
                                        rs,
                                        alpha,
                                    ));
                                }

                                assert_eq!(
                                    rounds.iter().map(|round| round.eqs(sc.poly())).sum::<Ext>(),
                                    sc.sum()
                                );
                            });

                        assert_eq!(round, k);
                        assert_eq!(k_folding, 0);

                        transcript.write_many(sc.poly()).unwrap();
                        let checkpoint: F = transcript.draw();
                        let proof = transcript.finalize();
                        (proof, checkpoint)
                    };

                    let checkpoint_verifier = {
                        let mut transcript = Reader::init(&proof, "");
                        let mut verifier = SumcheckVerifier::<F, Ext>::new(k);

                        n_ext_points
                            .iter()
                            .zip(n_base_points.iter())
                            .zip(ds.iter())
                            .for_each(|((&n_ext_points, &n_base_points), &d)| {
                                let claims_base: Vec<Claim<F, Ext>> = get_claims_base::<_, F, Ext>(
                                    &mut transcript,
                                    verifier.k,
                                    n_base_points,
                                )
                                .unwrap();

                                let claims_ext: Vec<Claim<Ext, Ext>> = get_claims_ext::<_, F, Ext>(
                                    &mut transcript,
                                    verifier.k,
                                    n_ext_points,
                                )
                                .unwrap();

                                let alpha = Challenge::<F, Ext>::draw(&mut transcript);
                                verifier
                                    .fold(&mut transcript, d, alpha, &claims_base, &claims_ext)
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
    }
}
