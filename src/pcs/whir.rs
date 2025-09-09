use itertools::Itertools;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use transpose::transpose;

use crate::{
    merkle::comm::{Commitment, CommitmentData},
    pcs::{
        params::{compute_number_of_rounds, SecurityAssumption},
        sumcheck::{Sumcheck, SumcheckVerifier},
        Claim,
    },
    poly::{Coeff, Point, Poly},
    transcript::{Challenge, ChallengeBits, Reader, Writer},
    utils::{unsafe_allocate_zero_vec, TwoAdicSlice},
};

pub fn commit_base<Transcript, F: TwoAdicField, MCom: Commitment<F>>(
    transcript: &mut Transcript,
    poly: &[F],
    rate: usize,
    folding: usize,
    mat_comm: &MCom,
) -> Result<CommitmentData<F, F, MCom::Digest>, crate::Error>
where
    Transcript: Writer<MCom::Digest>,
{
    // pad
    let mut padded: Vec<F> = crate::utils::unsafe_allocate_zero_vec(1 << (poly.k() + rate));
    padded[..poly.len()].copy_from_slice(poly);

    // encode and commit
    let mat = RowMajorMatrix::new(padded, 1 << folding);
    let codeword = tracing::info_span!("dft-base", height = mat.height(), width = mat.width())
        .in_scope(|| Radix2DFTSmallBatch::<F>::default().dft_batch(mat));
    mat_comm.commit_base(transcript, codeword)
}

pub fn commit_ext<
    Transcript,
    F: TwoAdicField,
    Ext: ExtensionField<F> + TwoAdicField,
    MCom: Commitment<F>,
>(
    transcript: &mut Transcript,
    poly: &[Ext],
    rate: usize,
    folding: usize,
    mat_comm: &MCom,
) -> Result<CommitmentData<F, Ext, MCom::Digest>, crate::Error>
where
    Transcript: Writer<MCom::Digest>,
{
    // pad
    let mut padded: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec(1 << (poly.k() + rate));
    padded[..poly.len()].copy_from_slice(poly);

    // encode and commit
    let mat = RowMajorMatrix::new(padded, 1 << folding);
    let codeword = tracing::info_span!("dft-ext", h = mat.height(), w = mat.width())
        .in_scope(|| Radix2DFTSmallBatch::<F>::default().dft_algebra_batch(mat));
    mat_comm.commit_ext(transcript, codeword)
}

fn stir_indicies<Transcript>(
    transcript: &mut Transcript,
    bits: usize,
    n_queries: usize,
) -> Vec<usize>
where
    Transcript: ChallengeBits,
{
    (0..n_queries)
        .map(|_| ChallengeBits::draw(transcript, bits))
        .sorted()
        .dedup()
        .collect::<Vec<_>>()
}

pub struct Whir<
    F: TwoAdicField,
    Ext: ExtensionField<F>,
    MCom: Commitment<F>,
    MComExt: Commitment<F>,
> {
    pub k: usize,
    pub folding: usize,
    pub rate: usize,
    pub initial_reduction: usize,
    soundness: SecurityAssumption,
    security_level: usize,
    comm: MCom,
    comm_ext: MComExt,
    _marker: std::marker::PhantomData<(F, Ext)>,
}

impl<
        F: TwoAdicField,
        Ext: ExtensionField<F> + TwoAdicField,
        MCom: Commitment<F>,
        MComExt: Commitment<F>,
    > Whir<F, Ext, MCom, MComExt>
{
    pub fn new(
        k: usize,
        folding: usize,
        rate: usize,
        initial_reduction: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        comm: MCom,
        comm_ext: MComExt,
    ) -> Self {
        Self {
            k,
            folding,
            rate,
            initial_reduction,
            soundness,
            security_level,
            comm,
            comm_ext,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn commit<Transcript>(
        &self,
        transcript: &mut Transcript,
        poly: &Poly<F, Coeff>,
    ) -> Result<CommitmentData<F, F, MCom::Digest>, crate::Error>
    where
        Transcript: Writer<Ext> + Writer<MCom::Digest> + Challenge<F, Ext>,
    {
        let mut transposed = unsafe_allocate_zero_vec(poly.len());
        let width = 1 << self.folding;
        tracing::info_span!("transpose")
            .in_scope(|| transpose::transpose(poly, &mut transposed, poly.len() / width, width));
        commit_base(transcript, &transposed, self.rate, self.folding, &self.comm)
    }

    fn n_ood_queries(&self, k: usize, rate: usize) -> usize {
        self.soundness
            .n_ood_queries(self.security_level, k, rate, Ext::bits())
    }

    fn n_stir_queries(&self, rate: usize) -> usize {
        self.soundness.n_stir_queries(self.security_level, rate)
    }

    fn round_params(&self, round: usize) -> (usize, usize, usize) {
        let prev_reduction = std::iter::once(self.initial_reduction)
            .chain(std::iter::repeat(1))
            .take(round)
            .sum::<usize>();
        let this_reduction = std::iter::once(self.initial_reduction)
            .chain(std::iter::repeat(1))
            .take(round + 1)
            .sum::<usize>();

        let k_domain = self.rate + self.k - prev_reduction;
        let k_folded_domain = k_domain - self.folding;
        let prev_rate = self.rate + round * self.folding - prev_reduction;
        let this_rate = self.rate + (round + 1) * self.folding - this_reduction;

        (k_folded_domain, this_rate, prev_rate)
    }

    pub fn open<Transcript>(
        &self,
        transcript: &mut Transcript,
        claims: Vec<Claim<Ext, Ext>>,
        comm_data: CommitmentData<F, F, MCom::Digest>,
        poly: Poly<F, Coeff>,
    ) -> Result<(), crate::Error>
    where
        Transcript: Writer<F>
            + Writer<Ext>
            + Writer<MCom::Digest>
            + Writer<MComExt::Digest>
            + Challenge<F, Ext>
            + ChallengeBits,
    {
        assert_eq!(poly.k(), self.k);

        let n_ood_queries = self.n_ood_queries(self.k, self.rate);
        // draw an univarite point
        // evaluate poly at the point
        // derive the multivariate point
        let ood_claims = tracing::info_span!("ood claims").in_scope(|| {
            (0..n_ood_queries)
                .map(|_| {
                    let point: Ext = Challenge::draw(transcript);
                    let eval = poly.eval_univariate(point);
                    transcript.write(eval)?;
                    let point = Point::expand(poly.k(), point);
                    Ok(Claim::new(point, eval))
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        let claims = itertools::chain!(claims, ood_claims).collect::<Vec<_>>();

        // draw sumcheck constraint combination challenge
        let alpha: Ext = Challenge::draw(transcript);
        // initialize the sumcheck instance
        // `self.folding` number of rounds is run
        let mut sumcheck =
            Sumcheck::new(transcript, self.folding, alpha, &claims, &poly.to_evals())?;

        // derive number of rounds
        let (n_rounds, final_sumcheck_rounds) = compute_number_of_rounds(self.folding, self.k);

        let width = 1 << self.folding;
        let mut round_data: Option<CommitmentData<F, Ext, _>> = None;
        let mut round_point: Option<Point<Ext>> = None;

        // run folding rounds
        for round in 0..n_rounds {
            tracing::info_span!("round", round = round).in_scope(|| {
                // derive round params
                let (k_folded_domain, this_rate, prev_rate) = self.round_params(round);

                // find the current polynomial in coefficient representation
                let poly_in_coeffs = sumcheck.poly().clone().to_coeffs();

                // commit to the polynomial in coefficient representation
                let _round_data: CommitmentData<F, Ext, _> = {
                    // rearrange coefficients since variables are fixed in backwards order
                    let mut transposed = unsafe_allocate_zero_vec(poly_in_coeffs.len());
                    let height = poly_in_coeffs.len() / width;
                    transpose(&poly_in_coeffs, &mut transposed, height, width);
                    // encode and commit with the new rate
                    commit_ext::<_, F, Ext, _>(
                        transcript,
                        &transposed,
                        this_rate,
                        self.folding,
                        &self.comm_ext,
                    )
                }?;

                // find ood claims
                let ood_claims = {
                    let n_ood_queries = self.n_ood_queries(sumcheck.k(), this_rate);
                    (0..n_ood_queries)
                        .map(|_| {
                            let point: Ext = Challenge::draw(transcript);
                            let eval = poly_in_coeffs.eval_univariate(point);
                            transcript.write(eval)?;
                            let point = Point::expand(poly_in_coeffs.k(), point);
                            Ok(Claim::new(point, eval))
                        })
                        .collect::<Result<Vec<_>, _>>()
                }?;

                // derive number of stir queries
                let n_stir_queries = self.n_stir_queries(prev_rate);
                // draw stir points
                let indicies = stir_indicies(transcript, k_folded_domain, n_stir_queries);

                let omega = F::two_adic_generator(k_folded_domain);
                let points = indicies
                    .iter()
                    .map(|&index| omega.exp_u64(index as u64))
                    .collect::<Vec<_>>();

                // find stir claims
                let stir_claims = if round == 0 {
                    let round_point = sumcheck.rs();
                    let cw_local_polys = indicies
                        .iter()
                        .map(|&index| self.comm.query(transcript, index, &comm_data))
                        .collect::<Result<Vec<_>, crate::Error>>()?;

                    cw_local_polys
                        .iter()
                        .zip(points.iter())
                        .map(|(local_poly, &point)| {
                            let eval = Poly::<_, Coeff>::new(local_poly.to_vec())
                                .eval(&round_point.reversed());
                            let poly = sumcheck.poly();
                            let point = Point::expand(poly.k(), point);

                            #[cfg(debug_assertions)]
                            assert_eq!(eval, sumcheck.poly().eval(&point.as_ext::<Ext>()));

                            Claim::new(point, eval)
                        })
                        .collect::<Vec<_>>()
                } else {
                    let round_point = round_point.as_ref().unwrap();
                    let cw_local_polys = indicies
                        .iter()
                        .map(|&index| {
                            self.comm_ext
                                .query(transcript, index, round_data.as_ref().unwrap())
                        })
                        .collect::<Result<Vec<_>, crate::Error>>()?;

                    cw_local_polys
                        .iter()
                        .zip(points.iter())
                        .map(|(local_poly, &point)| {
                            let eval = Poly::<_, Coeff>::new(local_poly.to_vec())
                                .eval(&round_point.reversed());
                            let poly = sumcheck.poly();
                            let point = Point::expand(poly.k(), point);

                            #[cfg(debug_assertions)]
                            assert_eq!(eval, sumcheck.poly().eval(&point.as_ext::<Ext>()));

                            Claim::new(point, eval)
                        })
                        .collect::<Vec<_>>()
                };

                // draw combination challenge and run next rounds of sumcheck
                let alpha = Challenge::draw(transcript);
                // run sumcheck rounds and update the round point
                round_point = Some(sumcheck.fold(
                    transcript,
                    self.folding,
                    alpha,
                    &stir_claims,
                    &ood_claims,
                )?);
                round_data = Some(_round_data);
                Ok(())
            })?;
        }

        tracing::info_span!("final").in_scope(|| {
            // find final polynomial in coefficient representation
            // and send it to the verifier
            let poly_in_coeffs = sumcheck.poly().clone().to_coeffs();
            transcript.write_many(&poly_in_coeffs)?;

            // derive round params
            let (k_folded_domain, _, prev_rate) = self.round_params(n_rounds);

            // derive number of stir queries
            let n_stir_queries = self.n_stir_queries(prev_rate);
            // draw stir points
            let indicies = stir_indicies(transcript, k_folded_domain, n_stir_queries);

            if let Some(round_data) = &round_data {
                indicies.iter().try_for_each(|&index| {
                    let _local_poly = self.comm_ext.query(transcript, index, round_data)?;

                    #[cfg(debug_assertions)]
                    {
                        let point = F::two_adic_generator(k_folded_domain).exp_u64(index as u64);
                        let round_point = round_point.as_ref().unwrap().reversed();
                        let eval = Poly::<_, Coeff>::new(_local_poly.to_vec()).eval(&round_point);
                        assert_eq!(eval, poly_in_coeffs.eval_univariate(Ext::from(point)));
                    }
                    Ok(())
                })?;
            } else {
                indicies.iter().try_for_each(|&index| {
                    let _local_poly = self.comm.query(transcript, index, &comm_data)?;

                    #[cfg(debug_assertions)]
                    {
                        let point = F::two_adic_generator(k_folded_domain).exp_u64(index as u64);
                        let round_point = sumcheck.rs().reversed();
                        let eval = Poly::<_, Coeff>::new(_local_poly.to_vec()).eval(&round_point);
                        assert_eq!(eval, poly_in_coeffs.eval_univariate(Ext::from(point)));
                    }

                    Ok(())
                })?;
            };
            Ok(())
        })?;

        assert_eq!(sumcheck.k(), final_sumcheck_rounds);
        Ok(())
    }

    pub fn verify<Transcript>(
        &self,
        transcript: &mut Transcript,
        claims: Vec<Claim<Ext, Ext>>,
        comm: MCom::Digest,
    ) -> Result<(), crate::Error>
    where
        Transcript: Reader<F>
            + Reader<Ext>
            + Reader<MComExt::Digest>
            + Reader<MCom::Digest>
            + Challenge<F, Ext>
            + ChallengeBits,
    {
        // read ood claims
        let n_ood_queries = self.n_ood_queries(self.k, self.rate);
        let ood_claims = (0..n_ood_queries)
            .map(|_| {
                let point: Ext = Challenge::draw(transcript);
                let eval: Ext = transcript.read()?;
                let point = Point::expand(self.k, point);
                Ok(Claim::new(point, eval))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // claims for the first set of sumcheck rounds
        let claims = itertools::chain!(claims, ood_claims).collect::<Vec<_>>();

        // create the sumcheck verifier
        let mut sumcheck = SumcheckVerifier::<F, Ext>::new(self.k);

        // draw the combination challenge and run the first set of sumcheck rounds
        let alpha = Challenge::draw(transcript);

        // run first set of sumcheck rounds
        let mut round_point = sumcheck.fold(transcript, self.folding, alpha, &[], &claims)?;

        // round commitment is used validate stir points in the next set of rounds
        let mut round_comm: Option<MComExt::Digest> = None;

        // run folding rounds
        let (n_rounds, final_sumcheck_rounds) = compute_number_of_rounds(self.folding, self.k);
        for round in 0..n_rounds {
            // derive round params
            let (k_folded_domain, this_rate, prev_rate) = self.round_params(round);

            // read round commitment
            let _round_comm: MComExt::Digest = transcript.read()?;

            // read ood claims
            let n_ood_queries = self.n_ood_queries(sumcheck.k(), this_rate);
            let ood_claims = (0..n_ood_queries)
                .map(|_| {
                    let point: Ext = Challenge::draw(transcript);
                    let eval: Ext = transcript.read()?;
                    let point = Point::expand(sumcheck.k(), point);
                    Ok(Claim::new(point, eval))
                })
                .collect::<Result<Vec<_>, _>>()?;

            // derive number of stir queries
            let n_stir_queries = self.n_stir_queries(prev_rate);
            // draw stir indexes
            let indicies = stir_indicies(transcript, k_folded_domain, n_stir_queries);

            // get domain generator
            let omega = F::two_adic_generator(k_folded_domain);
            let stir_claims = if round == 0 {
                // in the first round stir points will be verified against the data commitment
                indicies
                    .iter()
                    .map(|&index| {
                        // verify and get leafs which will behave as local polynomials
                        let local_poly = self.comm.verify::<_, F>(
                            transcript,
                            comm,
                            index,
                            1 << self.folding,
                            k_folded_domain,
                        )?;
                        // derive the multivariate point from first variable
                        let point = Point::expand(sumcheck.k(), omega.exp_u64(index as u64));
                        // evaluate the local polynomial at the round point
                        // and return the claim for sumcheck
                        let eval = Poly::<_, Coeff>::new(local_poly).eval(&round_point.reversed());
                        Ok(Claim::new(point, eval))
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                // in the following round stir points will be verified against the round commitment
                indicies
                    .iter()
                    .map(|&index| {
                        // verify and get leafs which will behave as local polynomials
                        let local_poly = self.comm_ext.verify::<_, Ext>(
                            transcript,
                            *round_comm.as_ref().unwrap(),
                            index,
                            1 << self.folding,
                            k_folded_domain,
                        )?;
                        // derive the multivariate point from first variable
                        let point = Point::expand(sumcheck.k(), omega.exp_u64(index as u64));
                        // evaluate the local polynomial at the round point
                        // and return the claim for sumcheck
                        let eval = Poly::<_, Coeff>::new(local_poly).eval(&round_point.reversed());
                        Ok(Claim::new(point, eval))
                    })
                    .collect::<Result<Vec<_>, _>>()
            }?;

            // update the round commitment
            round_comm = Some(_round_comm);

            // draw the combination challenge
            let alpha = Challenge::draw(transcript);
            // run current set of sumcheck rounds
            round_point =
                sumcheck.fold(transcript, self.folding, alpha, &stir_claims, &ood_claims)?;
        }

        // read the final polynomial
        assert_eq!(sumcheck.k(), final_sumcheck_rounds);
        let poly_in_coeffs = Poly::<Ext, Coeff>::new(transcript.read_many(1 << sumcheck.k())?);

        // derive round params for the final set of rounds
        let (k_folded_domain, _, prev_rate) = self.round_params(n_rounds);

        // derive number of stir queries
        let n_stir_queries = self.n_stir_queries(prev_rate);
        // draw stir points
        let indicies = stir_indicies(transcript, k_folded_domain, n_stir_queries);

        // get domain generator
        let omega = F::two_adic_generator(k_folded_domain);
        if let Some(round_comm) = round_comm {
            // stir points will be verified against the previous round commitment
            indicies
                .iter()
                .map(|&index| {
                    let local_poly = self.comm_ext.verify::<_, Ext>(
                        transcript,
                        round_comm,
                        index,
                        1 << self.folding,
                        k_folded_domain,
                    )?;
                    // derive the univariate point
                    let point = omega.exp_u64(index as u64);
                    // evaluate the local polynomial at the round point
                    let eval = Poly::<_, Coeff>::new(local_poly).eval(&round_point.reversed());

                    // verify stir evaluations against final polynomial
                    (eval == poly_in_coeffs.eval_univariate(Ext::from(point)))
                        .then_some(())
                        .ok_or(crate::Error::Verify)?;

                    // derive the multivariate point from first variable
                    // and return the claim for sumcheck
                    let point = Point::expand(sumcheck.k(), omega.exp_u64(index as u64));
                    Ok(Claim::new(point, eval))
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            // if there are no intermediate rounds stir points will be verified against the data commitment
            indicies
                .iter()
                .map(|&index| {
                    let local_poly = self.comm.verify(
                        transcript,
                        comm,
                        index,
                        1 << self.folding,
                        k_folded_domain,
                    )?;
                    // derive the univariate point
                    let point = omega.exp_u64(index as u64);
                    // evaluate the local polynomial at the round point
                    let eval = Poly::<_, Coeff>::new(local_poly).eval(&round_point.reversed());

                    // verify stir evaluations against final polynomial
                    (eval == poly_in_coeffs.eval_univariate(Ext::from(point)))
                        .then_some(())
                        .ok_or(crate::Error::Verify)?;

                    // derive the multivariate point from first variable
                    // and return the claim for sumcheck
                    let point = Point::expand(sumcheck.k(), omega.exp_u64(index as u64));
                    Ok(Claim::new(point, eval))
                })
                .collect::<Result<Vec<_>, _>>()
        }?;

        sumcheck.finalize(transcript, Some(poly_in_coeffs.clone().to_evals()))
    }
}
