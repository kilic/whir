use crate::{
    params::{SecurityAssumption, compute_number_of_rounds},
    sumcheck::{
        EqClaim, PowClaim,
        prover::ProverLayout,
        verifier::{SumcheckVerifier, VerifierLayout},
    },
};
use common::Error;
use common::utils::TwoAdicSlice;
use itertools::Itertools;
use merkle::{MerkleData, MerkleTree, MerkleTreeExt};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use poly::{Point, Poly};
use transcript::{Challenge, ChallengeBits, Reader, Writer};

#[tracing::instrument(skip_all)]
pub fn commit_base<Transcript, F: TwoAdicField, Dft: TwoAdicSubgroupDft<F>, MT: MerkleTree<F>>(
    dft: &Dft,
    transcript: &mut Transcript,
    poly: &[F],
    rate: usize,
    folding: usize,
    mt: &MT,
) -> Result<MT::MerkleData, Error>
where
    Transcript: Writer<<MT::MerkleData as MerkleData<F>>::Digest>,
{
    let mut mat = RowMajorMatrix::new(poly.to_vec(), 1 << folding);
    mat.pad_to_height(1 << (poly.k() + rate - folding), F::ZERO);
    let codeword = tracing::info_span!("dft-base", height = mat.height(), width = mat.width())
        .in_scope(|| dft.dft_batch(mat).to_row_major_matrix());
    mt.commit(transcript, codeword)
}

#[tracing::instrument(skip_all)]
pub fn commit_ext<
    Transcript,
    F: TwoAdicField,
    Ext: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
    MT: MerkleTreeExt<F, Ext>,
>(
    dft: &Dft,
    transcript: &mut Transcript,
    poly: &[Ext],
    rate: usize,
    folding: usize,
    mt: &MT,
) -> Result<MT::MerkleData, Error>
where
    Transcript: Writer<<MT::MerkleData as MerkleData<Ext>>::Digest>,
{
    let mut mat = RowMajorMatrix::new(poly.to_vec(), 1 << folding);
    mat.pad_to_height(1 << (poly.k() + rate - folding), Ext::ZERO);
    let codeword = tracing::info_span!("dft-ext", h = mat.height(), w = mat.width())
        .in_scope(|| dft.dft_algebra_batch(mat));
    mt.commit(transcript, codeword)
}

#[tracing::instrument(skip_all)]
fn stir_indexes<Transcript>(
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
    MT: MerkleTree<F>,
    MTExt: MerkleTreeExt<F, Ext>,
> {
    pub k: usize,
    pub folding: usize,
    pub rate: usize,
    pub initial_reduction: usize,
    soundness: SecurityAssumption,
    security_level: usize,
    mt: MT,
    mt_ext: MTExt,
    _marker: std::marker::PhantomData<(F, Ext)>,
}

impl<
    F: TwoAdicField,
    Ext: ExtensionField<F> + TwoAdicField,
    MT: MerkleTree<F>,
    MTExt: MerkleTreeExt<F, Ext>,
> Whir<F, Ext, MT, MTExt>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: usize,
        folding: usize,
        rate: usize,
        initial_reduction: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        mt: MT,
        mt_ext: MTExt,
    ) -> Self {
        Self {
            k,
            folding,
            rate,
            initial_reduction,
            soundness,
            security_level,
            mt,
            mt_ext,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn commit<Transcript, Dft: TwoAdicSubgroupDft<F>>(
        &self,
        dft: &Dft,
        transcript: &mut Transcript,
        poly: &Poly<F>,
    ) -> Result<MT::MerkleData, Error>
    where
        Transcript:
            Writer<Ext> + Writer<<MT::MerkleData as MerkleData<F>>::Digest> + Challenge<F, Ext>,
    {
        commit_base(dft, transcript, poly, self.rate, self.folding, &self.mt)
    }

    fn n_ood_queries(&self, k: usize, rate: usize) -> usize {
        self.soundness
            .n_ood_queries(self.security_level, k, rate, Ext::bits())
    }

    fn n_stir_queries(&self, rate: usize) -> usize {
        self.soundness.n_stir_queries(self.security_level, rate)
    }

    #[tracing::instrument(skip_all)]
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

    pub fn open<Transcript, Dft: TwoAdicSubgroupDft<F>>(
        &self,
        dft: &Dft,
        transcript: &mut Transcript,
        comm_data: MT::MerkleData,
        mut layout: ProverLayout<F, Ext>,
    ) -> Result<(), Error>
    where
        Transcript: Writer<F>
            + Writer<Ext>
            + Writer<<MT::MerkleData as MerkleData<F>>::Digest>
            + Writer<<MTExt::MerkleData as MerkleData<Ext>>::Digest>
            + Challenge<F, Ext>
            + ChallengeBits,
    {
        assert_eq!(layout.k(), self.k);
        assert_eq!(layout.k_svo(), self.folding);

        // generate out-of-domain claims
        tracing::info_span!("ood claims").in_scope(|| {
            let n_ood_queries = self.n_ood_queries(self.k, self.rate);
            (0..n_ood_queries)
                .map(|_| layout.write_virtual_eval(transcript))
                .collect::<Result<Vec<_>, _>>()
        })?;

        // initialize the sumcheck instance
        // `self.folding` number of rounds is run
        let mut sumcheck = layout.new_prover(transcript)?;

        // derive number of rounds
        let (n_rounds, final_sumcheck_rounds) = compute_number_of_rounds(self.folding, self.k);

        let mut round_data: Option<MTExt::MerkleData> = None;
        let mut round_point: Option<Point<Ext>> = None;

        // run folding rounds
        for round in 0..n_rounds {
            tracing::info_span!("round", round = round).in_scope(|| {
                // derive round params
                let (k_folded_domain, this_rate, prev_rate) = self.round_params(round);
                let poly = sumcheck.poly();

                // commit to the polynomial in coefficient representation
                let _round_data: MTExt::MerkleData = {
                    commit_ext::<_, F, Ext, _, _>(
                        dft,
                        transcript,
                        poly,
                        this_rate,
                        self.folding,
                        &self.mt_ext,
                    )
                }?;

                // generate out-of-domain claims
                let ood_claims = tracing::info_span!("ood claims in round").in_scope(|| {
                    let n_ood_queries = self.n_ood_queries(sumcheck.k(), this_rate);
                    (0..n_ood_queries)
                        .map(|_| {
                            let point: Ext = Challenge::draw(transcript);
                            let point = Point::expand(sumcheck.k(), point);
                            let eval = poly.eval_ext(&point);
                            transcript.write(eval)?;
                            Ok(EqClaim::new(point, eval))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })?;

                // derive number of stir queries
                let n_stir_queries = self.n_stir_queries(prev_rate);
                // draw stir points
                let indexes = stir_indexes(transcript, k_folded_domain, n_stir_queries);

                let omega = F::two_adic_generator(k_folded_domain);
                let vars = indexes
                    .iter()
                    .map(|&index| omega.exp_u64(index as u64))
                    .collect::<Vec<_>>();

                // find stir claims
                let stir_claims: Vec<PowClaim<F, Ext>> =
                    tracing::info_span!("stir claims", round = round).in_scope(|| {
                        if round == 0 {
                            let cw_local_polys = indexes
                                .iter()
                                .map(|&index| self.mt.query(transcript, index, &comm_data))
                                .collect::<Result<Vec<_>, Error>>()?;

                            Ok(cw_local_polys
                                .iter()
                                .zip(vars.iter())
                                .map(|(local_poly, &var)| {
                                    let eval =
                                        Poly::new(local_poly.to_vec()).eval_base(sumcheck.rs());
                                    #[cfg(debug_assertions)]
                                    assert_eq!(eval, poly.eval_univariate(Ext::from(var)));
                                    PowClaim::new(var, eval, sumcheck.k())
                                })
                                .collect::<Vec<_>>())
                        } else {
                            let round_point = round_point.as_ref().unwrap();
                            let cw_local_polys = indexes
                                .iter()
                                .map(|&index| {
                                    self.mt_ext.query(
                                        transcript,
                                        index,
                                        round_data.as_ref().unwrap(),
                                    )
                                })
                                .collect::<Result<Vec<_>, Error>>()?;

                            Ok(cw_local_polys
                                .iter()
                                .zip(vars.iter())
                                .map(|(local_poly, &var)| {
                                    let eval =
                                        Poly::new(local_poly.to_vec()).eval_ext::<F>(round_point);
                                    #[cfg(debug_assertions)]
                                    assert_eq!(eval, poly.eval_univariate(Ext::from(var)));
                                    PowClaim::new(var, eval, sumcheck.k())
                                })
                                .collect())
                        }
                    })?;

                // run sumcheck rounds and update the round point
                round_point = match &round_data {
                    Some(_) => {
                        Some(sumcheck.fold(transcript, self.folding, &ood_claims, &stir_claims)?)
                    }
                    None => Some(sumcheck.fold_with_code(
                        transcript,
                        self.folding,
                        &ood_claims,
                        &indexes,
                        comm_data.data(),
                    )?),
                };

                round_data = Some(_round_data);
                Ok(())
            })?;
        }

        tracing::info_span!("final").in_scope(|| {
            let poly = sumcheck.poly();
            // send final polynomial to the verifier
            transcript.write_many(poly)?;

            // derive round params
            let (k_folded_domain, _, prev_rate) = self.round_params(n_rounds);

            // derive number of stir queries
            let n_stir_queries = self.n_stir_queries(prev_rate);
            // draw stir points
            let indicies = stir_indexes(transcript, k_folded_domain, n_stir_queries);

            if let Some(round_data) = &round_data {
                indicies.iter().try_for_each(|&index| {
                    let _cw_local_poly = self.mt_ext.query(transcript, index, round_data)?;

                    #[cfg(debug_assertions)]
                    {
                        let var = F::two_adic_generator(k_folded_domain).exp_u64(index as u64);
                        let round_point = round_point.as_ref().unwrap();
                        let eval = Poly::new(_cw_local_poly.to_vec()).eval_ext::<F>(round_point);
                        assert_eq!(eval, poly.eval_univariate(Ext::from(var)));
                    }
                    Ok(())
                })?;
            } else {
                indicies.iter().try_for_each(|&index| {
                    let _cw_local_poly = self.mt.query(transcript, index, &comm_data)?;

                    #[cfg(debug_assertions)]
                    {
                        let var = F::two_adic_generator(k_folded_domain).exp_u64(index as u64);
                        let eval = Poly::new(_cw_local_poly.to_vec()).eval_base(sumcheck.rs());
                        assert_eq!(eval, poly.eval_univariate(Ext::from(var)));
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
        mut layout: VerifierLayout<F, Ext>,
        comm: <MT::MerkleData as MerkleData<F>>::Digest,
    ) -> Result<(), Error>
    where
        Transcript: Reader<F>
            + Reader<Ext>
            + Reader<<MT::MerkleData as MerkleData<F>>::Digest>
            + Reader<<MTExt::MerkleData as MerkleData<Ext>>::Digest>
            + Challenge<F, Ext>
            + ChallengeBits,
    {
        // read ood claims
        let n_ood_queries = self.n_ood_queries(self.k, self.rate);
        (0..n_ood_queries)
            .map(|_| layout.read_virtual_eval(transcript))
            .collect::<Result<Vec<_>, _>>()?;

        // create the sumcheck verifier
        // and run first set of sumcheck rounds
        let (mut sumcheck, mut round_point) = SumcheckVerifier::<F, Ext>::new(transcript, layout)?;

        // round commitment is used validate stir points in the next set of rounds
        let mut round_comm: Option<<MTExt::MerkleData as MerkleData<Ext>>::Digest> = None;

        // run folding rounds
        let (n_rounds, final_sumcheck_rounds) = compute_number_of_rounds(self.folding, self.k);
        for round in 0..n_rounds {
            // derive round params
            let (k_folded_domain, this_rate, prev_rate) = self.round_params(round);

            // read round commitment
            let _round_comm: <MTExt::MerkleData as MerkleData<Ext>>::Digest = transcript.read()?;

            // read ood claims
            let n_ood_queries = self.n_ood_queries(sumcheck.k(), this_rate);
            let ood_claims = (0..n_ood_queries)
                .map(|_| {
                    // TODO use EqClaim::read
                    let point: Ext = Challenge::draw(transcript);
                    let eval: Ext = transcript.read()?;
                    let point = Point::expand(sumcheck.k(), point);
                    Ok(EqClaim::new(point, eval))
                })
                .collect::<Result<Vec<_>, _>>()?;

            // derive number of stir queries
            let n_stir_queries = self.n_stir_queries(prev_rate);
            // draw stir indexes
            let indicies = stir_indexes(transcript, k_folded_domain, n_stir_queries);

            // get domain generator
            let omega = F::two_adic_generator(k_folded_domain);
            let stir_claims = if round == 0 {
                // in the first round stir points will be verified against the data commitment
                indicies
                    .iter()
                    .map(|&index| {
                        // verify and get leafs which will behave as local polynomials
                        let local_poly = self.mt.verify(
                            transcript,
                            comm,
                            index,
                            1 << self.folding,
                            k_folded_domain,
                        )?;
                        // derive the multivariate point from first variable
                        let var = omega.exp_u64(index as u64);
                        // let point = Point::expand(sumcheck.k, omega.exp_u64(index as u64));
                        // evaluate the local polynomial at the round point
                        // and return the claim for sumcheck
                        let eval = Poly::new(local_poly).eval_base(&round_point);
                        Ok(PowClaim::new(var, eval, sumcheck.k()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                // in the following round stir points will be verified against the round commitment
                indicies
                    .iter()
                    .map(|&index| {
                        // verify and get leafs which will behave as local polynomials
                        let local_poly = self.mt_ext.verify(
                            transcript,
                            *round_comm.as_ref().unwrap(),
                            index,
                            1 << self.folding,
                            k_folded_domain,
                        )?;
                        // derive the multivariate point from first variable
                        let var = omega.exp_u64(index as u64);
                        // let point = Point::expand(sumcheck.k, omega.exp_u64(index as u64));
                        // evaluate the local polynomial at the round point
                        // and return the claim for sumcheck
                        let eval = Poly::new(local_poly).eval_ext::<F>(&round_point);
                        Ok(PowClaim::new(var, eval, sumcheck.k()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            }?;

            // update the round commitment
            round_comm = Some(_round_comm);

            // run current set of sumcheck rounds
            round_point = sumcheck.fold(transcript, self.folding, &ood_claims, &stir_claims)?;
        }

        // read the final polynomial
        assert_eq!(sumcheck.k(), final_sumcheck_rounds);
        let final_poly = Poly::<Ext>::new(transcript.read_many(1 << sumcheck.k())?);

        // derive round params for the final set of rounds
        let (k_folded_domain, _, prev_rate) = self.round_params(n_rounds);

        // derive number of stir queries
        let n_stir_queries = self.n_stir_queries(prev_rate);
        // draw stir indicies
        let indicies = stir_indexes(transcript, k_folded_domain, n_stir_queries);

        // get domain generator
        let omega = F::two_adic_generator(k_folded_domain);
        if let Some(round_comm) = round_comm {
            // stir points will be verified against the previous round commitment
            indicies.iter().try_for_each(|&index| {
                let local_poly = self.mt_ext.verify(
                    transcript,
                    round_comm,
                    index,
                    1 << self.folding,
                    k_folded_domain,
                )?;
                // derive the univariate point
                let var = omega.exp_u64(index as u64);
                // evaluate the local polynomial at the round point
                let eval = Poly::new(local_poly).eval_ext::<F>(&round_point);

                // verify stir evaluations against final polynomial
                (eval == final_poly.eval_univariate(Ext::from(var)))
                    .then_some(())
                    .ok_or(Error::Verify)
            })
        } else {
            // if there are no intermediate rounds stir points will be verified against the data commitment
            indicies.iter().try_for_each(|&index| {
                let local_poly =
                    self.mt
                        .verify(transcript, comm, index, 1 << self.folding, k_folded_domain)?;
                // derive the univariate point
                let var = omega.exp_u64(index as u64);
                // evaluate the local polynomial at the round point
                let eval = Poly::new(local_poly).eval_base(&round_point);

                // verify stir evaluations against final polynomial
                (eval == final_poly.eval_univariate(Ext::from(var)))
                    .then_some(())
                    .ok_or(Error::Verify)
            })
        }?;

        sumcheck.finalize(transcript, Some(final_poly))
    }
}
