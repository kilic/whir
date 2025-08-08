use itertools::Itertools;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use transpose::transpose;

use crate::{
    merkle::matrix::{MatrixCommitment, MatrixCommitmentData},
    pcs::{
        params::{compute_number_of_rounds, SecurityAssumption},
        sumcheck::Sumcheck,
        Claim,
    },
    poly::{Point, Poly},
    transcript::{Challenge, ChallengeBits, Writer},
    utils::{unsafe_allocate_zero_vec, TwoAdicSlice},
};

pub fn commit<
    Transcript,
    F: TwoAdicField,
    Ext: ExtensionField<F> + TwoAdicField,
    MCom: MatrixCommitment<Ext>,
    DFT: TwoAdicSubgroupDft<Ext>,
>(
    transcript: &mut Transcript,
    poly: &[Ext],
    rate: usize,
    folding: usize,
    mat_comm: &MCom,
) -> Result<MatrixCommitmentData<Ext, MCom::Digest>, crate::Error>
where
    Transcript: Writer<MCom::Digest>,
{
    // pad
    let size = 1 << (poly.k() + rate);
    println!("commit k {}", poly.k() + rate);
    let mut padded: Vec<Ext> = crate::utils::unsafe_allocate_zero_vec(size);
    padded[..poly.len()].copy_from_slice(&poly);
    // encode and commit
    let codeword = DFT::default().dft_algebra_batch(RowMajorMatrix::new(padded, 1 << folding));
    mat_comm.commit(transcript, codeword)
}

fn stir_indicies<Transcript>(
    transcript: &mut Transcript,
    bit_size: usize,
    n_queries: usize,
) -> Vec<usize>
where
    Transcript: ChallengeBits,
{
    (0..n_queries)
        .map(|_| ChallengeBits::draw(transcript, bit_size))
        .sorted()
        .dedup()
        .collect::<Vec<_>>()
}

pub struct Whir<
    F: TwoAdicField,
    Ext: ExtensionField<F>,
    MCom: MatrixCommitment<F>,
    MComExt: MatrixCommitment<Ext>,
> {
    pub k: usize,
    pub folding: usize,
    pub rate: usize,
    soundness: SecurityAssumption,
    security_level: usize,
    comm: MCom,
    comm_ext: MComExt,
    _marker: std::marker::PhantomData<(F, Ext)>,
}

impl<
        F: TwoAdicField,
        Ext: ExtensionField<F> + TwoAdicField,
        MCom: MatrixCommitment<F>,
        MComExt: MatrixCommitment<Ext>,
    > Whir<F, Ext, MCom, MComExt>
{
    pub fn new(
        k: usize,
        folding: usize,
        rate: usize,
        soundness: SecurityAssumption,
        security_level: usize,
        comm: MCom,
        comm_ext: MComExt,
    ) -> Self {
        Self {
            k,
            folding,
            rate,
            soundness,
            security_level,
            comm,
            comm_ext,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn commit<Transcript, DFT: TwoAdicSubgroupDft<F>>(
        &self,
        transcript: &mut Transcript,
        poly_in_coeffs: &Poly<F>,
    ) -> Result<MatrixCommitmentData<F, MCom::Digest>, crate::Error>
    where
        Transcript: Writer<Ext> + Writer<MCom::Digest> + Challenge<F, Ext>,
    {
        let mut transposed = unsafe_allocate_zero_vec(poly_in_coeffs.len());
        let width = 1 << self.folding;
        transpose::transpose(
            &poly_in_coeffs,
            &mut transposed,
            poly_in_coeffs.len() / width,
            width,
        );
        commit::<_, _, _, _, DFT>(transcript, &transposed, self.rate, self.folding, &self.comm)
    }

    fn n_ood_queries(&self, k: usize, rate: usize) -> usize {
        self.soundness
            .n_ood_queries(self.security_level, k, rate, Ext::bits())
    }

    fn n_stir_queries(&self, rate: usize) -> usize {
        self.soundness.n_stir_queries(self.security_level, rate)
    }

    pub fn open<Transcript, DFT: TwoAdicSubgroupDft<Ext>>(
        &self,
        transcript: &mut Transcript,
        claims: Vec<Claim<Ext, Ext>>,
        comm_data: MatrixCommitmentData<F, MCom::Digest>,
        poly_in_coeffs: &Poly<F>,
    ) -> Result<(), crate::Error>
    where
        Transcript: Writer<F>
            + Writer<Ext>
            + Writer<MCom::Digest>
            + Writer<MComExt::Digest>
            + Challenge<F, Ext>
            + ChallengeBits,
    {
        let n_ood_queries = self.n_ood_queries(self.k, self.rate);
        println!("n_ood: {} {} {}", self.k, self.rate, n_ood_queries);

        // draw the first univarite point
        // evaluate poly at the point
        // derive the multivariate point
        let ood_claims = (0..n_ood_queries)
            .map(|_| {
                let point: Ext = Challenge::draw(transcript);
                let eval = poly_in_coeffs.eval_univariate(point);
                transcript.write(eval)?;
                let point = Point::expand(poly_in_coeffs.k(), point);
                Ok(Claim::new(point, eval))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let claims = itertools::chain!(claims, ood_claims).collect::<Vec<_>>();

        // draw sumcheck constraint combination challenge
        let alpha: Ext = Challenge::draw(transcript);

        // initialize the sumcheck instance
        // `self.folding` number of rounds is run
        let mut sumcheck = Sumcheck::new(
            transcript,
            self.k,
            self.folding,
            alpha,
            &claims,
            &poly_in_coeffs.clone().to_lagrange(),
        )?;

        // derive number of rounds
        let (n_rounds, final_sumcheck_rounds) = compute_number_of_rounds(self.folding, self.k);

        // `k_domain` log2 size of initial domain
        // let mut k_domain = self.rate + self.k;
        // let mut rate = self.rate;
        // folding width
        let width = 1 << self.folding;
        let rate_step = self.folding - 1;

        let mut round_data = None;
        let mut round_point: Option<Point<Ext>> = None;

        // run folding rounds
        for round in 0..n_rounds {
            // derive round params
            let k_domain = self.rate + self.k - round;
            let k_folded_domain = k_domain - self.folding;
            let prev_rate = self.rate + rate_step * round;
            let this_rate = prev_rate + rate_step;
            assert_eq!(this_rate, k_domain - sumcheck.k() - 1);

            // find the current polynomial in coefficient representation
            let poly_in_coeffs = sumcheck.poly().clone().to_coeffs();

            // commit to the polynomial in coefficient representation
            let this_round_data = {
                // rearrange coefficients since variables are fixed in backwards order
                let mut transposed = unsafe_allocate_zero_vec(poly_in_coeffs.len());
                let height = poly_in_coeffs.len() / width;
                transpose(&poly_in_coeffs, &mut transposed, height, width);

                // encode and commit with the new rate
                commit::<_, _, _, _, DFT>(
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
                println!("n_ood: {} {} {}", sumcheck.k(), this_rate, n_ood_queries);

                // draw_eval_and_write::<_, Ext, Ext>(transcript, n, sumcheck.poly())
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

            // draw stir points
            let n_stir_queries = self.n_stir_queries(prev_rate);
            let indicies = stir_indicies(transcript, k_folded_domain, n_stir_queries);
            println!("stir {} {}", n_stir_queries, k_folded_domain);
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
                        let eval =
                            Poly::new(local_poly.to_vec()).eval_coeffs(&round_point.reversed());
                        let poly = sumcheck.poly();
                        let point = Point::expand(poly.k(), point);

                        #[cfg(debug_assertions)]
                        assert_eq!(eval, sumcheck.poly().eval_lagrange_ext(&point));

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
                        let eval =
                            Poly::new(local_poly.to_vec()).eval_coeffs(&round_point.reversed());
                        let poly = sumcheck.poly();
                        let point = Point::expand(poly.k(), point);

                        #[cfg(debug_assertions)]
                        assert_eq!(eval, sumcheck.poly().eval_lagrange_ext(&point));

                        Claim::new(point, eval)
                    })
                    .collect::<Vec<_>>()
            };

            // draw combination challenge and run next rounds of sumcheck
            let alpha = Challenge::draw(transcript);
            round_point =
                Some(sumcheck.fold(transcript, self.folding, alpha, &stir_claims, &ood_claims)?);

            round_data = Some(this_round_data);
        }

        // find current polynomial in coefficient representation
        // and send it to the verifier
        let poly_in_coeffs = sumcheck.poly().clone().to_coeffs();
        transcript.write_many(&poly_in_coeffs)?;

        // derive round params
        let prev_rate = self.rate + rate_step * n_rounds;
        let k_domain = self.rate + self.k - n_rounds;
        let k_folded_domain = k_domain - self.folding;
        let n_stir_queries = self.n_stir_queries(prev_rate);

        // draw stir points
        let indicies = stir_indicies(transcript, k_folded_domain, n_stir_queries);
        println!("stir {} {}", n_stir_queries, k_folded_domain);

        if let Some(round_data) = &round_data {
            let _cw_local_polys = indicies
                .iter()
                .map(|&index| self.comm_ext.query(transcript, index, &round_data))
                .collect::<Result<Vec<_>, crate::Error>>()?;

            #[cfg(debug_assertions)]
            {
                let omega = F::two_adic_generator(k_folded_domain);
                _cw_local_polys
                    .iter()
                    .zip(indicies.iter())
                    .for_each(|(local_poly, &index)| {
                        let point = omega.exp_u64(index as u64);
                        let eval = Poly::new(local_poly.to_vec())
                            .eval_coeffs(&round_point.as_ref().unwrap().reversed());
                        assert_eq!(eval, poly_in_coeffs.eval_univariate(Ext::from(point)));
                    });
            }
        } else {
            let _cw_local_polys = indicies
                .iter()
                .map(|&index| self.comm.query(transcript, index, &comm_data))
                .collect::<Result<Vec<_>, crate::Error>>()?;

            #[cfg(debug_assertions)]
            {
                let omega = F::two_adic_generator(k_folded_domain);
                _cw_local_polys
                    .iter()
                    .zip(indicies.iter())
                    .for_each(|(local_poly, &index)| {
                        let point = omega.exp_u64(index as u64);
                        let eval = Poly::new(local_poly.to_vec())
                            .eval_coeffs(&round_point.as_ref().unwrap().reversed());
                        assert_eq!(eval, poly_in_coeffs.eval_univariate(Ext::from(point)));
                    });
            }
        };

        sumcheck.fold(transcript, final_sumcheck_rounds, Ext::ONE, &[], &[])?;
        assert_eq!(sumcheck.k(), 0);
        Ok(())
    }
}
