use itertools::Itertools;
use p3_field::{ExtensionField, Field, TwoAdicField};

use crate::{
    merkle::matrix::{MatrixCommitment, MatrixCommitmentData},
    pcs::Claim,
    poly::{Point, Poly},
    transcript::{ChallengeBits, Writer},
};

pub(super) fn stir_indicies<Transcript>(
    transcript: &mut Transcript,
    k_folding: usize,
    k_domain: usize,
    n_queries: usize,
) -> Vec<usize>
where
    Transcript: ChallengeBits,
{
    println!("STIR BITS {} {}", k_domain, k_domain - k_folding);
    (0..n_queries)
        .map(|_| ChallengeBits::draw(transcript, k_domain - k_folding))
        .sorted()
        .dedup()
        .collect_vec()
}

pub(super) fn stir_queries<Transcript, F: TwoAdicField>(
    transcript: &mut Transcript,
    k: usize,
    k_folding: usize,
    k_domain: usize,
    n_queries: usize,
) -> (Vec<Point<F>>, Vec<usize>)
where
    Transcript: ChallengeBits,
{
    let indicies = stir_indicies(transcript, k_folding, k_domain, n_queries);
    let omega = F::two_adic_generator(k_domain - k_folding);
    let points = indicies
        .iter()
        .map(|&i| Point::expand(k, omega.exp_u64(i as u64)))
        .collect();
    (points, indicies)
}

pub(super) fn query_stir<Transcript, F: Field, MCom: MatrixCommitment<F>>(
    transcript: &mut Transcript,
    mat_comm: &MCom,
    data: &MatrixCommitmentData<F, MCom::Digest>,
    indicies: &[usize],
) -> Result<Vec<Vec<F>>, crate::Error>
where
    Transcript: Writer<F> + Writer<MCom::Digest>,
{
    indicies
        .iter()
        .map(|&index| mat_comm.query(transcript, index, data))
        .try_collect::<_, Vec<_>, _>()
}

pub(super) fn stir_claims<Tr, F: Field, Ext: ExtensionField<F>, MCom: MatrixCommitment<F>>(
    transcript: &mut Tr,
    mat_comm: &MCom,
    data: &MatrixCommitmentData<F, MCom::Digest>,
    indicies: &[usize],
    points: &[Point<Ext>],
    round_point: &Point<Ext>,
) -> Result<Vec<Claim<Ext>>, crate::Error>
where
    Tr: Writer<F> + Writer<MCom::Digest>,
{
    let evals = query_stir(transcript, mat_comm, data, indicies)?;
    Ok(evals
        .iter()
        .zip_eq(points.iter())
        .map(|(evals, point)| {
            let eval = Poly::new(evals.clone()).eval_lagrange(round_point);
            Claim::<Ext>::new(point.clone(), eval)
        })
        .collect_vec())
}
