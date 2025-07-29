use serde::Serialize;
use std::f64::consts::LOG2_10;

/// Security assumptions determines which proximity parameters and conjectures are assumed by the error computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SecurityAssumption {
    /// Unique decoding assumes that the distance of each oracle is within the UDR of the code.
    /// We refer to this configuration as UD for short.
    /// This requires no conjectures.
    UniqueDecoding,

    /// Johnson bound assumes that the distance of each oracle is within the Johnson bound (1 - √ρ).
    /// We refer to this configuration as JB for short.
    /// This assumes that RS have mutual correlated agreement for proximity parameter up to (1 - √ρ).
    JohnsonBound,

    /// Capacity bound assumes that the distance of each oracle is within the capacity bound 1 - ρ.
    /// We refer to this configuration as CB for short.
    /// This requires conjecturing that RS codes are decodable up to capacity and have correlated agreement (mutual in WHIR) up to capacity.
    CapacityBound,
}

impl SecurityAssumption {
    /// In both JB and CB theorems such as list-size only hold for proximity parameters slighly below the bound.
    /// E.g. in JB proximity gaps holds for every δ ∈ (0, 1 - √ρ).
    /// η is the distance between the chosen proximity parameter and the bound.
    /// I.e. in JB δ = 1 - √ρ - η and in CB δ = 1 - ρ - η.
    // TODO: Maybe it makes more sense to be multiplicative. I think this can be set in a better way.
    #[must_use]
    pub const fn log_eta(&self, log_inv_rate: usize) -> f64 {
        match self {
            // We don't use η in UD
            Self::UniqueDecoding => 0., // TODO: Maybe just panic and avoid calling it in UD?
            // Set as √ρ/20
            Self::JohnsonBound => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
            // Set as ρ/20
            Self::CapacityBound => -(log_inv_rate as f64 + LOG2_10 + 1.),
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate), compute the list size at the specified distance δ.
    #[must_use]
    pub const fn list_size_bits(&self, log_degree: usize, log_inv_rate: usize) -> f64 {
        let log_eta = self.log_eta(log_inv_rate);
        match self {
            // In UD the list size is 1
            Self::UniqueDecoding => 0.,

            // By the JB, RS codes are (1 - √ρ - η, (2*η*√ρ)^-1)-list decodable.
            Self::JohnsonBound => {
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }

            // In CB we assume that RS codes are (1 - ρ - η, d/ρ*η)-list decodable (see Conjecture 5.6 in STIR).
            Self::CapacityBound => (log_degree + log_inv_rate) as f64 - log_eta,
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate) a field_size and an arity, compute the proximity gaps error (in bits) at the specified distance
    #[must_use]
    pub fn prox_gaps_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
    ) -> f64 {
        // The error computed here is from [BCIKS20] for the combination of two functions. Then we multiply it by the folding factor.
        let log_eta = self.log_eta(log_inv_rate);
        // Note that this does not include the field_size
        let error = match self {
            // In UD the error is |L|/|F| = d/ρ*|F|
            Self::UniqueDecoding => (log_degree + log_inv_rate) as f64,

            // In JB the error is degree^2/|F| * (2 * min{ 1 - √ρ - δ, √ρ/20 })^7
            // Since δ = 1 - √ρ - η then 1 - √ρ - δ = η
            // Thus the error is degree^2/|F| * (2 * min { η, √ρ/20 })^7
            Self::JohnsonBound => {
                let numerator = (2 * log_degree) as f64;
                let sqrt_rho_20 = 1. + LOG2_10 + 0.5 * log_inv_rate as f64;
                numerator + 7. * (sqrt_rho_20.min(-log_eta) - 1.)
            }

            // In JB we assume the error is degree/η*ρ^2
            Self::CapacityBound => (log_degree + 2 * log_inv_rate) as f64 - log_eta,
        };

        // Error is  (num_functions - 1) * error/|F|;
        let num_functions_1_log = (num_functions as f64 - 1.).log2();
        field_size_bits as f64 - (error + num_functions_1_log)
    }

    /// The query error is (1 - δ)^t where t is the number of queries.
    /// This computes log(1 - δ).
    /// - In UD, δ is (1 - ρ)/2
    /// - In JB, δ is (1 - √ρ - η)
    /// - In CB, δ is (1 - ρ - η)
    #[must_use]
    pub fn log_1_delta(&self, log_inv_rate: usize) -> f64 {
        let log_eta = self.log_eta(log_inv_rate);
        let eta = 2_f64.powf(log_eta);
        let rate = 1. / f64::from(1 << log_inv_rate);

        let delta = match self {
            Self::UniqueDecoding => 0.5 * (1. - rate),
            Self::JohnsonBound => 1. - rate.sqrt() - eta,
            Self::CapacityBound => 1. - rate - eta,
        };

        (1. - delta).log2()
    }

    /// Compute the number of queries to match the security level
    /// The error to drive down is (1-δ)^t < 2^-λ.
    /// Where δ is set as in the `log_1_delta` function.
    #[must_use]
    pub fn n_stir_queries(&self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);

        num_queries_f.ceil() as usize
    }

    /// Compute the error for the given number of queries
    /// The error to drive down is (1-δ)^t < 2^-λ.
    /// Where δ is set as in the `log_1_delta` function.
    #[must_use]
    pub fn queries_error(&self, log_inv_rate: usize, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        -num_queries * self.log_1_delta(log_inv_rate)
    }

    /// Compute the error for the OOD samples of the protocol
    /// See Lemma 4.5 in STIR.
    /// The error is list_size^2 * (degree/field_size_bits)^reps
    /// NOTE: Here we are discounting the domain size as we assume it is negligible compared to the size of the field.
    #[must_use]
    pub const fn ood_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        ood_samples: usize,
    ) -> f64 {
        if matches!(self, Self::UniqueDecoding) {
            return 0.;
        }

        let list_size_bits = self.list_size_bits(log_degree, log_inv_rate);

        let error = 2. * list_size_bits + (log_degree * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Computes the number of OOD samples required to achieve security_level bits of security
    /// We note that in both STIR and WHIR there are various strategies to set OOD samples.
    /// In this case, we are just sampling one element from the extension field
    #[must_use]
    pub fn n_ood_queries(
        &self,
        security_level: usize,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
    ) -> usize {
        if matches!(self, Self::UniqueDecoding) {
            return 0;
        }

        for ood_samples in 1..64 {
            if self.ood_error(log_degree, log_inv_rate, field_size_bits, ood_samples)
                >= security_level as f64
            {
                return ood_samples;
            }
        }

        panic!("Could not find an appropriate number of OOD samples");
    }
}

/// Each WHIR steps folds the polymomial, which reduces the number of variables.
/// As soon as the number of variables is less than or equal to `MAX_NUM_VARIABLES_TO_SEND_COEFFS`,
/// the prover sends directly the coefficients of the polynomial.
const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 6;

/// Computes the number of WHIR rounds and the number of rounds in the final sumcheck.
#[must_use]
pub(crate) fn compute_number_of_rounds(factor: usize, num_variables: usize) -> (usize, usize) {
    if num_variables <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
        // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
        return (0, num_variables - factor);
    }
    // Starting from `num_variables`, each round reduces the number of variables by `factor`. As soon as the
    // number of variables is less of equal than `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the
    // prover sends directly the coefficients of the polynomial.
    let num_rounds = (num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(factor);
    let final_sumcheck_rounds = num_variables - num_rounds * factor;
    // The -1 accounts for the fact that the last round does not require another folding.
    (num_rounds - 1, final_sumcheck_rounds)
}
