//! Per-plate-appearance strikeout simulator.
//!
//! The v2 model prices a start by combining two *independent* predictions: a K
//! rate and a batters-faced count. [`crate::model::strikeout_pmf`] mixes a
//! beta-binomial over an empirical batters-faced residual distribution, which is
//! a fine description of each margin and a wrong description of their joint.
//! Independence fails in one specific, priceable direction: a starter who is
//! actually striking people out allows less traffic, survives the hook longer,
//! and is handed more batters to strike out. Pricing the two halves apart also
//! lets a card disagree with itself across market rungs, because nothing forces
//! the rungs to come from one distribution.
//!
//! This module replaces the product with a simulation. One draw walks a start
//! plate appearance by plate appearance:
//!
//! 1. the batter in the slot strikes out with a log5 blend of the pitcher's
//!    predicted rate, the batter's rate against that hand, and the league rate,
//!    scaled by a times-through-the-order multiplier;
//! 2. a plate appearance that is not a strikeout resolves into a minimal outcome
//!    alphabet (walk, home run, hit in play, out in play) drawn from league
//!    frequencies, which is what gives the hook something to react to;
//! 3. the starter is removed with a logistic hazard in how deep he is and how
//!    much traffic he has allowed *relative to his own expectation*.
//!
//! Ten thousand draws become one PMF over exact strikeout counts — the same
//! shape [`crate::Projection::strikeout_probabilities`] already carries — so
//! every rung of a strikeout ladder is priced off a single coherent object, and
//! K and batters faced are correlated by construction rather than by assumption.
//!
//! # Assumptions, stated plainly
//!
//! - **The starter always leaves before the game does.** No inning, out, or
//!   score state is tracked, and the 27th out never ends a simulated start: only
//!   0.50% of the 16,477 starts in `training_starts_v3.csv` went nine innings, so
//!   carrying out-state would add machinery that changes essentially no prices.
//!   [`MAX_BATTERS_FACED`] stands in for "no start runs forever".
//! - **The batting order never changes.** Plate appearance `i` is taken by lineup
//!   slot `(i - 1) mod 9`, which is exactly true while the starter is in the game:
//!   the order cycles regardless of outcomes, and pinch hitters against a starter
//!   are rare. That also makes times-through-the-order exact rather than modelled.
//! - **The opponent is counted about one and a quarter times.** The v2 K-rate
//!   regression already carries an opponent-offense feature, so the log5 batter
//!   term partly double-counts opponent quality. The regression's factor is an
//!   ER/IP ratio — a run-scoring proxy, not a strikeout proxy — so the overlap is
//!   partial and the lineup term still carries most of the K-specific signal. A
//!   future phase can drop the opponent feature from the K-rate target instead.
//! - **The non-strikeout alphabet is league-average for everyone.** Walk, home
//!   run and hit rates do not vary by pitcher in v1. They exist to give the hazard
//!   state, and the hazard is calibrated against observed batters faced, so a
//!   per-pitcher alphabet would mostly be re-absorbed by that calibration.
//!
//! Parameters live in `models/pns-sim-params-v1.json`, fitted by
//! `cargo run --release -p pns-core --example fit_sim_params`. As with the
//! training dataset in [`crate::data::asof`], a copy on disk wins so a refit
//! takes effect without a rebuild, and an embedded copy keeps a binary that runs
//! outside the repo working.

use crate::data::batters::BatterKRates;
use crate::model::MAX_STRIKEOUTS;
use crate::{PnsError, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SIM_PARAMS_SCHEMA_VERSION: u32 = 1;

/// Hard cap on simulated batters faced. The deepest start in the v3 dataset
/// faced 37, so 45 is unreachable in practice and exists only so a degenerate
/// parameter set cannot produce an unbounded loop.
pub const MAX_BATTERS_FACED: usize = 45;

/// Draws behind a shipped PMF. At 10,000 draws the Monte Carlo standard error on
/// a rung near 50% is about 0.5 cents, an order of magnitude below the spread the
/// paper-trading engine pays.
pub const DEFAULT_DRAWS: usize = 10_000;

const LINEUP_SLOTS: usize = 9;

/// Batters faced that the hazard treats as a neutral workload. The hazard pivot
/// is anchored here and stretched by `pivot_scale`, so `pivot_scale = 1.0` moves
/// the pivot one-for-one with the model's batters-faced prediction.
const BF_PIVOT_CENTER: f64 = 24.0;

/// Rates are clamped away from 0 and 1 before any odds are taken, because
/// `odds(0)` is 0 (a batter who can never strike out would zero out the product)
/// and `odds(1)` is infinite.
const RATE_FLOOR: f64 = 0.005;
const RATE_CEILING: f64 = 0.95;

/// Fallback copy of the fitted parameters for a binary run outside the repo.
/// Mirrors how the model artifact is embedded in `lib.rs`.
const EMBEDDED_SIM_PARAMS: &str = include_str!("../../../../models/pns-sim-params-v1.json");

/// Multiplicative strikeout-rate adjustment by times through the order.
///
/// The times-through-the-order penalty is one of the better-replicated findings
/// in public baseball research (Tango/Lichtman/Dolphin, *The Book*, ch. 8, and
/// the many Fangraphs/Baseball Prospectus replications since): each additional
/// pass costs the pitcher roughly .008–.010 of opponent wOBA, and league
/// strikeout rate falls by something like half a point to a point per pass —
/// about 22.5% first time through, 21.8% second, 21.2% third. The defaults
/// below are that shape, normalized to 1.0 on the first pass.
///
/// The magnitudes matter less than the shape, because the multipliers are
/// re-centered per start (see [`TimesThroughOrder::mean_multiplier`]): the model's
/// `k_rate_mean` is a whole-start average that already contains whatever TTO
/// penalty the pitcher's history carried, so an uncentered penalty would
/// double-count it and depress every projection by ~3%. Centering makes these
/// numbers redistribute a start's strikeouts across passes without changing how
/// many the start expects.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimesThroughOrder {
    pub first: f64,
    pub second: f64,
    pub third_plus: f64,
}

impl Default for TimesThroughOrder {
    fn default() -> Self {
        Self {
            first: 1.00,
            second: 0.97,
            third_plus: 0.94,
        }
    }
}

impl TimesThroughOrder {
    /// The pass multiplier for the `pa_index`-th batter faced (1-based).
    fn multiplier(&self, pa_index: usize) -> f64 {
        match pa_index.saturating_sub(1) / LINEUP_SLOTS {
            0 => self.first,
            1 => self.second,
            _ => self.third_plus,
        }
    }

    /// Mean multiplier over a start of `batters` plate appearances. Dividing the
    /// per-pass multiplier by this keeps the start's expected strikeout rate equal
    /// to the model's `k_rate_mean` no matter how deep the start is predicted to
    /// go, so the TTO shape never becomes a hidden bias on the level.
    fn mean_multiplier(&self, batters: f64) -> f64 {
        let batters = (batters.round().max(1.0) as usize).min(MAX_BATTERS_FACED);
        (1..=batters)
            .map(|index| self.multiplier(index))
            .sum::<f64>()
            / batters as f64
    }
}

/// League-average outcome shares for a plate appearance that did *not* end in a
/// strikeout, so the four shares sum to 1.0.
///
/// Derived from 2023–2025 league rates per plate appearance — K .225, walk plus
/// hit-by-pitch .094, home run .032, non-home-run hit .190, out in play .459 —
/// each divided by the .775 of plate appearances that are not strikeouts. Walks
/// and hit batsmen are one category because the hazard cannot tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutcomeFrequencies {
    pub walk: f64,
    pub home_run: f64,
    pub hit_in_play: f64,
    pub out_in_play: f64,
}

impl Default for OutcomeFrequencies {
    fn default() -> Self {
        Self {
            walk: 0.121_290,
            home_run: 0.041_290,
            hit_in_play: 0.245_161,
            out_in_play: 0.592_259,
        }
    }
}

impl OutcomeFrequencies {
    fn total(&self) -> f64 {
        self.walk + self.home_run + self.hit_in_play + self.out_in_play
    }

    /// Traffic charged to a non-strikeout plate appearance, given a uniform draw
    /// `u` in `[0, 1)`. A home run counts twice: it is both a batter the pitcher
    /// failed to retire and a run on the board, and it is the run on the board
    /// that actually gets starters pulled.
    fn traffic_for(&self, u: f64) -> f64 {
        let total = self.total();
        let walk = self.walk / total;
        let home_run = walk + self.home_run / total;
        let hit = home_run + self.hit_in_play / total;
        if u < walk {
            1.0
        } else if u < home_run {
            2.0
        } else if u < hit {
            1.0
        } else {
            0.0
        }
    }

    /// Expected traffic per non-strikeout plate appearance. Subtracting this each
    /// time through keeps the hazard's traffic term mean-zero, so a start's
    /// *expected* length still tracks the model's batters-faced prediction while
    /// an individual draw that runs into trouble still gets pulled early.
    fn expected_traffic(&self) -> f64 {
        (self.walk + self.hit_in_play + 2.0 * self.home_run) / self.total()
    }
}

/// Logistic removal hazard: after each plate appearance the starter is pulled
/// with probability
/// `sigmoid(intercept + pa_slope * (pa_index - pivot) + traffic_slope * excess_traffic)`.
///
/// `pivot` is the model's batters-faced prediction, stretched about
/// [`BF_PIVOT_CENTER`] by `pivot_scale`, so the hazard is calibrated *conditional
/// on* what the batters-faced regression already knows. `intercept` then sets
/// where the median hook lands relative to that prediction, `pa_slope` sets how
/// tightly starts cluster around it, and `traffic_slope` buys the left tail —
/// the four-inning disaster start that the pitcher's prediction never saw coming.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RemovalHazard {
    pub intercept: f64,
    pub pa_slope: f64,
    pub traffic_slope: f64,
    pub pivot_scale: f64,
}

impl Default for RemovalHazard {
    fn default() -> Self {
        Self {
            intercept: -0.85,
            pa_slope: 0.52,
            traffic_slope: 0.45,
            pivot_scale: 1.0,
        }
    }
}

impl RemovalHazard {
    fn pivot(&self, batters_faced_mean: f64) -> f64 {
        BF_PIVOT_CENTER + self.pivot_scale * (batters_faced_mean - BF_PIVOT_CENTER)
    }

    fn probability(&self, pa_index: usize, pivot: f64, excess_traffic: f64) -> f64 {
        let logit = self.intercept
            + self.pa_slope * (pa_index as f64 - pivot)
            + self.traffic_slope * excess_traffic;
        1.0 / (1.0 + (-logit).exp())
    }
}

/// Observed versus simulated batters-faced quantiles for one predicted-workload
/// bucket, as produced by the fit. This is the fit's own report card: if the
/// simulated quantiles drift from the observed ones, the hazard is wrong and the
/// strikeout counts riding on it are wrong too.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BucketFit {
    pub label: String,
    pub starts: usize,
    pub mean_predicted_batters_faced: f64,
    pub observed_p10: f64,
    pub observed_p50: f64,
    pub observed_p90: f64,
    pub simulated_p10: f64,
    pub simulated_p50: f64,
    pub simulated_p90: f64,
}

/// Provenance for a fitted parameter set.
///
/// `through` is the leakage boundary: the fit saw only starts played strictly
/// before it. `None` means the fit used every start available, which is right for
/// live pricing and wrong for a backtest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitMetadata {
    pub through: Option<NaiveDate>,
    pub generated_at: String,
    pub model_artifact_id: String,
    pub starts_fitted: usize,
    pub search_starts: usize,
    pub draws_per_start: usize,
    /// Weighted per-bucket quantile loss the search settled on. `None` before any
    /// fit has run — JSON has no way to spell NaN, so absence says it explicitly.
    pub quantile_loss: Option<f64>,
    /// Honest note about how the fit was produced, kept with the numbers so a
    /// reader of the JSON cannot miss it.
    pub caveat: String,
    pub buckets: Vec<BucketFit>,
}

impl Default for FitMetadata {
    fn default() -> Self {
        Self {
            through: None,
            generated_at: String::new(),
            model_artifact_id: String::new(),
            starts_fitted: 0,
            search_starts: 0,
            draws_per_start: 0,
            quantile_loss: None,
            caveat: "unfitted defaults".into(),
            buckets: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SimParams {
    pub schema_version: u32,
    pub times_through_order: TimesThroughOrder,
    pub outcomes: OutcomeFrequencies,
    pub hazard: RemovalHazard,
    pub fit: FitMetadata,
}

impl SimParams {
    /// Unfitted defaults: documented constants plus a hazard that has not seen
    /// data. Useful as a search starting point and as a test fixture; the shipped
    /// artifact is what pricing should use.
    pub fn defaults() -> Self {
        Self {
            schema_version: SIM_PARAMS_SCHEMA_VERSION,
            times_through_order: TimesThroughOrder::default(),
            outcomes: OutcomeFrequencies::default(),
            hazard: RemovalHazard::default(),
            fit: FitMetadata::default(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let invalid = |message: &str| Err(PnsError::Model(format!("sim params {message}")));
        if self.schema_version != SIM_PARAMS_SCHEMA_VERSION {
            return invalid(&format!(
                "schema {} is unsupported; expected {SIM_PARAMS_SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        let tto = self.times_through_order;
        if [tto.first, tto.second, tto.third_plus]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 2.0)
        {
            return invalid("times-through-order multipliers must be finite and in (0, 2]");
        }
        let outcomes = self.outcomes;
        let shares = [
            outcomes.walk,
            outcomes.home_run,
            outcomes.hit_in_play,
            outcomes.out_in_play,
        ];
        if shares
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
            || (outcomes.total() - 1.0).abs() > 1e-6
        {
            return invalid("non-strikeout outcome shares must be non-negative and sum to 1");
        }
        let hazard = self.hazard;
        if [
            hazard.intercept,
            hazard.pa_slope,
            hazard.traffic_slope,
            hazard.pivot_scale,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || hazard.pa_slope <= 0.0
            || hazard.traffic_slope < 0.0
            || hazard.pivot_scale <= 0.0
        {
            return invalid("removal hazard parameters are out of range");
        }
        Ok(())
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let params: Self = serde_json::from_str(json)?;
        params.validate()?;
        Ok(params)
    }

    pub fn load_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path).map_err(|error| {
            PnsError::Model(format!(
                "could not read sim params {}: {error}",
                path.display()
            ))
        })?;
        Self::from_json(&json)
    }

    /// The fitted parameters. A disk copy wins so a refit takes effect without a
    /// rebuild; the embedded copy keeps a binary outside the repo working. Same
    /// resolution order as [`crate::data::asof`]'s training dataset.
    pub fn load() -> Result<Self> {
        if let Ok(path) = std::env::var("PNS_SIM_PARAMS") {
            return Self::load_path(path);
        }
        if let Ok(current) = std::env::current_dir()
            && let Some(path) = Self::repo_candidates(&current)
                .into_iter()
                .find(|path| path.exists())
        {
            return Self::load_path(path);
        }
        Self::from_json(EMBEDDED_SIM_PARAMS)
    }

    fn repo_candidates(current: &Path) -> [PathBuf; 2] {
        [
            current.join("models/pns-sim-params-v1.json"),
            current.join("../models/pns-sim-params-v1.json"),
        ]
    }

    pub fn to_json_pretty(&self) -> Result<String> {
        self.validate()?;
        Ok(format!("{}\n", serde_json::to_string_pretty(self)?))
    }
}

/// Everything one simulated start needs. Rates are per plate appearance.
///
/// `k_rate_mean` and `batters_faced_mean` come straight off the v2 model's
/// [`crate::Projection`]; the lineup rates come from
/// [`BatterKRates::k_rate`] for the hand the starter throws.
#[derive(Debug, Clone, PartialEq)]
pub struct StartInputs {
    pub k_rate_mean: f64,
    pub batters_faced_mean: f64,
    pub league_k_rate: f64,
    pub lineup_k_rates: [f64; LINEUP_SLOTS],
}

impl StartInputs {
    /// A start against a league-average nine. Every unknown lineup resolves here,
    /// which reduces the simulation to "this pitcher against an average team" —
    /// the log5 blend returns the pitcher's own rate exactly when the batter's
    /// rate equals the league's.
    pub fn league_lineup(k_rate_mean: f64, batters_faced_mean: f64, league_k_rate: f64) -> Self {
        Self {
            k_rate_mean,
            batters_faced_mean,
            league_k_rate,
            lineup_k_rates: [league_k_rate; LINEUP_SLOTS],
        }
    }

    /// Overrides the first `rates.len()` lineup slots. A short lineup — the
    /// dataset carries an empty `opponent_batters` for every pre-v3 row — leaves
    /// the remaining slots at the league rate rather than recycling the batters
    /// that are known, because recycling would overweight whoever happened to be
    /// recorded.
    pub fn with_lineup(mut self, rates: &[f64]) -> Self {
        for (slot, rate) in self.lineup_k_rates.iter_mut().zip(rates) {
            *slot = *rate;
        }
        self
    }

    /// The lineup a starter actually faced, resolved through the as-of batter
    /// rate fold. An unknown batter id already resolves to the league rate inside
    /// [`BatterKRates::k_rate`], so a partially-known lineup needs no special case.
    pub fn from_lineup(
        k_rate_mean: f64,
        batters_faced_mean: f64,
        rates: &BatterKRates,
        batters: &[u32],
        hand: char,
    ) -> Self {
        let league = rates.league_k_rate(hand);
        let lineup: Vec<_> = batters
            .iter()
            .take(LINEUP_SLOTS)
            .map(|id| rates.k_rate(*id, hand))
            .collect();
        Self::league_lineup(k_rate_mean, batters_faced_mean, league).with_lineup(&lineup)
    }
}

/// One simulated start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartDraw {
    pub strikeouts: u16,
    pub batters_faced: u16,
}

/// `n` simulated starts. Deterministic in `seed`: the same seed replays the same
/// starts, so a projection never jitters between runs.
pub fn simulate_draws(
    inputs: &StartInputs,
    params: &SimParams,
    draws: usize,
    seed: u64,
) -> Vec<StartDraw> {
    let schedule = strikeout_schedule(inputs, params);
    let pivot = params.hazard.pivot(inputs.batters_faced_mean);
    let expected_traffic = params.outcomes.expected_traffic();
    let mut rng = Rng::new(seed);
    let mut results = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut strikeouts = 0u16;
        let mut batters_faced = 0u16;
        let mut excess_traffic = 0.0;
        for pa_index in 1..=MAX_BATTERS_FACED {
            let strikeout_probability = schedule[pa_index - 1];
            batters_faced += 1;
            // One uniform resolves the whole plate appearance: below the
            // strikeout probability is a strikeout, and the rest of the interval
            // is rescaled onto the non-strikeout alphabet.
            let u = rng.uniform();
            let traffic = if u < strikeout_probability {
                strikeouts += 1;
                0.0
            } else {
                let rescaled = (u - strikeout_probability) / (1.0 - strikeout_probability);
                params.outcomes.traffic_for(rescaled)
            };
            excess_traffic += traffic - (1.0 - strikeout_probability) * expected_traffic;
            if rng.uniform() < params.hazard.probability(pa_index, pivot, excess_traffic) {
                break;
            }
        }
        results.push(StartDraw {
            strikeouts,
            batters_faced,
        });
    }
    results
}

/// The strikeout PMF for one start: index `k` is P(K = k), summing to 1.0 — the
/// same shape and length as [`crate::Projection::strikeout_probabilities`], so a
/// consumer can swap one for the other. Counts above [`MAX_STRIKEOUTS`] land in
/// the top bin; with a 45-batter cap that bin is unreachable in practice.
pub fn simulate_start(
    inputs: &StartInputs,
    params: &SimParams,
    draws: usize,
    seed: u64,
) -> Vec<f64> {
    let mut pmf = vec![0.0; MAX_STRIKEOUTS + 1];
    if draws == 0 {
        pmf[0] = 1.0;
        return pmf;
    }
    for draw in simulate_draws(inputs, params, draws, seed) {
        pmf[(draw.strikeouts as usize).min(MAX_STRIKEOUTS)] += 1.0;
    }
    for value in &mut pmf {
        *value /= draws as f64;
    }
    pmf
}

/// Mean of a strikeout PMF, for comparing a simulated distribution against the
/// model's compositional expectation.
pub fn expected_strikeouts(pmf: &[f64]) -> f64 {
    pmf.iter()
        .enumerate()
        .map(|(k, probability)| k as f64 * probability)
        .sum()
}

/// Per-plate-appearance strikeout probability for every slot a start can reach.
fn strikeout_schedule(inputs: &StartInputs, params: &SimParams) -> [f64; MAX_BATTERS_FACED] {
    let center = params
        .times_through_order
        .mean_multiplier(inputs.batters_faced_mean);
    let mut schedule = [0.0; MAX_BATTERS_FACED];
    for (index, slot) in schedule.iter_mut().enumerate() {
        let pa_index = index + 1;
        let batter = inputs.lineup_k_rates[index % LINEUP_SLOTS];
        let base = log5(inputs.k_rate_mean, batter, inputs.league_k_rate);
        let multiplier = params.times_through_order.multiplier(pa_index) / center;
        *slot = (base * multiplier).clamp(RATE_FLOOR, RATE_CEILING);
    }
    schedule
}

/// Odds-ratio (log5) combination: `odds(p) = p / (1 - p)`, and the matchup odds
/// are the pitcher's odds times the batter's odds divided by the league's. A
/// league-average batter returns the pitcher's rate unchanged, which is the
/// property that lets an unknown lineup fall back cleanly.
fn log5(pitcher: f64, batter: f64, league: f64) -> f64 {
    let odds = |p: f64| {
        let p = p.clamp(RATE_FLOOR, RATE_CEILING);
        p / (1.0 - p)
    };
    let combined = odds(pitcher) * odds(batter) / odds(league);
    combined / (1.0 + combined)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64*)
// ---------------------------------------------------------------------------

/// Same generator the paper-trading bootstrap uses (`paper_stats::Rng`), kept
/// separate so the simulator and the statistics engine cannot perturb each
/// other's streams. Adding a dependency for this would be a poor trade: the
/// whole generator is nine lines and the repo needs it to be reproducible, not
/// cryptographic.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // A zero state is a fixed point for xorshift and would emit only zeros.
        Self(if seed == 0 { 1 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`, using the 53 bits an f64 mantissa can hold exactly.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A middling right-hander against a league-average nine.
    fn baseline() -> StartInputs {
        StartInputs::league_lineup(0.24, 23.0, 0.22)
    }

    fn over(pmf: &[f64], line: usize) -> f64 {
        pmf.iter().skip(line).sum()
    }

    #[test]
    fn pmf_is_a_distribution_and_is_deterministic_in_the_seed() {
        let params = SimParams::defaults();
        let pmf = simulate_start(&baseline(), &params, 10_000, 7);
        assert_eq!(pmf.len(), MAX_STRIKEOUTS + 1);
        assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(pmf.iter().all(|probability| *probability >= 0.0));

        let replay = simulate_start(&baseline(), &params, 10_000, 7);
        assert_eq!(pmf, replay, "same seed must replay the same distribution");

        let other = simulate_start(&baseline(), &params, 10_000, 8);
        assert!(
            (other.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "a second seed must still produce a distribution"
        );
        assert_ne!(pmf, other, "different seeds must produce different draws");
    }

    #[test]
    fn a_better_strikeout_pitcher_clears_six_more_often() {
        let params = SimParams::defaults();
        let weak = StartInputs {
            k_rate_mean: 0.18,
            ..baseline()
        };
        let strong = StartInputs {
            k_rate_mean: 0.30,
            ..baseline()
        };
        let weak_over = over(&simulate_start(&weak, &params, 10_000, 11), 6);
        let strong_over = over(&simulate_start(&strong, &params, 10_000, 11), 6);
        assert!(
            strong_over > weak_over,
            "P(K >= 6) was {strong_over} for the better pitcher and {weak_over} for the worse one"
        );
    }

    #[test]
    fn a_high_strikeout_lineup_raises_expected_strikeouts() {
        let params = SimParams::defaults();
        let whiffers = baseline().with_lineup(&[0.30; LINEUP_SLOTS]);
        let contact = baseline().with_lineup(&[0.15; LINEUP_SLOTS]);
        let high = expected_strikeouts(&simulate_start(&whiffers, &params, 10_000, 3));
        let low = expected_strikeouts(&simulate_start(&contact, &params, 10_000, 3));
        assert!(
            high > low + 1.0,
            "expected K was {high} against a high-strikeout nine and {low} against a contact nine"
        );
        // And the average lineup sits between them, because log5 is monotone in
        // the batter's rate.
        let average = expected_strikeouts(&simulate_start(&baseline(), &params, 10_000, 3));
        assert!(low < average && average < high);
    }

    #[test]
    fn an_unknown_lineup_reduces_to_the_pitcher_against_an_average_team() {
        let params = SimParams::defaults();
        // No batter lines at all: every id, and the league itself, resolve to the
        // seed league rate.
        let rates = BatterKRates::default();
        let unknown = StartInputs::from_lineup(0.24, 23.0, &rates, &[], 'R');
        assert!(
            unknown
                .lineup_k_rates
                .iter()
                .all(|rate| *rate == unknown.league_k_rate),
            "an empty lineup must leave every slot at the league rate"
        );
        let pmf = simulate_start(&unknown, &params, 10_000, 5);
        assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(pmf.iter().all(|probability| *probability >= 0.0));
        // Identical to an explicit league-average nine, seed for seed.
        let explicit =
            StartInputs::league_lineup(0.24, 23.0, unknown.league_k_rate).with_lineup(&[]);
        assert_eq!(pmf, simulate_start(&explicit, &params, 10_000, 5));
    }

    #[test]
    fn log5_returns_the_pitcher_rate_against_an_average_batter() {
        assert!((log5(0.28, 0.22, 0.22) - 0.28).abs() < 1e-12);
        assert!(log5(0.28, 0.30, 0.22) > 0.28);
        assert!(log5(0.28, 0.15, 0.22) < 0.28);
    }

    #[test]
    fn centering_keeps_the_simulated_rate_on_the_models_rate() {
        let params = SimParams::defaults();
        // A start's mean strikeout rate per batter faced must reproduce
        // k_rate_mean whatever workload it is predicted for, or the TTO shape
        // becomes a hidden bias on the level.
        for batters_faced_mean in [18.0, 24.0, 30.0] {
            let inputs = StartInputs::league_lineup(0.25, batters_faced_mean, 0.22);
            let schedule = strikeout_schedule(&inputs, &params);
            let faced = batters_faced_mean.round() as usize;
            let mean = schedule[..faced].iter().sum::<f64>() / faced as f64;
            assert!(
                (mean - 0.25).abs() < 1e-9,
                "mean rate {mean} over {faced} batters faced"
            );
        }
    }

    #[test]
    fn strikeouts_and_batters_faced_are_positively_correlated_within_a_start() {
        // The whole point of the simulation: a draw that misses bats allows less
        // traffic, survives the hook, and gets more batters to work with.
        let draws = simulate_draws(&baseline(), &SimParams::defaults(), 20_000, 21);
        let n = draws.len() as f64;
        let rate: Vec<_> = draws
            .iter()
            .map(|draw| draw.strikeouts as f64 / draw.batters_faced as f64)
            .collect();
        let faced: Vec<_> = draws.iter().map(|draw| draw.batters_faced as f64).collect();
        let mean = |values: &[f64]| values.iter().sum::<f64>() / n;
        let (rate_mean, faced_mean) = (mean(&rate), mean(&faced));
        let covariance = rate
            .iter()
            .zip(&faced)
            .map(|(r, f)| (r - rate_mean) * (f - faced_mean))
            .sum::<f64>()
            / n;
        assert!(
            covariance > 0.0,
            "strikeout rate and batters faced came out with covariance {covariance}"
        );
    }

    #[test]
    fn a_higher_hazard_intercept_shortens_starts() {
        let mut quick = SimParams::defaults();
        quick.hazard.intercept += 1.0;
        let mean_faced = |params: &SimParams| {
            let draws = simulate_draws(&baseline(), params, 5_000, 42);
            draws
                .iter()
                .map(|draw| draw.batters_faced as f64)
                .sum::<f64>()
                / draws.len() as f64
        };
        assert!(mean_faced(&quick) < mean_faced(&SimParams::defaults()));
    }

    #[test]
    fn params_round_trip_and_reject_impossible_values() {
        let params = SimParams::defaults();
        let json = params.to_json_pretty().unwrap();
        assert_eq!(SimParams::from_json(&json).unwrap(), params);

        let mut broken = params.clone();
        broken.outcomes.walk += 0.1;
        assert!(broken.validate().is_err());

        let mut broken = params.clone();
        broken.hazard.pa_slope = 0.0;
        assert!(broken.validate().is_err());

        let mut broken = params;
        broken.schema_version += 1;
        assert!(broken.validate().is_err());
    }

    /// The shipped artifact has to keep loading, embedded copy included.
    #[test]
    fn the_shipped_params_load_and_validate() {
        SimParams::from_json(EMBEDDED_SIM_PARAMS).unwrap();
        SimParams::load().unwrap().validate().unwrap();
    }
}
