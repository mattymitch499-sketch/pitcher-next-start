//! Correlation-aware statistics over the paper-trading log.
//!
//! Every row in `paper_trades.csv` is one Kalshi market, but rows are not
//! independent observations. A single pitcher-start gets bets on several
//! thresholds at once (4+, 5+, 6+ …) and those legs resolve together: if the
//! pitcher throws three strikeouts, every "over" leg loses as a block. Treating
//! legs as independent inflates the sample size and makes every confidence
//! interval far too narrow.
//!
//! Everything here clusters by **start** — `(pitcher_id, event_date)` — so the
//! reported uncertainty reflects how many starts were actually observed rather
//! than how many rows happen to be in the file.

use crate::paper::{PaperEntry, Side};
use serde::Serialize;
use std::collections::BTreeMap;

/// Bootstrap resamples. Cheap enough (microseconds per draw at this scale) that
/// the Monte-Carlo jitter falls below the precision the UI prints.
pub(crate) const BOOTSTRAP_DRAWS: usize = 10_000;
/// Below this many distinct starts a cluster bootstrap is not trustworthy, so
/// intervals are suppressed rather than shown at fake precision.
pub(crate) const MIN_CLUSTERS_FOR_INTERVAL: usize = 25;
/// Fixed-width calibration bins. Deliberately not deciles: empirical deciles
/// reshuffle as data arrives, so the curve would not be comparable run to run.
const CALIBRATION_BINS: usize = 10;
/// Fixed seed so identical data always yields an identical interval — the
/// number must not jitter when the user simply reopens the tab.
pub(crate) const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// A point estimate with an optional 95% interval. `low`/`high` are `None` when
/// there were too few starts to say anything honest.
#[derive(Debug, Clone, Serialize)]
pub struct Interval {
    pub point: f64,
    pub low: Option<f64>,
    pub high: Option<f64>,
}

/// The model's forecasting skill measured against the market's own forecast on
/// exactly the same questions, both scored with Brier over identical legs.
///
/// This is the project's central question. The market's price is a competing
/// prediction, not just a cost to pay, and beating it is what an edge *is*.
#[derive(Debug, Clone, Serialize)]
pub struct SkillComparison {
    pub legs: usize,
    pub starts: usize,
    pub model_brier: f64,
    pub market_brier: f64,
    /// `market_brier - model_brier`, so positive means the model forecast
    /// better. Lower Brier is better, hence the subtraction order.
    pub advantage: Option<Interval>,
}

/// One rung of the ladder: what the model said, what the market said, and what
/// actually happened. An aggregate can hide a model that is sharp at 6+ and
/// useless at 9+; this is where that shows up.
#[derive(Debug, Clone, Serialize)]
pub struct ThresholdReliability {
    pub threshold: u16,
    pub legs: usize,
    pub starts: usize,
    pub model_probability: f64,
    pub market_probability: Option<f64>,
    pub observed_rate: f64,
    pub observed_low: Option<f64>,
    pub observed_high: Option<f64>,
    /// Distance from what actually happened. Whichever is smaller was the
    /// better forecast at this rung.
    pub model_error: f64,
    pub market_error: Option<f64>,
}

/// One pitcher-start: the real unit of independent observation.
#[derive(Debug, Clone, Serialize)]
pub struct StartRollup {
    pub pitcher_id: u32,
    pub pitcher_name: String,
    pub event_date: String,
    pub legs: usize,
    pub graded_legs: usize,
    pub profit_cents: i32,
    pub staked_cents: u64,
    /// Every graded bet on this start won, or every one lost.
    pub all_same_direction: bool,
    /// Holds a NO at a low threshold together with a YES at a higher one.
    /// Those cannot both win, so at least one leg is a guaranteed loss.
    pub contradictory: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DateRollup {
    pub event_date: String,
    pub starts: usize,
    pub legs: usize,
    pub profit_cents: i32,
    pub calibration_gap: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterStats {
    pub graded_legs: usize,
    pub graded_starts: usize,
    pub legs_per_start: Option<f64>,
    /// How many independent legs' worth of information the data actually holds.
    pub effective_legs: Option<f64>,
    /// Kish design effect from the bootstrap variance.
    pub design_effect: Option<f64>,
    /// Independent ICC-based estimate of the same quantity. If these two
    /// disagree by more than roughly 2x, suspect a bug.
    pub design_effect_icc: Option<f64>,
    pub same_direction_rate: Option<f64>,
    pub contradictory_starts: usize,
    /// Mean model probability minus observed YES rate. Positive means the model
    /// runs optimistic. Note this cancels sign-flipping errors — see `ece`.
    pub calibration_gap: Option<Interval>,
    pub mean_start_pnl_cents: Option<Interval>,
    pub brier: Option<f64>,
    pub log_score: Option<f64>,
    /// Expected calibration error: does not cancel the way the gap does.
    pub ece: Option<f64>,
    pub per_date: Vec<DateRollup>,
    pub worst_start: Option<StartRollup>,
    pub best_start: Option<StartRollup>,
    /// Range of mean-P&L-per-start across leave-one-date-out refits. A
    /// sensitivity range, deliberately *not* a confidence interval.
    pub date_sensitivity: Option<(f64, f64)>,
    pub skill: Option<SkillComparison>,
    pub by_threshold: Vec<ThresholdReliability>,
    /// Starts whose quoted ladder allowed a risk-free trade: some lower
    /// threshold's ask sat below a higher threshold's bid.
    pub arbitrage_starts: usize,
}

/// One settled market: the two competing forecasts and the outcome.
#[derive(Debug, Clone, Copy)]
struct GradedLeg {
    threshold: u16,
    model_probability: f64,
    /// Mid of the quote, where both sides were quoted. This is the market's
    /// belief; the spread around it is a trading cost, not a forecast error,
    /// so it is deliberately not charged against the market here.
    market_probability: Option<f64>,
    result_yes: bool,
}

/// Legs of one start, split by which statistic each is eligible for. The two
/// populations differ: calibration uses every settled market (bet or not),
/// while P&L only uses markets where a position was actually opened.
#[derive(Debug, Clone, Default)]
struct Cluster {
    pitcher_id: u32,
    pitcher_name: String,
    event_date: String,
    legs: usize,
    graded: Vec<GradedLeg>,
    profit_cents: i32,
    staked_cents: u64,
    graded_bets: usize,
    wins: usize,
    contradictory: bool,
    arbitrage: bool,
}

impl Cluster {
    fn all_same_direction(&self) -> bool {
        self.graded_bets > 1 && (self.wins == 0 || self.wins == self.graded_bets)
    }

    fn rollup(&self) -> StartRollup {
        StartRollup {
            pitcher_id: self.pitcher_id,
            pitcher_name: self.pitcher_name.clone(),
            event_date: self.event_date.clone(),
            legs: self.legs,
            graded_legs: self.graded.len(),
            profit_cents: self.profit_cents,
            staked_cents: self.staked_cents,
            all_same_direction: self.all_same_direction(),
            contradictory: self.contradictory,
        }
    }
}

/// Groups non-voided entries into starts.
fn build_clusters(entries: &[PaperEntry]) -> Vec<Cluster> {
    let mut by_start: BTreeMap<(u32, String), Cluster> = BTreeMap::new();
    // Positions per start, kept to spot mutually exclusive pairs afterwards.
    let mut positions: BTreeMap<(u32, String), Vec<(Side, u16)>> = BTreeMap::new();
    // Quotes per start, kept to spot risk-free ladders afterwards.
    let mut quotes: BTreeMap<(u32, String), Vec<(u16, u16, u16)>> = BTreeMap::new();

    for entry in entries.iter().filter(|entry| !entry.voided) {
        let key = (entry.pitcher_id, entry.event_date.clone());
        let cluster = by_start.entry(key.clone()).or_insert_with(|| Cluster {
            pitcher_id: entry.pitcher_id,
            pitcher_name: entry.pitcher_name.clone(),
            event_date: entry.event_date.clone(),
            ..Cluster::default()
        });
        cluster.legs += 1;
        if let (Some(bid), Some(ask)) = (entry.yes_bid, entry.yes_ask) {
            quotes
                .entry(key.clone())
                .or_default()
                .push((entry.threshold, bid, ask));
        }
        if let Some(result) = entry.result_yes {
            cluster.graded.push(GradedLeg {
                threshold: entry.threshold,
                model_probability: entry.model_probability,
                market_probability: market_probability(entry.yes_bid, entry.yes_ask),
                result_yes: result,
            });
        }
        if let Some(side) = entry.side {
            positions
                .entry(key)
                .or_default()
                .push((side, entry.threshold));
            if let (Some(profit), Some(price)) = (entry.profit_cents, entry.entry_price_cents) {
                cluster.profit_cents += profit;
                cluster.staked_cents += u64::from(price);
                cluster.graded_bets += 1;
                if profit > 0 {
                    cluster.wins += 1;
                }
            }
        }
    }

    for (key, taken) in positions {
        // A NO at threshold `a` wins only when K < a; a YES at threshold `b > a`
        // wins only when K >= b. Both cannot pay, and when a <= K < b neither
        // does. (The mirror image — YES low, NO high — is coherent: it is a bet
        // that K lands inside a range, and both legs can win.)
        let clash = taken.iter().any(|(side, low)| {
            *side == Side::No
                && taken
                    .iter()
                    .any(|(other, high)| *other == Side::Yes && high > low)
        });
        if clash && let Some(cluster) = by_start.get_mut(&key) {
            cluster.contradictory = true;
        }
    }

    for (key, ladder) in quotes {
        // P(K >= k) can only fall as k rises, so a lower rung's ask must never
        // sit below a higher rung's bid. When it does, buying YES low and NO
        // high costs under 100c and pays at least 100c — free money, and no
        // model is needed to believe it.
        let free = ladder.iter().any(|(low, _, ask)| {
            ladder
                .iter()
                .any(|(high, bid, _)| high > low && ask < bid)
        });
        if free && let Some(cluster) = by_start.get_mut(&key) {
            cluster.arbitrage = true;
        }
    }

    by_start.into_values().collect()
}

/// The market's implied probability from a two-sided quote. Nonsense or
/// one-sided quotes yield `None` rather than a guess.
fn market_probability(bid: Option<u16>, ask: Option<u16>) -> Option<f64> {
    let (bid, ask) = (bid?, ask?);
    (bid <= ask && ask <= 100 && (bid > 0 || ask > 0))
        .then(|| f64::from(bid + ask) / 200.0)
}

// ---------------------------------------------------------------------------
// Statistics evaluated over a set of clusters
// ---------------------------------------------------------------------------

/// Mean model probability minus observed YES rate, pooled over every graded leg
/// in the given clusters. `None` when the draw contains no graded legs.
fn calibration_gap(clusters: &[&Cluster]) -> Option<f64> {
    let mut predicted = 0.0;
    let mut observed = 0.0;
    let mut count = 0usize;
    for cluster in clusters {
        for leg in &cluster.graded {
            predicted += leg.model_probability;
            observed += f64::from(u8::from(leg.result_yes));
            count += 1;
        }
    }
    (count > 0).then(|| (predicted - observed) / count as f64)
}

/// Observed YES rate, pooled over graded legs. Used for the design effect.
fn yes_rate(clusters: &[&Cluster]) -> Option<f64> {
    let mut observed = 0.0;
    let mut count = 0usize;
    for cluster in clusters {
        for leg in &cluster.graded {
            observed += f64::from(u8::from(leg.result_yes));
            count += 1;
        }
    }
    (count > 0).then(|| observed / count as f64)
}

/// How much better the model forecast than the market, in Brier points, over
/// the legs where the market actually quoted a two-sided price. Positive means
/// the model won. `None` when no leg carries a market price.
fn brier_advantage(clusters: &[&Cluster]) -> Option<f64> {
    let (mut model, mut market, mut count) = (0.0, 0.0, 0usize);
    for cluster in clusters {
        for leg in &cluster.graded {
            let Some(quoted) = leg.market_probability else {
                continue;
            };
            let outcome = f64::from(u8::from(leg.result_yes));
            model += (leg.model_probability - outcome).powi(2);
            market += (quoted - outcome).powi(2);
            count += 1;
        }
    }
    (count > 0).then(|| (market - model) / count as f64)
}

/// Mean profit per start, counting only starts that actually carry a graded
/// bet. A start with no position contributes nothing and is not in the divisor.
fn mean_start_pnl(clusters: &[&Cluster]) -> Option<f64> {
    let betting: Vec<_> = clusters
        .iter()
        .filter(|cluster| cluster.graded_bets > 0)
        .collect();
    (!betting.is_empty()).then(|| {
        betting
            .iter()
            .map(|cluster| f64::from(cluster.profit_cents))
            .sum::<f64>()
            / betting.len() as f64
    })
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64*)
// ---------------------------------------------------------------------------

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
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

    /// Uniform index below `n`. The modulo bias is on the order of `n / 2^64`,
    /// which is unmeasurable for the cluster counts here.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Normal quantile helpers (needed for BCa)
// ---------------------------------------------------------------------------

fn normal_cdf(z: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 applied to erf, ~1.5e-7 absolute.
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    0.5 * (1.0 + sign * erf)
}

fn normal_quantile(p: f64) -> f64 {
    // Acklam's rational approximation, ~1.15e-9 relative over (0,1).
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let low = 0.024_25;
    if p < low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn percentile_of_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (p * (sorted.len() - 1) as f64).clamp(0.0, (sorted.len() - 1) as f64);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (rank - lower as f64) * (sorted[upper] - sorted[lower])
    }
}

// ---------------------------------------------------------------------------
// Cluster bootstrap
// ---------------------------------------------------------------------------

/// Bootstrap replicates of the three statistics, drawn from one shared resample
/// per iteration. Sharing the draw is deliberate: the calibration and P&L
/// populations are different subsets of the same starts, so resampling the
/// union once keeps them consistent with each other.
struct Replicates {
    gap: Vec<f64>,
    pnl: Vec<f64>,
    rate: Vec<f64>,
    advantage: Vec<f64>,
    /// Observed YES rate per threshold, so the ladder rows get intervals from
    /// the same draws rather than a separate bootstrap each.
    by_threshold: BTreeMap<u16, Vec<f64>>,
}

fn bootstrap(clusters: &[Cluster], thresholds: &[u16]) -> Replicates {
    let mut rng = Rng::new(SEED);
    let mut replicates = Replicates {
        gap: Vec::with_capacity(BOOTSTRAP_DRAWS),
        pnl: Vec::with_capacity(BOOTSTRAP_DRAWS),
        rate: Vec::with_capacity(BOOTSTRAP_DRAWS),
        advantage: Vec::with_capacity(BOOTSTRAP_DRAWS),
        by_threshold: thresholds
            .iter()
            .map(|threshold| (*threshold, Vec::with_capacity(BOOTSTRAP_DRAWS)))
            .collect(),
    };
    if clusters.is_empty() {
        return replicates;
    }
    let mut draw: Vec<&Cluster> = Vec::with_capacity(clusters.len());
    for _ in 0..BOOTSTRAP_DRAWS {
        draw.clear();
        for _ in 0..clusters.len() {
            draw.push(&clusters[rng.below(clusters.len())]);
        }
        // A draw can legitimately contain no eligible legs for a given
        // statistic; skip it there rather than recording a divide-by-zero.
        if let Some(value) = calibration_gap(&draw) {
            replicates.gap.push(value);
        }
        if let Some(value) = mean_start_pnl(&draw) {
            replicates.pnl.push(value);
        }
        if let Some(value) = yes_rate(&draw) {
            replicates.rate.push(value);
        }
        if let Some(value) = brier_advantage(&draw) {
            replicates.advantage.push(value);
        }
        for (threshold, samples) in replicates.by_threshold.iter_mut() {
            let (mut hits, mut count) = (0.0, 0usize);
            for cluster in &draw {
                for leg in cluster.graded.iter().filter(|leg| leg.threshold == *threshold) {
                    hits += f64::from(u8::from(leg.result_yes));
                    count += 1;
                }
            }
            if count > 0 {
                samples.push(hits / count as f64);
            }
        }
    }
    replicates
}

/// BCa interval. Percentile intervals inherit the O(1/M) bias these ratio
/// estimators carry; BCa corrects for that bias and for skew, using a
/// leave-one-cluster-out jackknife for the acceleration term.
///
/// Generic in the cluster type so any correlated-observation problem in the
/// crate — paper-trading legs grouped by start, backtest ladder rungs grouped
/// by start — gets the same interval rather than a second implementation.
pub(crate) fn bca_interval<C, F>(
    point: f64,
    replicates: &[f64],
    clusters: &[C],
    statistic: F,
) -> Option<Interval>
where
    F: Fn(&[&C]) -> Option<f64>,
{
    if clusters.len() < MIN_CLUSTERS_FOR_INTERVAL || replicates.len() < BOOTSTRAP_DRAWS / 2 {
        return Some(Interval {
            point,
            low: None,
            high: None,
        });
    }
    let mut sorted = replicates.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Bias correction: how far the bootstrap median sits from the estimate.
    let below = sorted.iter().filter(|value| **value < point).count();
    let proportion = (below as f64 / sorted.len() as f64).clamp(1e-6, 1.0 - 1e-6);
    let z0 = normal_quantile(proportion);

    // Acceleration from the cluster jackknife.
    let mut jackknife = Vec::with_capacity(clusters.len());
    for skip in 0..clusters.len() {
        let subset: Vec<&C> = clusters
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != skip)
            .map(|(_, cluster)| cluster)
            .collect();
        if let Some(value) = statistic(&subset) {
            jackknife.push(value);
        }
    }
    if jackknife.len() < MIN_CLUSTERS_FOR_INTERVAL {
        return Some(Interval {
            point,
            low: None,
            high: None,
        });
    }
    let mean = jackknife.iter().sum::<f64>() / jackknife.len() as f64;
    let (mut sum2, mut sum3) = (0.0, 0.0);
    for value in &jackknife {
        let deviation = mean - value;
        sum2 += deviation * deviation;
        sum3 += deviation * deviation * deviation;
    }
    let acceleration = if sum2 <= 0.0 {
        0.0
    } else {
        sum3 / (6.0 * sum2.powf(1.5))
    };

    let adjust = |z: f64| {
        let numerator = z0 + z;
        let corrected = z0 + numerator / (1.0 - acceleration * numerator);
        normal_cdf(corrected)
    };
    let low = percentile_of_sorted(&sorted, adjust(normal_quantile(0.025)));
    let high = percentile_of_sorted(&sorted, adjust(normal_quantile(0.975)));
    Some(Interval {
        point,
        low: Some(low.min(high)),
        high: Some(low.max(high)),
    })
}

// ---------------------------------------------------------------------------
// Design effect
// ---------------------------------------------------------------------------

/// One-way ANOVA intra-cluster correlation on the binary outcomes, converted to
/// a design effect. Uses the size-weighted mean cluster size: with unequal
/// clusters the plain mean understates the effect substantially.
fn icc_design_effect(clusters: &[Cluster]) -> Option<f64> {
    let sizes: Vec<usize> = clusters
        .iter()
        .map(|cluster| cluster.graded.len())
        .filter(|size| *size > 0)
        .collect();
    let groups = sizes.len();
    let total: usize = sizes.iter().sum();
    if groups < 2 || total <= groups {
        return None;
    }
    let grand = yes_rate(&clusters.iter().collect::<Vec<_>>())?;
    let mut between = 0.0;
    let mut within = 0.0;
    for cluster in clusters.iter().filter(|c| !c.graded.is_empty()) {
        let size = cluster.graded.len() as f64;
        let mean = cluster
            .graded
            .iter()
            .map(|leg| f64::from(u8::from(leg.result_yes)))
            .sum::<f64>()
            / size;
        between += size * (mean - grand).powi(2);
        for leg in &cluster.graded {
            within += (f64::from(u8::from(leg.result_yes)) - mean).powi(2);
        }
    }
    let ms_between = between / (groups - 1) as f64;
    let ms_within = within / (total - groups) as f64;
    let sum_squares: f64 = sizes.iter().map(|size| (size * size) as f64).sum();
    // m0 for unbalanced one-way ANOVA.
    let m0 = (total as f64 - sum_squares / total as f64) / (groups - 1) as f64;
    if m0 <= 0.0 {
        return None;
    }
    let icc = ((ms_between - ms_within) / (ms_between + (m0 - 1.0) * ms_within)).clamp(0.0, 1.0);
    // Size-weighted mean cluster size, not the plain mean.
    let weighted = sum_squares / total as f64;
    Some(1.0 + (weighted - 1.0) * icc)
}

// ---------------------------------------------------------------------------
// Proper scoring rules
// ---------------------------------------------------------------------------

/// Brier, log score and ECE over every graded leg. Unlike the calibration gap
/// these do not cancel when the model is high in one region and low in another.
fn scores(clusters: &[Cluster]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let legs: Vec<(f64, bool)> = clusters
        .iter()
        .flat_map(|cluster| {
            cluster
                .graded
                .iter()
                .map(|leg| (leg.model_probability, leg.result_yes))
        })
        .collect();
    probability_scores(&legs)
}

/// The same three proper scores over any `(forecast probability, outcome)`
/// pairs, so a second forecaster — a market quote, a simulated ladder — is
/// graded by exactly the machinery the model is graded by. ECE uses
/// [`CALIBRATION_BINS`] fixed-width bins.
pub(crate) fn probability_scores(legs: &[(f64, bool)]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if legs.is_empty() {
        return (None, None, None);
    }
    let count = legs.len() as f64;
    let brier = legs
        .iter()
        .map(|(probability, outcome)| (probability - f64::from(u8::from(*outcome))).powi(2))
        .sum::<f64>()
        / count;
    let log_score = -legs
        .iter()
        .map(|(probability, outcome)| {
            let clamped = probability.clamp(1e-6, 1.0 - 1e-6);
            if *outcome {
                clamped.ln()
            } else {
                (1.0 - clamped).ln()
            }
        })
        .sum::<f64>()
        / count;

    let mut bin_total = [0.0; CALIBRATION_BINS];
    let mut bin_hits = [0.0; CALIBRATION_BINS];
    let mut bin_count = [0usize; CALIBRATION_BINS];
    for (probability, outcome) in legs {
        let index = ((probability * CALIBRATION_BINS as f64) as usize).min(CALIBRATION_BINS - 1);
        bin_total[index] += probability;
        bin_hits[index] += f64::from(u8::from(*outcome));
        bin_count[index] += 1;
    }
    let ece = (0..CALIBRATION_BINS)
        .filter(|index| bin_count[*index] > 0)
        .map(|index| {
            let size = bin_count[index] as f64;
            let predicted = bin_total[index] / size;
            let observed = bin_hits[index] / size;
            (size / count) * (predicted - observed).abs()
        })
        .sum::<f64>();
    (Some(brier), Some(log_score), Some(ece))
}

// ---------------------------------------------------------------------------
// Ladder breakdown
// ---------------------------------------------------------------------------

/// One row per threshold: model forecast, market forecast, and outcome. Rows
/// use plain percentile intervals rather than BCa — they are a breakdown, and
/// a jackknife per rung would cost far more than the extra accuracy is worth.
fn threshold_reliability(
    clusters: &[Cluster],
    replicates: &BTreeMap<u16, Vec<f64>>,
) -> Vec<ThresholdReliability> {
    let mut rows: BTreeMap<u16, (f64, f64, usize, f64, usize, usize)> = BTreeMap::new();
    for cluster in clusters {
        let mut counted: BTreeMap<u16, ()> = BTreeMap::new();
        for leg in &cluster.graded {
            let row = rows.entry(leg.threshold).or_insert((0.0, 0.0, 0, 0.0, 0, 0));
            row.0 += leg.model_probability;
            row.1 += f64::from(u8::from(leg.result_yes));
            row.2 += 1;
            if let Some(quoted) = leg.market_probability {
                row.3 += quoted;
                row.4 += 1;
            }
            if counted.insert(leg.threshold, ()).is_none() {
                row.5 += 1;
            }
        }
    }
    rows.into_iter()
        .map(
            |(threshold, (model_sum, hits, legs, market_sum, quoted, starts))| {
                let count = legs as f64;
                let model = model_sum / count;
                let observed = hits / count;
                let market = (quoted > 0).then(|| market_sum / quoted as f64);
                let samples = replicates.get(&threshold);
                let (low, high) = match samples {
                    Some(values)
                        if starts >= MIN_CLUSTERS_FOR_INTERVAL
                            && values.len() >= BOOTSTRAP_DRAWS / 2 =>
                    {
                        let mut sorted = values.clone();
                        sorted.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        (
                            Some(percentile_of_sorted(&sorted, 0.025)),
                            Some(percentile_of_sorted(&sorted, 0.975)),
                        )
                    }
                    _ => (None, None),
                };
                ThresholdReliability {
                    threshold,
                    legs,
                    starts,
                    model_probability: model,
                    market_probability: market,
                    observed_rate: observed,
                    observed_low: low,
                    observed_high: high,
                    model_error: (model - observed).abs(),
                    market_error: market.map(|value| (value - observed).abs()),
                }
            },
        )
        .collect()
}

/// Model versus market as competing forecasters over identical questions.
fn skill_comparison(
    clusters: &[Cluster],
    replicates: &[f64],
    advantage: Option<f64>,
) -> Option<SkillComparison> {
    let advantage = advantage?;
    let (mut model, mut market, mut legs, mut starts) = (0.0, 0.0, 0usize, 0usize);
    for cluster in clusters {
        let mut quoted_here = false;
        for leg in &cluster.graded {
            let Some(quoted) = leg.market_probability else {
                continue;
            };
            let outcome = f64::from(u8::from(leg.result_yes));
            model += (leg.model_probability - outcome).powi(2);
            market += (quoted - outcome).powi(2);
            legs += 1;
            quoted_here = true;
        }
        if quoted_here {
            starts += 1;
        }
    }
    (legs > 0).then(|| SkillComparison {
        legs,
        starts,
        model_brier: model / legs as f64,
        market_brier: market / legs as f64,
        advantage: bca_interval(advantage, replicates, clusters, brier_advantage),
    })
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Cluster-aware view of the paper-trading log. Deliberately separate from
/// [`crate::paper::summarize`], which must stay cheap.
pub fn cluster_stats(entries: &[PaperEntry]) -> ClusterStats {
    let clusters = build_clusters(entries);
    let graded: Vec<Cluster> = clusters
        .iter()
        .filter(|cluster| !cluster.graded.is_empty() || cluster.graded_bets > 0)
        .cloned()
        .collect();
    let refs: Vec<&Cluster> = graded.iter().collect();

    let graded_legs: usize = graded.iter().map(|cluster| cluster.graded.len()).sum();
    let graded_starts = graded.len();
    let multi_leg: Vec<&Cluster> = graded
        .iter()
        .filter(|cluster| cluster.graded_bets > 1)
        .collect();

    let thresholds: Vec<u16> = graded
        .iter()
        .flat_map(|cluster| cluster.graded.iter().map(|leg| leg.threshold))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let replicates = bootstrap(&graded, &thresholds);
    let gap_point = calibration_gap(&refs);
    let pnl_point = mean_start_pnl(&refs);
    let advantage_point = brier_advantage(&refs);

    // Kish design effect: bootstrap variance of the YES rate against the
    // simple-random-sample variance the same number of legs would give.
    let (effective_legs, design_effect) = match (yes_rate(&refs), replicates.rate.len()) {
        (Some(rate), count) if count > 1 && graded_legs > 0 => {
            let mean = replicates.rate.iter().sum::<f64>() / count as f64;
            let variance = replicates
                .rate
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (count - 1) as f64;
            if variance > 0.0 {
                let n_eff = rate * (1.0 - rate) / variance;
                (Some(n_eff), Some(graded_legs as f64 / n_eff))
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    };

    let (brier, log_score, ece) = scores(&graded);

    // Per-date rollups, printed raw because starts on one date share a model
    // artifact and a league-wide run environment and so are correlated too.
    let mut per_date: BTreeMap<String, DateRollup> = BTreeMap::new();
    for cluster in &graded {
        let entry = per_date
            .entry(cluster.event_date.clone())
            .or_insert_with(|| DateRollup {
                event_date: cluster.event_date.clone(),
                starts: 0,
                legs: 0,
                profit_cents: 0,
                calibration_gap: None,
            });
        entry.starts += 1;
        entry.legs += cluster.legs;
        entry.profit_cents += cluster.profit_cents;
    }
    for rollup in per_date.values_mut() {
        let same_date: Vec<&Cluster> = graded
            .iter()
            .filter(|cluster| cluster.event_date == rollup.event_date)
            .collect();
        rollup.calibration_gap = calibration_gap(&same_date);
    }
    let per_date: Vec<DateRollup> = per_date.into_values().collect();

    // Leave-one-date-out sensitivity. A range, not an interval: with only a
    // handful of dates a date-level bootstrap would invent false precision.
    let date_sensitivity = (per_date.len() > 1).then(|| {
        let estimates: Vec<f64> = per_date
            .iter()
            .filter_map(|rollup| {
                let subset: Vec<&Cluster> = graded
                    .iter()
                    .filter(|cluster| cluster.event_date != rollup.event_date)
                    .collect();
                mean_start_pnl(&subset)
            })
            .collect();
        let low = estimates.iter().copied().fold(f64::INFINITY, f64::min);
        let high = estimates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    });

    let betting: Vec<&Cluster> = graded
        .iter()
        .filter(|cluster| cluster.graded_bets > 0)
        .collect();

    ClusterStats {
        graded_legs,
        graded_starts,
        legs_per_start: (graded_starts > 0)
            .then(|| graded.iter().map(|c| c.legs).sum::<usize>() as f64 / graded_starts as f64),
        effective_legs,
        design_effect,
        design_effect_icc: icc_design_effect(&graded),
        same_direction_rate: (!multi_leg.is_empty()).then(|| {
            multi_leg
                .iter()
                .filter(|cluster| cluster.all_same_direction())
                .count() as f64
                / multi_leg.len() as f64
        }),
        contradictory_starts: graded
            .iter()
            .filter(|cluster| cluster.contradictory)
            .count(),
        calibration_gap: gap_point
            .and_then(|point| bca_interval(point, &replicates.gap, &graded, calibration_gap)),
        mean_start_pnl_cents: pnl_point
            .and_then(|point| bca_interval(point, &replicates.pnl, &graded, mean_start_pnl)),
        brier,
        log_score,
        ece,
        per_date,
        worst_start: betting
            .iter()
            .min_by_key(|cluster| cluster.profit_cents)
            .map(|cluster| cluster.rollup()),
        best_start: betting
            .iter()
            .max_by_key(|cluster| cluster.profit_cents)
            .map(|cluster| cluster.rollup()),
        date_sensitivity,
        skill: skill_comparison(&graded, &replicates.advantage, advantage_point),
        by_threshold: threshold_reliability(&graded, &replicates.by_threshold),
        arbitrage_starts: graded.iter().filter(|cluster| cluster.arbitrage).count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::{PaperEntry, Side};

    /// Overrides the quote so a test can control what the market "believed".
    fn quoted(
        pitcher: u32,
        threshold: u16,
        probability: f64,
        market: f64,
        result: bool,
    ) -> PaperEntry {
        let cents = (market * 100.0).round() as u16;
        let mut entry = leg(
            pitcher,
            "2026-07-21",
            threshold,
            probability,
            None,
            Some(result),
            None,
        );
        entry.yes_bid = Some(cents);
        entry.yes_ask = Some(cents);
        entry
    }

    fn leg(
        pitcher: u32,
        date: &str,
        threshold: u16,
        probability: f64,
        side: Option<Side>,
        result: Option<bool>,
        profit: Option<i32>,
    ) -> PaperEntry {
        PaperEntry {
            market_ticker: format!("T-{pitcher}-{date}-{threshold}"),
            event_date: date.into(),
            pitcher_id: pitcher,
            pitcher_name: format!("Pitcher {pitcher}"),
            threshold,
            model_probability: probability,
            yes_bid: Some(40),
            yes_ask: Some(45),
            side,
            entry_price_cents: side.map(|_| 45),
            edge: side.map(|_| 0.1),
            recorded_at: "now".into(),
            actual_strikeouts: result.map(|_| 6.0),
            result_yes: result,
            profit_cents: profit,
            voided: false,
            model_expected_strikeouts: Some(6.0),
            model_sd_strikeouts: Some(2.0),
            model_artifact_id: "m".into(),
            sim_probability: None,
            sim_expected_strikeouts: None,
        }
    }

    /// Every leg in a start resolves identically, so the whole start carries
    /// only one observation's worth of information.
    #[test]
    fn perfectly_correlated_legs_collapse_to_one_observation_per_start() {
        let mut entries = Vec::new();
        for pitcher in 0..40u32 {
            let outcome = pitcher % 2 == 0;
            for threshold in 4..8u16 {
                entries.push(leg(
                    pitcher,
                    "2026-07-21",
                    threshold,
                    0.5,
                    Some(Side::Yes),
                    Some(outcome),
                    Some(if outcome { 55 } else { -45 }),
                ));
            }
        }
        let stats = cluster_stats(&entries);
        assert_eq!(stats.graded_legs, 160);
        assert_eq!(stats.graded_starts, 40);
        let effective = stats.effective_legs.unwrap();
        // 160 legs, but only 40 independent starts.
        assert!(
            (30.0..55.0).contains(&effective),
            "effective legs {effective} should be near the 40 starts, not 160"
        );
        let deff = stats.design_effect.unwrap();
        assert!((2.5..6.0).contains(&deff), "design effect {deff} near 4");
        assert_eq!(stats.same_direction_rate, Some(1.0));
    }

    /// One leg per start means legs already are independent: no inflation.
    #[test]
    fn singleton_starts_have_no_design_effect() {
        let entries: Vec<PaperEntry> = (0..60u32)
            .map(|pitcher| {
                leg(
                    pitcher,
                    "2026-07-21",
                    6,
                    0.5,
                    Some(Side::Yes),
                    Some(pitcher % 2 == 0),
                    Some(if pitcher % 2 == 0 { 55 } else { -45 }),
                )
            })
            .collect();
        let stats = cluster_stats(&entries);
        assert_eq!(stats.graded_starts, 60);
        let deff = stats.design_effect.unwrap();
        assert!((0.6..1.6).contains(&deff), "design effect {deff} near 1");
        // No multi-leg starts, so the same-direction rate is undefined.
        assert_eq!(stats.same_direction_rate, None);
    }

    /// The two design-effect estimates come from unrelated math and should
    /// broadly agree; large divergence means one of them is wrong.
    #[test]
    fn bootstrap_and_icc_design_effects_agree() {
        let mut entries = Vec::new();
        for pitcher in 0..40u32 {
            let outcome = pitcher % 3 == 0;
            for threshold in 4..8u16 {
                entries.push(leg(
                    pitcher,
                    "2026-07-21",
                    threshold,
                    0.4,
                    Some(Side::Yes),
                    Some(outcome),
                    Some(if outcome { 55 } else { -45 }),
                ));
            }
        }
        let stats = cluster_stats(&entries);
        let bootstrap = stats.design_effect.unwrap();
        let icc = stats.design_effect_icc.unwrap();
        let ratio = bootstrap.max(icc) / bootstrap.min(icc);
        assert!(ratio < 2.0, "bootstrap {bootstrap} vs icc {icc}");
    }

    #[test]
    fn intervals_are_suppressed_below_the_cluster_floor() {
        let entries: Vec<PaperEntry> = (0..5u32)
            .map(|pitcher| {
                leg(
                    pitcher,
                    "2026-07-21",
                    6,
                    0.5,
                    Some(Side::Yes),
                    Some(true),
                    Some(55),
                )
            })
            .collect();
        let stats = cluster_stats(&entries);
        let gap = stats.calibration_gap.unwrap();
        assert!(gap.low.is_none() && gap.high.is_none());
        // The point estimate is still reported.
        assert!(gap.point.is_finite());
    }

    #[test]
    fn flags_mutually_exclusive_positions_but_not_range_bets() {
        // NO at 4+ cannot win alongside YES at 7+.
        let clash = vec![
            leg(1, "2026-07-22", 4, 0.7, Some(Side::No), Some(true), Some(-21)),
            leg(
                1,
                "2026-07-22",
                7,
                0.26,
                Some(Side::Yes),
                Some(false),
                Some(-21),
            ),
        ];
        assert_eq!(cluster_stats(&clash).contradictory_starts, 1);

        // YES at 4+ with NO at 7+ is a coherent "somewhere in between" bet.
        let range = vec![
            leg(1, "2026-07-22", 4, 0.7, Some(Side::Yes), Some(true), Some(55)),
            leg(1, "2026-07-22", 7, 0.26, Some(Side::No), Some(true), Some(40)),
        ];
        assert_eq!(cluster_stats(&range).contradictory_starts, 0);
    }

    /// The interval must not move when the user simply reopens the tab.
    #[test]
    fn results_are_deterministic_across_runs() {
        let entries: Vec<PaperEntry> = (0..40u32)
            .flat_map(|pitcher| {
                (4..8u16).map(move |threshold| {
                    leg(
                        pitcher,
                        "2026-07-21",
                        threshold,
                        0.45,
                        Some(Side::Yes),
                        Some(pitcher % 2 == 0),
                        Some(if pitcher % 2 == 0 { 55 } else { -45 }),
                    )
                })
            })
            .collect();
        let first = cluster_stats(&entries);
        let second = cluster_stats(&entries);
        let (a, b) = (
            first.mean_start_pnl_cents.unwrap(),
            second.mean_start_pnl_cents.unwrap(),
        );
        assert_eq!(a.point.to_bits(), b.point.to_bits());
        assert_eq!(a.low.unwrap().to_bits(), b.low.unwrap().to_bits());
        assert_eq!(a.high.unwrap().to_bits(), b.high.unwrap().to_bits());
    }

    #[test]
    fn voided_entries_are_excluded_entirely() {
        let mut entries = vec![leg(
            1,
            "2026-07-21",
            6,
            0.5,
            Some(Side::Yes),
            Some(true),
            Some(55),
        )];
        entries[0].voided = true;
        let stats = cluster_stats(&entries);
        assert_eq!(stats.graded_starts, 0);
        assert_eq!(stats.graded_legs, 0);
    }

    /// The central question: is the model a better forecaster than the price?
    #[test]
    fn detects_which_forecaster_was_closer_to_reality() {
        // Model nails it, market is badly wrong: advantage clearly positive.
        let sharp: Vec<PaperEntry> = (0..40u32)
            .flat_map(|pitcher| {
                let hit = pitcher % 2 == 0;
                [
                    quoted(pitcher, 6, if hit { 0.95 } else { 0.05 }, 0.5, hit),
                    quoted(pitcher, 7, if hit { 0.95 } else { 0.05 }, 0.5, hit),
                ]
            })
            .collect();
        let skill = cluster_stats(&sharp).skill.unwrap();
        assert!(skill.model_brier < skill.market_brier);
        assert!(
            skill.advantage.as_ref().unwrap().low.unwrap() > 0.0,
            "a sharp model should beat the price outright: {:?}",
            skill.advantage
        );

        // Reverse the roles: the interval must land entirely below zero.
        let dull: Vec<PaperEntry> = (0..40u32)
            .flat_map(|pitcher| {
                let hit = pitcher % 2 == 0;
                [
                    quoted(pitcher, 6, 0.5, if hit { 0.95 } else { 0.05 }, hit),
                    quoted(pitcher, 7, 0.5, if hit { 0.95 } else { 0.05 }, hit),
                ]
            })
            .collect();
        let beaten = cluster_stats(&dull).skill.unwrap();
        assert!(beaten.advantage.as_ref().unwrap().high.unwrap() < 0.0);
    }

    #[test]
    fn identical_forecasts_show_no_advantage_either_way() {
        let entries: Vec<PaperEntry> = (0..40u32)
            .map(|pitcher| quoted(pitcher, 6, 0.4, 0.4, pitcher % 3 == 0))
            .collect();
        let skill = cluster_stats(&entries).skill.unwrap();
        assert!((skill.model_brier - skill.market_brier).abs() < 1e-9);
        assert!(skill.advantage.unwrap().point.abs() < 1e-9);
    }

    /// A one-sided quote is not a market forecast, but the leg still belongs to
    /// the model's own calibration.
    #[test]
    fn unquoted_legs_leave_the_market_out_but_keep_the_model_in() {
        let mut entries: Vec<PaperEntry> = (0..40u32)
            .map(|pitcher| quoted(pitcher, 6, 0.4, 0.4, pitcher % 3 == 0))
            .collect();
        entries[0].yes_ask = None;
        let stats = cluster_stats(&entries);
        assert_eq!(stats.graded_legs, 40, "every leg still counts for the model");
        assert_eq!(stats.skill.unwrap().legs, 39, "the unquoted leg is excluded");
        let row = &stats.by_threshold[0];
        assert_eq!(row.legs, 40);
        assert!(row.market_probability.is_some());
    }

    #[test]
    fn threshold_rows_account_for_every_graded_leg() {
        let entries: Vec<PaperEntry> = (0..30u32)
            .flat_map(|pitcher| {
                [
                    quoted(pitcher, 5, 0.6, 0.55, true),
                    quoted(pitcher, 6, 0.4, 0.45, false),
                    quoted(pitcher, 7, 0.2, 0.25, false),
                ]
            })
            .collect();
        let stats = cluster_stats(&entries);
        let counted: usize = stats.by_threshold.iter().map(|row| row.legs).sum();
        assert_eq!(counted, stats.graded_legs);
        assert_eq!(stats.by_threshold.len(), 3);
        // Rows arrive in ladder order so the UI can render them directly.
        assert_eq!(stats.by_threshold[0].threshold, 5);
        assert_eq!(stats.by_threshold[2].threshold, 7);
    }

    /// A lower rung asking less than a higher rung bids is free money: buy YES
    /// low and NO high for under 100c and collect at least 100c.
    #[test]
    fn spots_a_risk_free_ladder_but_not_a_normal_one() {
        let mut free = vec![
            leg(1, "2026-07-21", 6, 0.4, None, Some(true), None),
            leg(1, "2026-07-21", 7, 0.3, None, Some(true), None),
        ];
        free[0].yes_ask = Some(30);
        free[1].yes_bid = Some(45);
        assert_eq!(cluster_stats(&free).arbitrage_starts, 1);

        let mut normal = free.clone();
        normal[0].yes_bid = Some(55);
        normal[0].yes_ask = Some(60);
        normal[1].yes_bid = Some(30);
        normal[1].yes_ask = Some(35);
        assert_eq!(cluster_stats(&normal).arbitrage_starts, 0);
    }

    /// A model that is high on one region and low on another reports a
    /// calibration gap of zero, which is exactly why ECE is also reported.
    #[test]
    fn ece_catches_offsetting_errors_the_gap_cancels() {
        let mut entries = Vec::new();
        for pitcher in 0..40u32 {
            // Predicted 0.9, never happens.
            entries.push(leg(
                pitcher,
                "2026-07-21",
                4,
                0.9,
                None,
                Some(false),
                None,
            ));
            // Predicted 0.1, always happens.
            entries.push(leg(pitcher, "2026-07-21", 8, 0.1, None, Some(true), None));
        }
        let stats = cluster_stats(&entries);
        let gap = stats.calibration_gap.unwrap();
        assert!(gap.point.abs() < 1e-9, "gap cancels to zero: {}", gap.point);
        assert!(
            stats.ece.unwrap() > 0.7,
            "ece should expose it: {:?}",
            stats.ece
        );
    }
}
