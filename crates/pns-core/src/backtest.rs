//! Leakage-safe replay of historical starts, and the scoring that grades two
//! strikeout forecasters against what actually happened.
//!
//! The v2 model prices a start by multiplying a K-rate prediction by a
//! batters-faced prediction; [`crate::sim`] prices it by simulating plate
//! appearances against the real lineup and pulling the starter with a fitted
//! hazard. Both emit a PMF over exact strikeout counts, so both can be asked the
//! same eleven questions about the same start — P(K >= 2) through P(K >= 12) —
//! and graded against the same answer.
//!
//! Two properties make that grading worth anything, and both live here rather
//! than in the example that prints the table:
//!
//! - **Nothing dated on or after a start informs that start.** The feature row
//!   comes from the pitcher's five most recent starts *strictly before* the
//!   date, exactly the window [`crate::features::historical_rows`] trains on,
//!   and the batter K-rate fold absorbs a date's box-score lines only once every
//!   start on that date has been served. [`replay`] is the whole surface where
//!   that could go wrong, which is why it is library code with a regression test
//!   (`inputs_are_identical_with_and_without_future_rows`) rather than a script.
//! - **A start is one observation, not eleven.** The eleven rungs of one start
//!   resolve together: a pitcher who fans four loses every rung above four at
//!   once. Intervals therefore come from a bootstrap that resamples *starts*,
//!   reusing [`crate::paper_stats::bca_interval`] — the same estimator the
//!   paper-trading panel reports — rather than a second implementation that
//!   would drift from it.
//!
//! [`replay`] deliberately does not know what a model is. It resolves inputs;
//! the caller scores them with whatever artifact it likes and hands the
//! resulting PMFs back to [`evaluate`]. That split is what lets the leakage test
//! run on a handful of synthetic starts with no artifact, no files and no
//! network.

use crate::data::batters::{BatterGameLine, BatterKRates};
use crate::data::types::PitcherStartLog;
use crate::features::{self, FeatureRow};
use crate::model::MAX_STRIKEOUTS;
use crate::paper_stats::{self, Interval};
use crate::sim::StartInputs;
use crate::{PnsError, Result};
use chrono::NaiveDate;
use serde::Serialize;
use std::collections::HashMap;

/// Rolling history the feature builder uses. Must match
/// [`crate::model::artifact::DEFAULT_HISTORY_WINDOW`], or the replayed row would
/// not be the row the artifact was trained to read.
pub const HISTORY_WINDOW: usize = 5;

/// Ladder rungs the two forecasters are graded on, as `P(K >= t)`. The low end
/// starts at 2 because a rung nobody ever fails carries no information; the high
/// end stops at 12 because starts past that are too rare in a ten-week window to
/// read a frequency off.
pub const RUNGS: std::ops::RangeInclusive<u16> = 2..=12;

/// Why a start in the window could not be replayed. Every skipped start lands in
/// exactly one of these and is counted — a backtest that quietly drops rows is
/// a backtest of whichever rows happened to survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// Fewer than [`HISTORY_WINDOW`] starts strictly before this one, so the
    /// feature row would not be the row the model was trained on.
    ShortHistory,
    /// No `pitch_hand`, so neither the batter split nor the league rate can be
    /// resolved. Guessing a hand pollutes the lineup term in both directions.
    UnknownHand,
    /// Two starts by one pitcher on one date. The feature row is keyed by
    /// (pitcher, date), so the pair is ambiguous and both are dropped.
    AmbiguousDate,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SkippedStart {
    pub pitcher_id: u32,
    pub game_pk: u32,
    pub game_date: NaiveDate,
    pub reason: SkipReason,
}

/// One start, resolved into everything a forecaster needs and the truth it will
/// be graded against.
#[derive(Debug, Clone)]
pub struct ReplayCase {
    pub pitcher_id: u32,
    pub game_pk: u32,
    pub game_date: NaiveDate,
    pub hand: char,
    /// The five-start rolling window, built by the same code path that built the
    /// training set.
    pub row: FeatureRow,
    /// Lineup and league K rates as of this date, from
    /// [`StartInputs::from_lineup`]. The two mean fields are placeholders until
    /// a model fills them in through [`ReplayCase::inputs`].
    pub lineup: StartInputs,
    /// False when the dataset carries no starting nine for this start, in which
    /// case every slot fell back to the league rate and the simulator has no
    /// lineup-specific signal to add.
    pub lineup_known: bool,
    pub actual_strikeouts: u16,
    pub actual_batters_faced: u16,
}

impl ReplayCase {
    /// The simulator's inputs once a model has supplied the two predictions.
    pub fn inputs(&self, k_rate_mean: f64, batters_faced_mean: f64) -> StartInputs {
        StartInputs {
            k_rate_mean,
            batters_faced_mean,
            ..self.lineup.clone()
        }
    }

    /// Deterministic per-start seed. Derived from the start's identity rather
    /// than its position, so a start draws the same simulated season whatever
    /// else is in the replay — including a rerun over a different date window.
    pub fn seed(&self) -> u64 {
        u64::from(self.pitcher_id).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ u64::from(self.game_pk).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Replay {
    pub cases: Vec<ReplayCase>,
    pub skipped: Vec<SkippedStart>,
}

impl Replay {
    pub fn skipped_by_reason(&self, reason: SkipReason) -> usize {
        self.skipped
            .iter()
            .filter(|skip| skip.reason == reason)
            .count()
    }
}

/// Resolves every start played in `from..=through` into a [`ReplayCase`].
///
/// `starts` and `lines` may span any period; only the window is replayed, and
/// rows outside it exist purely to supply history. Rows dated after a case can
/// therefore be present in the input and must not change that case — the
/// property [`tests::inputs_are_identical_with_and_without_future_rows`] pins.
pub fn replay(
    starts: &[PitcherStartLog],
    lines: &[BatterGameLine],
    from: NaiveDate,
    through: NaiveDate,
) -> Result<Replay> {
    if from > through {
        return Err(PnsError::Feature(format!(
            "backtest window {from}..{through} runs backwards"
        )));
    }

    // Feature rows come from the training path verbatim. `historical_rows`
    // groups by pitcher and reads each group in slice order, so the slice has to
    // be in date order within a pitcher before it is handed over.
    let mut ordered = starts.to_vec();
    ordered.sort_by_key(|start| (start.pitcher_id, start.game_date, start.game_pk));
    let rows = features::historical_rows(&ordered, HISTORY_WINDOW)?;
    let mut by_key: HashMap<(u32, NaiveDate), &FeatureRow> = HashMap::new();
    let mut ambiguous: std::collections::HashSet<(u32, NaiveDate)> =
        std::collections::HashSet::new();
    for row in &rows {
        if by_key.insert((row.pitcher_id, row.as_of), row).is_some() {
            ambiguous.insert((row.pitcher_id, row.as_of));
        }
    }
    // A pitcher with two starts on one date is ambiguous even when only one of
    // them produced a feature row, so the raw starts are counted too.
    let mut seen: std::collections::HashSet<(u32, NaiveDate)> = std::collections::HashSet::new();
    for start in &ordered {
        if !seen.insert((start.pitcher_id, start.game_date)) {
            ambiguous.insert((start.pitcher_id, start.game_date));
        }
    }

    let mut by_date: Vec<&PitcherStartLog> = ordered.iter().collect();
    by_date.sort_by_key(|start| (start.game_date, start.pitcher_id, start.game_pk));
    let mut folded: Vec<&BatterGameLine> = lines.iter().collect();
    folded.sort_by_key(|line| line.game_date);

    let mut rates = BatterKRates::default();
    let mut cursor = 0usize;
    let mut replay = Replay::default();
    for start in by_date {
        // Absorb every line played strictly before this start's date, and not
        // one played on it: a game must never inform its own projection.
        while cursor < folded.len() && folded[cursor].game_date < start.game_date {
            rates.absorb(folded[cursor]);
            cursor += 1;
        }
        if start.game_date < from || start.game_date > through {
            continue;
        }
        let key = (start.pitcher_id, start.game_date);
        let skip = |reason| SkippedStart {
            pitcher_id: start.pitcher_id,
            game_pk: start.game_pk,
            game_date: start.game_date,
            reason,
        };
        if ambiguous.contains(&key) {
            replay.skipped.push(skip(SkipReason::AmbiguousDate));
            continue;
        }
        let Some(row) = by_key.get(&key) else {
            replay.skipped.push(skip(SkipReason::ShortHistory));
            continue;
        };
        if row.starts_used < HISTORY_WINDOW {
            replay.skipped.push(skip(SkipReason::ShortHistory));
            continue;
        }
        let Some(hand) = start.pitch_hand else {
            replay.skipped.push(skip(SkipReason::UnknownHand));
            continue;
        };
        replay.cases.push(ReplayCase {
            pitcher_id: start.pitcher_id,
            game_pk: start.game_pk,
            game_date: start.game_date,
            hand,
            row: (*row).clone(),
            lineup: StartInputs::from_lineup(
                f64::NAN,
                f64::NAN,
                &rates,
                &start.opponent_batters,
                hand,
            ),
            lineup_known: !start.opponent_batters.is_empty(),
            actual_strikeouts: start.strikeouts,
            actual_batters_faced: start.batters_faced,
        });
    }
    Ok(replay)
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Two competing PMFs for one start, plus the outcome. Both PMFs are indexed by
/// exact strikeout count and are expected to sum to 1.
#[derive(Debug, Clone)]
pub struct StartForecast {
    pub pitcher_id: u32,
    pub game_date: NaiveDate,
    pub model_pmf: Vec<f64>,
    pub sim_pmf: Vec<f64>,
    pub actual_strikeouts: u16,
}

/// One start as the bootstrap sees it: an indivisible block of correlated rungs.
#[derive(Debug, Clone)]
struct StartCluster {
    /// `(threshold, model P(K >= t), sim P(K >= t), outcome)`.
    legs: Vec<(u16, f64, f64, bool)>,
    model_log: f64,
    sim_log: f64,
}

/// One rung of the survival ladder. `observed_rate` is the frequency that
/// actually occurred, so a forecaster sitting above it at every rung is
/// systematically long strikeouts.
#[derive(Debug, Clone, Serialize)]
pub struct RungRow {
    pub threshold: u16,
    pub starts: usize,
    pub model_probability: f64,
    pub sim_probability: f64,
    pub observed_rate: f64,
    pub model_brier: f64,
    pub sim_brier: f64,
}

/// Pooled scores for one forecaster over every rung of every start.
#[derive(Debug, Clone, Serialize)]
pub struct ForecastScores {
    /// Mean Brier over the pooled rungs. Lower is better.
    pub pooled_brier: f64,
    /// Mean of `-ln P(K >= t)` style binary log loss over the pooled rungs.
    pub rung_log_loss: f64,
    /// Expected calibration error over the pooled rungs, 10 fixed-width bins.
    pub ece: f64,
    /// Mean `ln P(K = actual)` on the exact count. Higher (less negative) is
    /// better. This grades the whole distribution rather than the ladder.
    pub exact_log_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    pub starts: usize,
    pub legs: usize,
    pub rungs: Vec<RungRow>,
    pub model: ForecastScores,
    pub sim: ForecastScores,
    /// `sim_brier - model_brier`, pooled over rungs, with a 95% BCa interval
    /// from a bootstrap that resamples starts. Negative means the simulator
    /// forecast better. An interval straddling zero means the window cannot
    /// separate them.
    pub brier_difference: Option<Interval>,
}

/// Survival probability `P(K >= threshold)` from a PMF over exact counts.
fn survival(pmf: &[f64], threshold: u16) -> f64 {
    pmf.iter()
        .skip(threshold as usize)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

fn exact_log(pmf: &[f64], strikeouts: u16) -> f64 {
    let index = (strikeouts as usize).min(MAX_STRIKEOUTS);
    pmf.get(index).copied().unwrap_or(0.0).max(1e-12).ln()
}

/// Pooled `sim - model` Brier over a set of starts. Positive means the model
/// won. `None` for a resample that happens to contain no legs.
fn brier_difference(clusters: &[&StartCluster]) -> Option<f64> {
    let (mut difference, mut count) = (0.0, 0usize);
    for cluster in clusters {
        for (_, model, sim, outcome) in &cluster.legs {
            let actual = f64::from(u8::from(*outcome));
            difference += (sim - actual).powi(2) - (model - actual).powi(2);
            count += 1;
        }
    }
    (count > 0).then(|| difference / count as f64)
}

/// Grades both forecasters on the ladder and on the exact count.
pub fn evaluate(forecasts: &[StartForecast]) -> Evaluation {
    let clusters: Vec<StartCluster> = forecasts
        .iter()
        .map(|forecast| StartCluster {
            legs: RUNGS
                .map(|threshold| {
                    (
                        threshold,
                        survival(&forecast.model_pmf, threshold),
                        survival(&forecast.sim_pmf, threshold),
                        forecast.actual_strikeouts >= threshold,
                    )
                })
                .collect(),
            model_log: exact_log(&forecast.model_pmf, forecast.actual_strikeouts),
            sim_log: exact_log(&forecast.sim_pmf, forecast.actual_strikeouts),
        })
        .collect();
    let refs: Vec<&StartCluster> = clusters.iter().collect();
    let starts = clusters.len();
    let legs: usize = clusters.iter().map(|cluster| cluster.legs.len()).sum();

    let rungs: Vec<RungRow> = RUNGS
        .map(|threshold| {
            let matching: Vec<_> = clusters
                .iter()
                .filter_map(|cluster| {
                    cluster
                        .legs
                        .iter()
                        .find(|(rung, ..)| *rung == threshold)
                        .copied()
                })
                .collect();
            let count = matching.len().max(1) as f64;
            let mean = |select: fn(&(u16, f64, f64, bool)) -> f64| {
                matching.iter().map(select).sum::<f64>() / count
            };
            RungRow {
                threshold,
                starts: matching.len(),
                model_probability: mean(|leg| leg.1),
                sim_probability: mean(|leg| leg.2),
                observed_rate: mean(|leg| f64::from(u8::from(leg.3))),
                model_brier: mean(|leg| (leg.1 - f64::from(u8::from(leg.3))).powi(2)),
                sim_brier: mean(|leg| (leg.2 - f64::from(u8::from(leg.3))).powi(2)),
            }
        })
        .collect();

    let pooled = |select: fn(&(u16, f64, f64, bool)) -> f64| -> Vec<(f64, bool)> {
        clusters
            .iter()
            .flat_map(|cluster| cluster.legs.iter())
            .map(|leg| (select(leg), leg.3))
            .collect()
    };
    let scores = |legs: &[(f64, bool)], exact: f64| {
        let (brier, log_loss, ece) = paper_stats::probability_scores(legs);
        ForecastScores {
            pooled_brier: brier.unwrap_or(f64::NAN),
            rung_log_loss: log_loss.unwrap_or(f64::NAN),
            ece: ece.unwrap_or(f64::NAN),
            exact_log_score: exact,
        }
    };
    let mean_exact = |select: fn(&StartCluster) -> f64| {
        clusters.iter().map(select).sum::<f64>() / starts.max(1) as f64
    };

    // One bootstrap over starts, shared by the difference statistic. The draw
    // resamples whole starts because the eleven rungs of a start resolve
    // together; resampling legs would treat one pitcher's bad night as eleven
    // independent pieces of evidence.
    let point = brier_difference(&refs);
    let brier_difference_interval = point.and_then(|point| {
        let mut rng = paper_stats::Rng::new(paper_stats::SEED);
        let mut replicates = Vec::with_capacity(paper_stats::BOOTSTRAP_DRAWS);
        if !clusters.is_empty() {
            let mut draw: Vec<&StartCluster> = Vec::with_capacity(clusters.len());
            for _ in 0..paper_stats::BOOTSTRAP_DRAWS {
                draw.clear();
                for _ in 0..clusters.len() {
                    draw.push(&clusters[rng.below(clusters.len())]);
                }
                if let Some(value) = brier_difference(&draw) {
                    replicates.push(value);
                }
            }
        }
        paper_stats::bca_interval(point, &replicates, &clusters, brier_difference)
    });

    Evaluation {
        starts,
        legs,
        rungs,
        model: scores(
            &pooled(|leg| leg.1),
            mean_exact(|cluster| cluster.model_log),
        ),
        sim: scores(&pooled(|leg| leg.2), mean_exact(|cluster| cluster.sim_log)),
        brier_difference: brier_difference_interval,
    }
}

// ---------------------------------------------------------------------------
// Scoring against the market
// ---------------------------------------------------------------------------

/// One rung Kalshi actually listed for a start, with the pre-first-pitch
/// snapshot mid-price. The market never quotes a full 2..=12 ladder, so these
/// come as a ragged subset per start.
#[derive(Debug, Clone)]
pub struct MarketLeg {
    pub threshold: u16,
    pub probability: f64,
}

/// A replayed start plus whatever legs the market priced for it.
#[derive(Debug, Clone)]
pub struct MarketStartForecast {
    pub forecast: StartForecast,
    pub market_legs: Vec<MarketLeg>,
}

/// `(threshold, model, sim, market, outcome)` on matched legs only.
struct MarketCluster {
    legs: Vec<(u16, f64, f64, f64, bool)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MarketRungRow {
    pub threshold: u16,
    pub legs: usize,
    pub model_probability: f64,
    pub sim_probability: f64,
    pub market_probability: f64,
    pub observed_rate: f64,
    pub model_brier: f64,
    pub sim_brier: f64,
    pub market_brier: f64,
}

/// Ladder scores on the matched-leg subset. No exact-count log score: the
/// market quotes rungs, not a PMF.
#[derive(Debug, Clone, Serialize)]
pub struct MarketScores {
    pub pooled_brier: f64,
    pub rung_log_loss: f64,
    pub ece: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MarketEvaluation {
    /// Starts contributing at least one matched leg.
    pub starts: usize,
    pub legs: usize,
    pub rungs: Vec<MarketRungRow>,
    /// IMPORTANT: model and sim here are scored on the SAME matched legs as the
    /// market — these numbers are not comparable to [`Evaluation`]'s, which pool
    /// the full 2..=12 ladder.
    pub model: MarketScores,
    pub sim: MarketScores,
    pub market: MarketScores,
    /// Negative means the sim forecast the matched legs better than the market.
    pub sim_minus_market_brier: Option<Interval>,
    pub model_minus_market_brier: Option<Interval>,
}

fn paired_brier_difference(
    clusters: &[&MarketCluster],
    ours: fn(&(u16, f64, f64, f64, bool)) -> f64,
) -> Option<f64> {
    let (mut difference, mut count) = (0.0, 0usize);
    for cluster in clusters {
        for leg in &cluster.legs {
            let actual = f64::from(u8::from(leg.4));
            difference += (ours(leg) - actual).powi(2) - (leg.3 - actual).powi(2);
            count += 1;
        }
    }
    (count > 0).then(|| difference / count as f64)
}

/// Grades model, sim, and market on exactly the legs the market listed,
/// clustered by start for both bootstrap intervals.
pub fn evaluate_against_market(starts: &[MarketStartForecast]) -> MarketEvaluation {
    let clusters: Vec<MarketCluster> = starts
        .iter()
        .filter(|start| !start.market_legs.is_empty())
        .map(|start| MarketCluster {
            legs: start
                .market_legs
                .iter()
                .map(|leg| {
                    (
                        leg.threshold,
                        survival(&start.forecast.model_pmf, leg.threshold),
                        survival(&start.forecast.sim_pmf, leg.threshold),
                        leg.probability,
                        start.forecast.actual_strikeouts >= leg.threshold,
                    )
                })
                .collect(),
        })
        .collect();
    let legs: usize = clusters.iter().map(|cluster| cluster.legs.len()).sum();

    let mut thresholds: Vec<u16> = clusters
        .iter()
        .flat_map(|cluster| cluster.legs.iter().map(|leg| leg.0))
        .collect();
    thresholds.sort_unstable();
    thresholds.dedup();
    let rungs: Vec<MarketRungRow> = thresholds
        .into_iter()
        .map(|threshold| {
            let matching: Vec<_> = clusters
                .iter()
                .flat_map(|cluster| cluster.legs.iter())
                .filter(|leg| leg.0 == threshold)
                .collect();
            let count = matching.len().max(1) as f64;
            let mean = |select: fn(&(u16, f64, f64, f64, bool)) -> f64| {
                matching.iter().map(|leg| select(leg)).sum::<f64>() / count
            };
            MarketRungRow {
                threshold,
                legs: matching.len(),
                model_probability: mean(|leg| leg.1),
                sim_probability: mean(|leg| leg.2),
                market_probability: mean(|leg| leg.3),
                observed_rate: mean(|leg| f64::from(u8::from(leg.4))),
                model_brier: mean(|leg| (leg.1 - f64::from(u8::from(leg.4))).powi(2)),
                sim_brier: mean(|leg| (leg.2 - f64::from(u8::from(leg.4))).powi(2)),
                market_brier: mean(|leg| (leg.3 - f64::from(u8::from(leg.4))).powi(2)),
            }
        })
        .collect();

    let pooled = |select: fn(&(u16, f64, f64, f64, bool)) -> f64| -> Vec<(f64, bool)> {
        clusters
            .iter()
            .flat_map(|cluster| cluster.legs.iter())
            .map(|leg| (select(leg), leg.4))
            .collect()
    };
    let scores = |legs: &[(f64, bool)]| {
        let (brier, log_loss, ece) = paper_stats::probability_scores(legs);
        MarketScores {
            pooled_brier: brier.unwrap_or(f64::NAN),
            rung_log_loss: log_loss.unwrap_or(f64::NAN),
            ece: ece.unwrap_or(f64::NAN),
        }
    };

    let interval = |ours: fn(&(u16, f64, f64, f64, bool)) -> f64| -> Option<Interval> {
        let refs: Vec<&MarketCluster> = clusters.iter().collect();
        let point = paired_brier_difference(&refs, ours)?;
        let mut rng = paper_stats::Rng::new(paper_stats::SEED);
        let mut replicates = Vec::with_capacity(paper_stats::BOOTSTRAP_DRAWS);
        let mut draw: Vec<&MarketCluster> = Vec::with_capacity(clusters.len());
        for _ in 0..paper_stats::BOOTSTRAP_DRAWS {
            draw.clear();
            for _ in 0..clusters.len() {
                draw.push(&clusters[rng.below(clusters.len())]);
            }
            if let Some(value) = paired_brier_difference(&draw, ours) {
                replicates.push(value);
            }
        }
        paper_stats::bca_interval(point, &replicates, &clusters, |subset| {
            paired_brier_difference(subset, ours)
        })
    };

    MarketEvaluation {
        starts: clusters.len(),
        legs,
        rungs,
        model: scores(&pooled(|leg| leg.1)),
        sim: scores(&pooled(|leg| leg.2)),
        market: scores(&pooled(|leg| leg.3)),
        sim_minus_market_brier: interval(|leg| leg.2),
        model_minus_market_brier: interval(|leg| leg.1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::batters::BatterGameLine;

    fn date(day: i64) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 5, 1).expect("valid fixture date")
            + chrono::Duration::days(day - 1)
    }

    /// A synthetic start with enough shape that the feature row is not constant
    /// across the season — a constant history would make the leakage test pass
    /// for the wrong reason.
    fn hand(pitcher_id: u32) -> char {
        if pitcher_id.is_multiple_of(2) {
            'R'
        } else {
            'L'
        }
    }

    fn start(pitcher_id: u32, day: i64, strikeouts: u16) -> PitcherStartLog {
        PitcherStartLog {
            pitcher_id,
            pitcher_name: format!("Pitcher {pitcher_id}"),
            game_pk: 900_000 + pitcher_id * 100 + day as u32,
            game_date: date(day),
            team_id: 100 + pitcher_id,
            opponent_id: 200 + pitcher_id,
            opponent_name: Some("Fixture Opponent".into()),
            home: day % 2 == 0,
            innings_pitched: 5.0 + (day % 3) as f64 * 0.5,
            strikeouts,
            walks: (day % 4) as u16,
            home_runs: (day % 3) as u16,
            earned_runs: (day % 5) as u16,
            hits: 4 + (day % 4) as u16,
            batters_faced: 22 + (day % 5) as u16,
            pitch_count: Some(85 + (day % 10) as u16),
            ground_ball_pct: Some(0.40 + (day % 5) as f64 * 0.01),
            avg_fb_velo: Some(93.0 + (day % 4) as f64 * 0.4),
            historical_park_factor: Some(0.98 + (day % 6) as f64 * 0.01),
            historical_opponent_offense: Some(0.95 + (day % 7) as f64 * 0.02),
            pitch_hand: Some(hand(pitcher_id)),
            opponent_batters: (0..9).map(|slot| 500 + pitcher_id * 10 + slot).collect(),
            // Deterministic, and varied enough that the whiff features are not
            // constant across the fixture (a constant would standardize to a
            // zero-variance column and silently drop out of the fit).
            pitches: Some(90 + (day % 5) as u32 * 4),
            swinging_strikes: Some(9 + (day % 3) as u32),
            called_strikes: Some(16 + (day % 4) as u32),
        }
    }

    /// Two pitchers, a start every fifth day, plus batter lines on the same
    /// dates so the K-rate fold has something to move.
    fn fixture() -> (Vec<PitcherStartLog>, Vec<BatterGameLine>) {
        let mut starts = Vec::new();
        let mut lines = Vec::new();
        for pitcher_id in [1u32, 2] {
            for index in 0..12i64 {
                let day = 1 + index * 5;
                starts.push(start(pitcher_id, day, (index % 9) as u16 + 2));
                for slot in 0..9u32 {
                    lines.push(BatterGameLine {
                        game_date: date(day),
                        batter_id: 500 + pitcher_id * 10 + slot,
                        strikeouts: ((index + slot as i64) % 4) as u16,
                        plate_appearances: 4,
                        opposing_hand: hand(pitcher_id),
                    });
                }
            }
        }
        (starts, lines)
    }

    /// The single most important test in the backtest: a start's inputs must not
    /// move when data it could not have known about is present in the file.
    #[test]
    fn inputs_are_identical_with_and_without_future_rows() {
        let (starts, lines) = fixture();
        let boundary = date(31);
        let from = date(6);

        let with_future = replay(&starts, &lines, from, boundary).unwrap();
        let truncated_starts: Vec<_> = starts
            .iter()
            .filter(|start| start.game_date <= boundary)
            .cloned()
            .collect();
        let truncated_lines: Vec<_> = lines
            .iter()
            .filter(|line| line.game_date <= boundary)
            .copied()
            .collect();
        let without_future = replay(&truncated_starts, &truncated_lines, from, boundary).unwrap();

        assert!(
            !with_future.cases.is_empty(),
            "the fixture must actually produce cases"
        );
        assert_eq!(with_future.cases.len(), without_future.cases.len());
        assert!(
            starts.iter().any(|start| start.game_date > boundary),
            "the fixture must actually contain future rows to hide"
        );

        for (full, cut) in with_future.cases.iter().zip(&without_future.cases) {
            assert_eq!(
                (full.pitcher_id, full.game_pk),
                (cut.pitcher_id, cut.game_pk)
            );
            // Bitwise, not approximate: any drift at all is a leak.
            assert_eq!(
                full.row
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                cut.row
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "feature row for {} on {} moved",
                full.pitcher_id,
                full.game_date
            );
            assert_eq!(full.row.starts_used, cut.row.starts_used);
            assert_eq!(
                full.lineup.league_k_rate.to_bits(),
                cut.lineup.league_k_rate.to_bits(),
                "league K rate for {} moved",
                full.game_date
            );
            assert_eq!(
                full.lineup
                    .lineup_k_rates
                    .iter()
                    .map(|rate| rate.to_bits())
                    .collect::<Vec<_>>(),
                cut.lineup
                    .lineup_k_rates
                    .iter()
                    .map(|rate| rate.to_bits())
                    .collect::<Vec<_>>(),
                "lineup K rates for {} moved",
                full.game_date
            );
            assert_eq!(full.seed(), cut.seed());
        }
    }

    /// The fold must also exclude the start's own date, not merely later dates:
    /// a batter's line from the game being projected is the purest leak there
    /// is, and it would be invisible to the truncation test above.
    #[test]
    fn a_batter_line_from_the_same_day_never_reaches_that_days_lookup() {
        let (starts, lines) = fixture();
        let target = date(56);
        let lineup_bits = |replay: &Replay| -> Vec<u64> {
            replay
                .cases
                .iter()
                .flat_map(|case| case.lineup.lineup_k_rates)
                .map(f64::to_bits)
                .collect()
        };
        let baseline = replay(&starts, &lines, target, target).unwrap();
        assert_eq!(baseline.cases.len(), 2);

        // Nine hitters from pitcher 2's actual lineup striking out every time,
        // against the hand pitcher 2 throws. If any of it leaked, that lineup's
        // rates would jump.
        let poison = |day: i64| -> Vec<BatterGameLine> {
            let mut poisoned = lines.clone();
            poisoned.extend((0..9u32).map(|slot| BatterGameLine {
                game_date: date(day),
                batter_id: 520 + slot,
                strikeouts: 4,
                plate_appearances: 4,
                opposing_hand: 'R',
            }));
            poisoned
        };
        let same_day = replay(&starts, &poison(56), target, target).unwrap();
        assert_eq!(
            lineup_bits(&baseline),
            lineup_bits(&same_day),
            "a line played on the projected date reached that date's lookup"
        );

        // The identical line dated one start earlier must move the rates, or the
        // assertion above would be passing because nothing was absorbed at all.
        let day_before = replay(&starts, &poison(51), target, target).unwrap();
        assert_ne!(
            lineup_bits(&baseline),
            lineup_bits(&day_before),
            "a line from an earlier date was never absorbed"
        );
    }

    #[test]
    fn short_histories_and_unknown_hands_are_counted_not_dropped() {
        let (mut starts, lines) = fixture();
        starts[20].pitch_hand = None;
        let window = replay(&starts, &lines, date(1), date(60)).unwrap();
        // Day 1 through day 21 cannot have five prior starts for either pitcher.
        assert!(window.skipped_by_reason(SkipReason::ShortHistory) >= 8);
        assert_eq!(window.skipped_by_reason(SkipReason::UnknownHand), 1);
        let replayed = window.cases.len() + window.skipped.len();
        let in_window = starts
            .iter()
            .filter(|start| start.game_date >= date(1) && start.game_date <= date(60))
            .count();
        assert_eq!(
            replayed, in_window,
            "every start in the window is accounted for"
        );
        assert!(window.cases.iter().all(|case| case.row.starts_used == 5));
    }

    #[test]
    fn a_doubleheader_is_reported_rather_than_silently_resolved() {
        let (mut starts, lines) = fixture();
        let mut second = starts[11].clone();
        second.game_pk += 1;
        starts.push(second);
        let window = replay(&starts, &lines, date(56), date(56)).unwrap();
        assert_eq!(window.skipped_by_reason(SkipReason::AmbiguousDate), 2);
        assert!(window.cases.iter().all(|case| case.pitcher_id != 1));
    }

    fn pmf_at(counts: &[(usize, f64)]) -> Vec<f64> {
        let mut pmf = vec![0.0; MAX_STRIKEOUTS + 1];
        for (index, probability) in counts {
            pmf[*index] = *probability;
        }
        pmf
    }

    /// A forecaster that is right every time must beat one that is always
    /// wrong, and the interval must exclude zero.
    #[test]
    fn evaluation_separates_a_sharp_forecaster_from_a_blind_one() {
        let forecasts: Vec<_> = (0..80u32)
            .map(|index| {
                let actual = if index % 2 == 0 { 3 } else { 9 };
                StartForecast {
                    pitcher_id: index,
                    game_date: date(1 + i64::from(index)),
                    // Model hedges at the wrong end; the sim nails it.
                    model_pmf: pmf_at(&[(if actual == 3 { 9 } else { 3 }, 1.0)]),
                    sim_pmf: pmf_at(&[(actual as usize, 1.0)]),
                    actual_strikeouts: actual,
                }
            })
            .collect();
        let evaluation = evaluate(&forecasts);
        assert_eq!(evaluation.starts, 80);
        assert_eq!(evaluation.legs, 80 * 11);
        assert!(evaluation.sim.pooled_brier < evaluation.model.pooled_brier);
        assert!(evaluation.sim.exact_log_score > evaluation.model.exact_log_score);
        let difference = evaluation.brier_difference.unwrap();
        assert!(difference.point < 0.0, "sim must score lower Brier");
        assert!(
            difference.high.unwrap() < 0.0,
            "the interval must exclude zero: {difference:?}"
        );
    }

    /// Identical forecasters must show no difference and no spurious interval.
    #[test]
    fn identical_forecasters_show_no_difference() {
        let forecasts: Vec<_> = (0..60u32)
            .map(|index| {
                let pmf = pmf_at(&[(5, 0.5), (6, 0.5)]);
                StartForecast {
                    pitcher_id: index,
                    game_date: date(1 + i64::from(index)),
                    model_pmf: pmf.clone(),
                    sim_pmf: pmf,
                    actual_strikeouts: (index % 10) as u16,
                }
            })
            .collect();
        let evaluation = evaluate(&forecasts);
        let difference = evaluation.brier_difference.unwrap();
        assert!(difference.point.abs() < 1e-12);
        assert!(
            (evaluation.model.pooled_brier - evaluation.sim.pooled_brier).abs() < 1e-12,
            "identical PMFs must score identically"
        );
    }

    /// The ladder must line up with the frequencies it claims to describe.
    #[test]
    fn rung_rows_report_the_frequency_that_actually_occurred() {
        // Every start ends with exactly six strikeouts, so P(K >= t) is 1 for
        // t <= 6 and 0 above it.
        let forecasts: Vec<_> = (0..30u32)
            .map(|index| StartForecast {
                pitcher_id: index,
                game_date: date(1),
                model_pmf: pmf_at(&[(6, 1.0)]),
                sim_pmf: pmf_at(&[(6, 1.0)]),
                actual_strikeouts: 6,
            })
            .collect();
        let evaluation = evaluate(&forecasts);
        for row in &evaluation.rungs {
            let expected = f64::from(u8::from(row.threshold <= 6));
            assert!((row.observed_rate - expected).abs() < 1e-12);
            assert!((row.model_probability - expected).abs() < 1e-9);
            assert!(row.model_brier < 1e-12);
        }
        assert_eq!(evaluation.rungs.len(), 11);
        assert_eq!(evaluation.rungs[0].threshold, 2);
        assert_eq!(evaluation.rungs[10].threshold, 12);
    }

    /// Market scoring runs on the ragged subset of legs Kalshi listed — starts
    /// with no legs drop out, rung rows carry per-rung leg counts, and the
    /// model/sim scores here are subset scores, not full-ladder scores.
    #[test]
    fn market_evaluation_scores_only_the_listed_legs() {
        let forecast = |id: u32, actual: u16| StartForecast {
            pitcher_id: id,
            game_date: date(1 + i64::from(id)),
            model_pmf: pmf_at(&[(5, 1.0)]),
            sim_pmf: pmf_at(&[(actual as usize, 1.0)]),
            actual_strikeouts: actual,
        };
        let starts = vec![
            MarketStartForecast {
                forecast: forecast(1, 7),
                market_legs: vec![
                    MarketLeg {
                        threshold: 5,
                        probability: 0.8,
                    },
                    MarketLeg {
                        threshold: 7,
                        probability: 0.4,
                    },
                ],
            },
            MarketStartForecast {
                forecast: forecast(2, 3),
                market_legs: vec![MarketLeg {
                    threshold: 5,
                    probability: 0.3,
                }],
            },
            // Kalshi never listed this start: it must not count anywhere.
            MarketStartForecast {
                forecast: forecast(3, 12),
                market_legs: vec![],
            },
        ];
        let evaluation = evaluate_against_market(&starts);
        assert_eq!(evaluation.starts, 2);
        assert_eq!(evaluation.legs, 3);
        assert_eq!(evaluation.rungs.len(), 2);
        assert_eq!(evaluation.rungs[0].threshold, 5);
        assert_eq!(evaluation.rungs[0].legs, 2);
        assert_eq!(evaluation.rungs[1].threshold, 7);
        assert_eq!(evaluation.rungs[1].legs, 1);

        // Outcomes: start 1 clears 5 and 7; start 2 misses 5.
        // Market Briers: (0.8-1)^2, (0.4-1)^2, (0.3-0)^2 -> mean 0.163333...
        assert!((evaluation.market.pooled_brier - (0.04 + 0.36 + 0.09) / 3.0).abs() < 1e-12);
        // The sim PMF concentrates on the actual count, so it is perfect on
        // every listed leg; the market is not. The paired interval must favor
        // the sim on its point estimate.
        assert!(evaluation.sim.pooled_brier < 1e-12);
        let difference = evaluation.sim_minus_market_brier.unwrap();
        assert!(difference.point < 0.0);
        // The model's subset Brier uses ONLY the three listed legs: model says
        // P(K>=5)=1, P(K>=7)=0 -> (1-1)^2 + (0-1)^2 + (1-0)^2 over legs.
        assert!((evaluation.model.pooled_brier - (0.0 + 1.0 + 1.0) / 3.0).abs() < 1e-12);
    }
}
