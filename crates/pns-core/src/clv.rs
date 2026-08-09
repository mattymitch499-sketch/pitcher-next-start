//! Does a forecast *lead* the market price?
//!
//! Every earlier comparison in this crate asked whether a forecaster beats the
//! price the exchange showed just before first pitch, and the answer was no —
//! twice, with confidence. This asks a different question: when the forecast
//! disagrees with an *early* price, does the price subsequently move toward it?
//!
//! That is closing-line value, and it is a weaker claim than accuracy. It does
//! not require beating the closing price; it requires knowing something the book
//! has not absorbed yet. It is also the only version of the claim that survives
//! the earlier results, because a demonstrated lead over a *well calibrated*
//! close converts to expected value at settlement rather than only on paper —
//! and the close being well calibrated is exactly what
//! [`crate::backtest::evaluate_against_market`] established.
//!
//! Three statistics per anchor, all clustered by pitcher-start because the
//! rungs of one start resolve together:
//!
//! 1. **Drift slope.** Regress `close - anchor` on `forecast - anchor`. A slope
//!    of 1 means the price ends up exactly where the forecast said; 0 means the
//!    forecast carried no information the market lacked. This is the primary
//!    number.
//! 2. **Directional hit rate**, over legs where the forecast disagrees by at
//!    least [`SIGNAL_THRESHOLD`] — a sanity check that the slope is not driven
//!    by a handful of extreme disagreements.
//! 3. **Fee-aware settlement P&L**, taking the forecast's side at the anchor's
//!    *tradeable* quote (ask for YES, `100 - bid` for NO) and settling against
//!    the box score. The slope can be positive while the spread eats it, so the
//!    slope alone is not permission to trade.

use crate::paper::{Liquidity, kalshi_fee_cents};
use crate::paper_stats::{self, Interval};
use chrono::NaiveDate;
use serde::Serialize;

/// Minimum forecast-versus-price disagreement before a leg counts toward the
/// hit rate or gets a simulated position. Matches `paper::DEFAULT_EDGE_THRESHOLD`
/// so this study and the live path mean the same thing by "disagrees".
pub const SIGNAL_THRESHOLD: f64 = 0.05;

/// One rung of one start, priced at one anchor time.
#[derive(Debug, Clone)]
pub struct AnchorLeg {
    pub threshold: u16,
    /// Forecast `P(K >= threshold)`, 0..1.
    pub forecast: f64,
    /// Anchor quote in whole cents, as it stood that far before first pitch.
    pub anchor_yes_bid: u16,
    pub anchor_yes_ask: u16,
    /// Mid of the last usable quote before first pitch, 0..1.
    pub close_mid: f64,
    /// Whether YES settled, from the box score.
    pub settled_yes: bool,
}

impl AnchorLeg {
    /// Anchor mid, 0..1.
    pub fn anchor_mid(&self) -> f64 {
        f64::from(self.anchor_yes_bid + self.anchor_yes_ask) / 200.0
    }

    /// How far the forecast sits from the anchor price. Positive means the
    /// forecast is higher than the market.
    fn signal(&self) -> f64 {
        self.forecast - self.anchor_mid()
    }

    /// How far the price actually travelled between anchor and close.
    fn drift(&self) -> f64 {
        self.close_mid - self.anchor_mid()
    }

    /// Fee-aware settlement profit in cents from taking the forecast's side at
    /// the anchor's tradeable quote. `None` when the forecast does not disagree
    /// enough to act on.
    ///
    /// Crossing the spread is deliberate: a study that fills at the mid would
    /// report an edge that no order could have captured.
    fn realised_cents(&self) -> Option<f64> {
        let signal = self.signal();
        if signal.abs() < SIGNAL_THRESHOLD {
            return None;
        }
        let buy_yes = signal > 0.0;
        let price = if buy_yes {
            self.anchor_yes_ask
        } else {
            100u16.checked_sub(self.anchor_yes_bid)?
        };
        if !(1..=99).contains(&price) {
            return None;
        }
        let won = buy_yes == self.settled_yes;
        let gross = if won {
            f64::from(100 - price)
        } else {
            -f64::from(price)
        };
        Some(gross - f64::from(kalshi_fee_cents(price, Liquidity::Taker)))
    }
}

/// Every priced rung of one pitcher-start. The resampling unit.
#[derive(Debug, Clone)]
pub struct AnchorStart {
    pub pitcher_id: u32,
    pub game_date: NaiveDate,
    pub legs: Vec<AnchorLeg>,
}

/// What one anchor produced.
#[derive(Debug, Clone, Serialize)]
pub struct AnchorResult {
    /// Hours before first pitch this anchor sits at.
    pub hours_before: f64,
    pub legs: usize,
    pub starts: usize,
    /// Slope of `close - anchor` on `forecast - anchor`. The primary statistic:
    /// above zero means the price moved toward the forecast.
    pub drift_slope: Option<Interval>,
    /// Share of disagreeing legs where the price moved the way the forecast
    /// implied. 0.5 is coin-flip.
    pub hit_rate: Option<Interval>,
    /// Fee-aware settlement P&L per contract, in cents.
    pub pnl_per_contract: Option<Interval>,
    /// Legs clearing [`SIGNAL_THRESHOLD`] — the ones the hit rate and P&L use.
    pub signal_legs: usize,
    /// Mean absolute forecast-versus-price gap, in probability points.
    pub mean_abs_signal: f64,
    /// Mean absolute price movement between anchor and close. If this is ~0 the
    /// book never moved and no slope is measurable regardless of the forecast.
    pub mean_abs_drift: f64,
}

fn legs_of<'a>(clusters: &[&'a AnchorStart]) -> impl Iterator<Item = &'a AnchorLeg> {
    clusters
        .iter()
        .flat_map(|cluster| cluster.legs.iter())
        .collect::<Vec<_>>()
        .into_iter()
}

/// Ordinary least squares slope of drift on signal.
///
/// No intercept term is fitted: a market that is unbiased at the anchor has no
/// systematic drift to absorb, and forcing the line through the origin makes the
/// slope answer exactly the question asked — how much of the forecast's
/// disagreement does the price later travel?
fn drift_slope(clusters: &[&AnchorStart]) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for leg in legs_of(clusters) {
        numerator += leg.signal() * leg.drift();
        denominator += leg.signal() * leg.signal();
    }
    (denominator > 1e-12).then(|| numerator / denominator)
}

fn hit_rate(clusters: &[&AnchorStart]) -> Option<f64> {
    let mut hits = 0.0;
    let mut count = 0usize;
    for leg in legs_of(clusters) {
        let signal = leg.signal();
        if signal.abs() < SIGNAL_THRESHOLD {
            continue;
        }
        // A price that did not move is half a hit rather than a miss: counting
        // it as a miss would penalise a forecast for an illiquid book.
        let drift = leg.drift();
        hits += if drift == 0.0 {
            0.5
        } else {
            f64::from(u8::from(drift.signum() == signal.signum()))
        };
        count += 1;
    }
    (count > 0).then(|| hits / count as f64)
}

fn pnl_per_contract(clusters: &[&AnchorStart]) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for leg in legs_of(clusters) {
        if let Some(cents) = leg.realised_cents() {
            total += cents;
            count += 1;
        }
    }
    (count > 0).then(|| total / count as f64)
}

/// Scores one anchor over every start priced at it.
///
/// One bootstrap serves all three statistics, resampling whole starts: the rungs
/// of a start move together, so resampling legs would treat one pitcher's night
/// as many independent observations — the same design effect
/// [`crate::paper_stats`] measured at 2.05x on the live log.
pub fn evaluate_anchor(hours_before: f64, starts: &[AnchorStart]) -> AnchorResult {
    let clusters: Vec<&AnchorStart> = starts.iter().filter(|s| !s.legs.is_empty()).collect();
    let owned: Vec<AnchorStart> = clusters.iter().map(|c| (*c).clone()).collect();
    let all_legs: Vec<&AnchorLeg> = legs_of(&clusters).collect();

    let signal_legs = all_legs
        .iter()
        .filter(|leg| leg.signal().abs() >= SIGNAL_THRESHOLD)
        .count();
    let mean_abs_signal = if all_legs.is_empty() {
        0.0
    } else {
        all_legs.iter().map(|leg| leg.signal().abs()).sum::<f64>() / all_legs.len() as f64
    };
    let mean_abs_drift = if all_legs.is_empty() {
        0.0
    } else {
        all_legs.iter().map(|leg| leg.drift().abs()).sum::<f64>() / all_legs.len() as f64
    };

    let mut slope_replicates = Vec::with_capacity(paper_stats::BOOTSTRAP_DRAWS);
    let mut hit_replicates = Vec::with_capacity(paper_stats::BOOTSTRAP_DRAWS);
    let mut pnl_replicates = Vec::with_capacity(paper_stats::BOOTSTRAP_DRAWS);
    if !owned.is_empty() {
        let mut rng = paper_stats::Rng::new(paper_stats::SEED);
        let mut draw: Vec<&AnchorStart> = Vec::with_capacity(owned.len());
        for _ in 0..paper_stats::BOOTSTRAP_DRAWS {
            draw.clear();
            for _ in 0..owned.len() {
                draw.push(&owned[rng.below(owned.len())]);
            }
            if let Some(value) = drift_slope(&draw) {
                slope_replicates.push(value);
            }
            if let Some(value) = hit_rate(&draw) {
                hit_replicates.push(value);
            }
            if let Some(value) = pnl_per_contract(&draw) {
                pnl_replicates.push(value);
            }
        }
    }

    AnchorResult {
        hours_before,
        legs: all_legs.len(),
        starts: owned.len(),
        drift_slope: drift_slope(&clusters).and_then(|point| {
            paper_stats::bca_interval(point, &slope_replicates, &owned, drift_slope)
        }),
        hit_rate: hit_rate(&clusters)
            .and_then(|point| paper_stats::bca_interval(point, &hit_replicates, &owned, hit_rate)),
        pnl_per_contract: pnl_per_contract(&clusters).and_then(|point| {
            paper_stats::bca_interval(point, &pnl_replicates, &owned, pnl_per_contract)
        }),
        signal_legs,
        mean_abs_signal,
        mean_abs_drift,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg(threshold: u16, forecast: f64, bid: u16, ask: u16, close: f64, yes: bool) -> AnchorLeg {
        AnchorLeg {
            threshold,
            forecast,
            anchor_yes_bid: bid,
            anchor_yes_ask: ask,
            close_mid: close,
            settled_yes: yes,
        }
    }

    fn starts(build: impl Fn(u32) -> Vec<AnchorLeg>, count: u32) -> Vec<AnchorStart> {
        (0..count)
            .map(|index| AnchorStart {
                pitcher_id: index,
                game_date: NaiveDate::from_ymd_opt(2026, 6, 1).unwrap(),
                legs: build(index),
            })
            .collect()
    }

    /// A market that always closes exactly where the forecast said must give a
    /// slope of 1 — the calibration check for the primary statistic.
    #[test]
    fn a_price_that_lands_on_the_forecast_gives_slope_one() {
        let data = starts(
            |index| {
                let forecast = 0.30 + f64::from(index % 7) * 0.05;
                // Anchor mid 0.50, close lands exactly on the forecast.
                vec![leg(6, forecast, 49, 51, forecast, forecast > 0.5)]
            },
            40,
        );
        let result = evaluate_anchor(6.0, &data);
        let slope = result.drift_slope.unwrap();
        assert!(
            (slope.point - 1.0).abs() < 1e-9,
            "slope was {}",
            slope.point
        );
    }

    /// A market that never moves must give slope 0, and the interval must not
    /// claim an effect.
    #[test]
    fn a_price_that_never_moves_gives_slope_zero() {
        let data = starts(
            |index| {
                let forecast = 0.20 + f64::from(index % 9) * 0.06;
                vec![leg(6, forecast, 49, 51, 0.50, index % 2 == 0)]
            },
            40,
        );
        let result = evaluate_anchor(6.0, &data);
        let slope = result.drift_slope.unwrap();
        assert!(slope.point.abs() < 1e-9, "slope was {}", slope.point);
        assert!(result.mean_abs_drift.abs() < 1e-9);
    }

    /// A price that moves *away* from the forecast has to produce a negative
    /// slope. Without this the study could report "no evidence" for a forecast
    /// that is actively anti-predictive.
    #[test]
    fn a_price_moving_away_gives_a_negative_slope() {
        let data = starts(
            |index| {
                let forecast = 0.70 + f64::from(index % 5) * 0.02;
                // Forecast says up from 0.50; the price goes down instead.
                vec![leg(6, forecast, 49, 51, 0.40, false)]
            },
            40,
        );
        let result = evaluate_anchor(6.0, &data);
        assert!(result.drift_slope.unwrap().point < 0.0);
        assert!(result.hit_rate.unwrap().point < 0.5);
    }

    /// The P&L leg must cross the spread and pay the fee. A YES bought at a 55c
    /// ask that settles YES returns 45c gross, less the 2c taker fee on a 55c
    /// fill: 43c.
    #[test]
    fn realised_pnl_crosses_the_spread_and_pays_the_fee() {
        let winner = leg(6, 0.90, 45, 55, 0.60, true);
        assert_eq!(winner.realised_cents(), Some(43.0));

        // The same fill losing: -55c stake, still owing the 2c fee.
        let loser = leg(6, 0.90, 45, 55, 0.60, false);
        assert_eq!(loser.realised_cents(), Some(-57.0));

        // Below the disagreement threshold there is no position at all.
        let quiet = leg(6, 0.51, 49, 51, 0.50, true);
        assert_eq!(quiet.realised_cents(), None);
    }

    /// A NO position prices at `100 - bid`, not at the ask.
    #[test]
    fn a_no_position_pays_the_other_side_of_the_book() {
        // Forecast 0.10 against a 0.50 mid: buy NO at 100 - 45 = 55c.
        let entry = leg(6, 0.10, 45, 55, 0.30, false);
        // NO wins when YES does not settle: 45c gross less the 2c fee.
        assert_eq!(entry.realised_cents(), Some(43.0));
    }

    /// Legs that do not disagree enough are excluded from the hit rate, so a
    /// pile of near-mid quotes cannot dilute the signal toward 0.5.
    #[test]
    fn only_disagreeing_legs_count_toward_the_hit_rate() {
        let data = starts(
            |_| {
                vec![
                    // Disagrees by 20pp and the price follows.
                    leg(6, 0.70, 49, 51, 0.60, true),
                    // Agrees with the market; must be ignored entirely.
                    leg(7, 0.50, 49, 51, 0.20, false),
                ]
            },
            40,
        );
        let result = evaluate_anchor(6.0, &data);
        assert_eq!(result.legs, 80);
        assert_eq!(result.signal_legs, 40, "only the disagreeing leg counts");
        assert!((result.hit_rate.unwrap().point - 1.0).abs() < 1e-9);
    }

    /// Determinism: the bootstrap is seeded, so two runs must agree exactly.
    #[test]
    fn results_are_deterministic_across_runs() {
        let build = |index: u32| {
            let forecast = 0.25 + f64::from(index % 11) * 0.05;
            vec![leg(
                6,
                forecast,
                48,
                52,
                0.45 + f64::from(index % 3) * 0.05,
                index.is_multiple_of(2),
            )]
        };
        let data = starts(build, 60);
        let first = evaluate_anchor(6.0, &data);
        let second = evaluate_anchor(6.0, &data);
        assert_eq!(
            first.drift_slope.unwrap().low,
            second.drift_slope.unwrap().low
        );
        assert_eq!(
            first.pnl_per_contract.unwrap().point,
            second.pnl_per_contract.unwrap().point
        );
    }
}
