//! Paper trading against Kalshi's per-pitcher strikeout markets.
//!
//! Every matched market is logged with the model's probability so calibration
//! can be tracked even when no bet is taken. A simulated one-contract position
//! is recorded only when the model disagrees with a live quote by at least
//! [`DEFAULT_EDGE_THRESHOLD`]. No orders are ever sent anywhere.

use crate::data::kalshi::{self, KalshiClient, KalshiMarket};
use crate::{DecisionCard, ModelInfo, PnsError, Result};
use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Minimum model-versus-market probability gap before a paper position opens.
pub const DEFAULT_EDGE_THRESHOLD: f64 = 0.05;
/// Quotes wider than this are treated as no market (too thin to price).
const MAX_SPREAD_CENTS: u16 = 30;
/// Days to keep retrying a start with no result before concluding the
/// pitcher was scratched (no game, no reschedule found) and voiding the entry.
const SCRATCH_GRACE_DAYS: i64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Yes,
    No,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperEntry {
    pub market_ticker: String,
    pub event_date: String,
    pub pitcher_id: u32,
    pub pitcher_name: String,
    /// YES settles when strikeouts >= threshold.
    pub threshold: u16,
    pub model_probability: f64,
    pub yes_bid: Option<u16>,
    pub yes_ask: Option<u16>,
    pub side: Option<Side>,
    /// Cents paid per one-dollar contract for the chosen side.
    pub entry_price_cents: Option<u16>,
    pub edge: Option<f64>,
    pub recorded_at: String,
    pub actual_strikeouts: Option<f64>,
    pub result_yes: Option<bool>,
    pub profit_cents: Option<i32>,
    /// True when the pitcher never started (scratched, no reschedule found)
    /// and this entry has been excluded from grading and calibration.
    #[serde(default)]
    pub voided: bool,
    /// The model's mean and spread for this start. Recorded so calibration can
    /// later be binned by how far into the tail a threshold sits —
    /// `(threshold - mean) / sd` — which `model_probability` alone cannot
    /// separate, since a low probability means either a high threshold on a
    /// normal pitcher or a moderate one on a weak pitcher. Rows written before
    /// this field existed read back as `None`.
    #[serde(default)]
    pub model_expected_strikeouts: Option<f64>,
    #[serde(default)]
    pub model_sd_strikeouts: Option<f64>,
    pub model_artifact_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSummary {
    pub logged_markets: usize,
    pub open_positions: usize,
    pub graded_bets: usize,
    pub wins: usize,
    pub staked_cents: u64,
    pub profit_cents: i64,
    pub roi: Option<f64>,
    pub graded_logged: usize,
    pub mean_model_probability: Option<f64>,
    pub observed_yes_rate: Option<f64>,
    pub last_recorded_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PaperLog {
    pub entries: Vec<PaperEntry>,
    pub summary: PaperSummary,
    /// Cluster-aware view of the same entries. `summary` counts every logged
    /// market as one observation; this one counts pitcher-starts, which is the
    /// unit that actually resolves independently.
    pub clustering: crate::paper_stats::ClusterStats,
}

pub fn log_path(root: impl AsRef<Path>) -> PathBuf {
    root.as_ref().join("outputs/history/paper_trades.csv")
}

pub fn read_log(root: impl AsRef<Path>) -> Result<Vec<PaperEntry>> {
    let path = log_path(root);
    if !path.exists() {
        return Ok(Vec::new());
    }
    csv::Reader::from_path(path)?
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(PnsError::from)
}

/// Lowercased, diacritic-folded, alphanumeric-and-space form used to match
/// Kalshi market subtitles against MLB pitcher names.
fn normalize_name(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len());
    for character in name.chars().flat_map(char::to_lowercase) {
        let folded = match character {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' | 'ø' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ñ' => 'n',
            'ç' => 'c',
            other => other,
        };
        if folded.is_ascii_alphanumeric() {
            normalized.push(folded);
        } else if !normalized.ends_with(' ') && !normalized.is_empty() {
            normalized.push(' ');
        }
    }
    normalized.trim().to_string()
}

/// P(strikeouts >= threshold) from a card's discrete distribution.
fn model_over_probability(card: &DecisionCard, threshold: u16) -> Option<f64> {
    if card.strikeout_probabilities.is_empty() {
        return None;
    }
    Some(
        card.strikeout_probabilities
            .iter()
            .enumerate()
            .filter(|(k, _)| *k >= threshold as usize)
            .map(|(_, probability)| probability)
            .sum(),
    )
}

/// Mean and standard deviation of a card's discrete strikeout distribution.
/// `strikeout_probabilities` is a PMF: index `k` holds P(exactly k strikeouts).
fn model_moments(card: &DecisionCard) -> Option<(f64, f64)> {
    if card.strikeout_probabilities.is_empty() {
        return None;
    }
    let mean: f64 = card
        .strikeout_probabilities
        .iter()
        .enumerate()
        .map(|(k, probability)| k as f64 * probability)
        .sum();
    let variance: f64 = card
        .strikeout_probabilities
        .iter()
        .enumerate()
        .map(|(k, probability)| (k as f64 - mean).powi(2) * probability)
        .sum();
    Some((mean, variance.max(0.0).sqrt()))
}

/// Chooses a paper position when a live two-sided quote disagrees with the
/// model by at least `edge_threshold`. Buying YES costs the ask; buying NO
/// costs 100 minus the bid.
fn decide(
    model_probability: f64,
    yes_bid: Option<u16>,
    yes_ask: Option<u16>,
    edge_threshold: f64,
) -> Option<(Side, u16, f64)> {
    let (bid, ask) = (yes_bid?, yes_ask?);
    if !(1..=99).contains(&bid) || !(1..=99).contains(&ask) || ask < bid {
        return None;
    }
    if ask - bid > MAX_SPREAD_CENTS {
        return None;
    }
    let yes_edge = model_probability - ask as f64 / 100.0;
    let no_edge = bid as f64 / 100.0 - model_probability;
    if yes_edge >= edge_threshold && yes_edge >= no_edge {
        Some((Side::Yes, ask, yes_edge))
    } else if no_edge >= edge_threshold {
        Some((Side::No, 100 - bid, no_edge))
    } else {
        None
    }
}

/// Builds log entries for every market that matches a scored pitcher by name
/// and slate date. Pure so it is testable without the network.
pub(crate) fn match_markets(
    cards: &[DecisionCard],
    markets: &[KalshiMarket],
    model: &ModelInfo,
    recorded_at: &str,
    edge_threshold: f64,
) -> Vec<PaperEntry> {
    let by_name: BTreeMap<String, &DecisionCard> = cards
        .iter()
        .map(|card| (normalize_name(&card.pitcher_name), card))
        .collect();
    let mut entries = Vec::new();
    for market in markets {
        if kalshi::is_closed_status(&market.status) {
            continue;
        }
        let Some((event_date, threshold)) = kalshi::parse_ticker(&market.ticker) else {
            continue;
        };
        let Some(raw_name) = kalshi::market_pitcher_name(market) else {
            continue;
        };
        let Some(card) = by_name.get(&normalize_name(raw_name)) else {
            continue;
        };
        if card.target_game_date != event_date.to_string() {
            continue;
        }
        let Some(model_probability) = model_over_probability(card, threshold) else {
            continue;
        };
        let position = decide(
            model_probability,
            market.yes_bid,
            market.yes_ask,
            edge_threshold,
        );
        let moments = model_moments(card);
        entries.push(PaperEntry {
            market_ticker: market.ticker.clone(),
            event_date: event_date.to_string(),
            pitcher_id: card.pitcher_id,
            pitcher_name: card.pitcher_name.clone(),
            threshold,
            model_probability,
            yes_bid: market.yes_bid,
            yes_ask: market.yes_ask,
            side: position.map(|(side, _, _)| side),
            entry_price_cents: position.map(|(_, price, _)| price),
            edge: position.map(|(_, _, edge)| edge),
            recorded_at: recorded_at.to_string(),
            actual_strikeouts: None,
            result_yes: None,
            profit_cents: None,
            voided: false,
            model_expected_strikeouts: moments.map(|(mean, _)| mean),
            model_sd_strikeouts: moments.map(|(_, sd)| sd),
            model_artifact_id: model.artifact_id.clone(),
        });
    }
    reconcile_start_positions(&mut entries);
    entries
}

/// Clears mutually exclusive positions inside a single start.
///
/// A NO at threshold `a` pays only when K < a, and a YES at `b > a` only when
/// K >= b. The pair can never both win, and whenever `a <= K < b` both lose —
/// so holding them together is strictly worse than holding either alone,
/// whatever the model believes. The leg with the smaller edge behind it is
/// demoted to log-only; the row itself stays so it still counts for
/// calibration.
///
/// The mirror image — YES low with NO high — is deliberately left alone. That
/// is a coherent bet that K lands inside a range, and both legs can win.
fn reconcile_start_positions(entries: &mut [PaperEntry]) {
    let mut by_start: BTreeMap<(u32, String), Vec<usize>> = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        if entry.side.is_some() {
            by_start
                .entry((entry.pitcher_id, entry.event_date.clone()))
                .or_default()
                .push(index);
        }
    }
    for indices in by_start.values() {
        // Each pass demotes exactly one leg, so this terminates.
        while let Some((low, high)) = indices.iter().find_map(|&low| {
            indices.iter().find_map(|&high| {
                (entries[low].side == Some(Side::No)
                    && entries[high].side == Some(Side::Yes)
                    && entries[high].threshold > entries[low].threshold)
                    .then_some((low, high))
            })
        }) {
            let loser = if entries[low].edge.unwrap_or(0.0) < entries[high].edge.unwrap_or(0.0) {
                low
            } else {
                high
            };
            entries[loser].side = None;
            entries[loser].entry_price_cents = None;
            entries[loser].edge = None;
        }
    }
}

/// Applies a completed start to an entry: settles the YES question and, when a
/// paper position was taken, the one-contract profit in cents.
pub(crate) fn apply_result(entry: &mut PaperEntry, actual_strikeouts: f64) {
    entry.actual_strikeouts = Some(actual_strikeouts);
    let yes_settled = actual_strikeouts >= entry.threshold as f64;
    entry.result_yes = Some(yes_settled);
    if let (Some(side), Some(price)) = (entry.side, entry.entry_price_cents) {
        let won = match side {
            Side::Yes => yes_settled,
            Side::No => !yes_settled,
        };
        entry.profit_cents = Some(if won {
            100 - price as i32
        } else {
            -(price as i32)
        });
    }
}

pub fn summarize(entries: &[PaperEntry]) -> PaperSummary {
    // Scratched/voided markets never resolved, so they carry no P&L or
    // calibration signal — excluded from every stat below.
    let active: Vec<&PaperEntry> = entries.iter().filter(|entry| !entry.voided).collect();
    let bets: Vec<_> = active
        .iter()
        .filter(|entry| entry.side.is_some())
        .collect();
    let graded_bets: Vec<_> = bets
        .iter()
        .filter(|entry| entry.profit_cents.is_some())
        .collect();
    let graded_logged: Vec<_> = active
        .iter()
        .filter(|entry| entry.result_yes.is_some())
        .collect();
    let staked_cents: u64 = graded_bets
        .iter()
        .filter_map(|entry| entry.entry_price_cents)
        .map(u64::from)
        .sum();
    let profit_cents: i64 = graded_bets
        .iter()
        .filter_map(|entry| entry.profit_cents)
        .map(i64::from)
        .sum();
    PaperSummary {
        logged_markets: active.len(),
        open_positions: bets.len() - graded_bets.len(),
        graded_bets: graded_bets.len(),
        wins: graded_bets
            .iter()
            .filter(|entry| entry.profit_cents.is_some_and(|profit| profit > 0))
            .count(),
        staked_cents,
        profit_cents,
        roi: (staked_cents > 0).then(|| profit_cents as f64 / staked_cents as f64),
        graded_logged: graded_logged.len(),
        mean_model_probability: (!graded_logged.is_empty()).then(|| {
            graded_logged
                .iter()
                .map(|entry| entry.model_probability)
                .sum::<f64>()
                / graded_logged.len() as f64
        }),
        observed_yes_rate: (!graded_logged.is_empty()).then(|| {
            graded_logged
                .iter()
                .filter(|entry| entry.result_yes == Some(true))
                .count() as f64
                / graded_logged.len() as f64
        }),
        last_recorded_at: active.iter().map(|entry| entry.recorded_at.clone()).max(),
    }
}

/// Records today's matched markets. A market whose position is already
/// decided (a paper trade was opened) or whose game is already graded is
/// frozen forever so reruns never rewrite history. Everything else — a
/// market still waiting on a live quote or a real edge — is refreshed with
/// the latest quote and re-checked for a trade on every call.
pub fn record_slate(
    cards: &[DecisionCard],
    markets: &[KalshiMarket],
    model: &ModelInfo,
    root: impl AsRef<Path>,
) -> Result<Vec<PaperEntry>> {
    let mut by_ticker: BTreeMap<String, PaperEntry> = read_log(&root)?
        .into_iter()
        .map(|entry| (entry.market_ticker.clone(), entry))
        .collect();
    let recorded_at = Utc::now().to_rfc3339();
    for entry in match_markets(cards, markets, model, &recorded_at, DEFAULT_EDGE_THRESHOLD) {
        let ticker = entry.market_ticker.clone();
        let locked = by_ticker
            .get(&ticker)
            .is_some_and(|existing| existing.side.is_some() || existing.result_yes.is_some());
        if !locked {
            by_ticker.insert(ticker, entry);
        }
    }
    let entries: Vec<_> = by_ticker.into_values().collect();
    write_log(&log_path(&root), &entries)?;
    Ok(entries)
}

/// Fetches completed starts for any ungraded past entries and settles them.
pub fn grade_log(root: impl AsRef<Path>) -> Result<Vec<PaperEntry>> {
    let mut entries = read_log(&root)?;
    let today = Utc::now().date_naive();
    let pending: std::collections::BTreeSet<(u32, String)> = entries
        .iter()
        .filter(|entry| entry.result_yes.is_none() && !entry.voided)
        .filter(|entry| {
            NaiveDate::parse_from_str(&entry.event_date, "%Y-%m-%d").is_ok_and(|date| date < today)
        })
        .map(|entry| (entry.pitcher_id, entry.event_date.clone()))
        .collect();
    if pending.is_empty() {
        return Ok(entries);
    }
    let client = crate::data::MlbStatsClient::new()?;
    let mut actuals: BTreeMap<(u32, String), f64> = BTreeMap::new();
    let mut scratched: std::collections::BTreeSet<(u32, String)> = std::collections::BTreeSet::new();
    for (pitcher_id, event_date) in pending {
        let Ok(date) = NaiveDate::parse_from_str(&event_date, "%Y-%m-%d") else {
            continue;
        };
        // A rained-out (or otherwise postponed) start never happened on its
        // original date, so follow it to the makeup date once one is known
        // and it isn't still in the future (a same-day makeup is common and
        // its box score can already be final by the time we check).
        let mut lookup_date = date;
        if client.pitcher_starts_range(pitcher_id, date, date).is_err()
            && let Ok(Some(reschedule)) = client.postponed_reschedule_date(pitcher_id, date)
            && reschedule <= today
        {
            lookup_date = reschedule;
        }
        if let Ok(starts) = client.pitcher_starts_range(pitcher_id, lookup_date, lookup_date)
            && let Some(start) = starts.into_iter().find(|start| start.game_date == lookup_date)
        {
            actuals.insert((pitcher_id, event_date), start.strikeouts as f64);
        } else if today.signed_duration_since(date).num_days() >= SCRATCH_GRACE_DAYS {
            // No start on the original date, no (past) reschedule found, and
            // it's been long enough that this isn't just API lag: the
            // pitcher was scratched. Void it so it stops being retried.
            scratched.insert((pitcher_id, event_date));
        }
    }
    for entry in &mut entries {
        if entry.result_yes.is_some() {
            continue;
        }
        let key = (entry.pitcher_id, entry.event_date.clone());
        if let Some(actual) = actuals.get(&key) {
            apply_result(entry, *actual);
        } else if scratched.contains(&key) {
            entry.voided = true;
        }
    }
    write_log(&log_path(&root), &entries)?;
    Ok(entries)
}

/// Full daily update: fetch live Kalshi quotes, record matched markets for the
/// scored slate, grade past entries, and summarize.
pub fn update(
    cards: &[DecisionCard],
    model: &ModelInfo,
    root: impl AsRef<Path>,
) -> Result<PaperLog> {
    let markets = KalshiClient::new()?.strikeout_markets()?;
    record_slate(cards, &markets, model, &root)?;
    let entries = grade_log(&root)?;
    let summary = summarize(&entries);
    let clustering = crate::paper_stats::cluster_stats(&entries);
    Ok(PaperLog {
        entries,
        summary,
        clustering,
    })
}

pub fn status(root: impl AsRef<Path>) -> Result<PaperLog> {
    let entries = read_log(root)?;
    let summary = summarize(&entries);
    let clustering = crate::paper_stats::cluster_stats(&entries);
    Ok(PaperLog {
        entries,
        summary,
        clustering,
    })
}

fn write_log(path: &Path, entries: &[PaperEntry]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    for entry in entries {
        writer.serialize(entry)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Side;
    use crate::data::kalshi::KalshiMarket;

    fn market(ticker: &str, subtitle: &str, bid: Option<u16>, ask: Option<u16>) -> KalshiMarket {
        KalshiMarket {
            ticker: ticker.into(),
            yes_sub_title: subtitle.into(),
            yes_bid: bid,
            yes_ask: ask,
            status: "active".into(),
        }
    }

    #[test]
    fn skips_markets_that_are_no_longer_tradeable() {
        let cards = crate::score_date_fixture("2026-07-21").unwrap();
        let model = crate::ModelInfo {
            artifact_id: "test-artifact".into(),
            schema_version: 2,
            trained_through: chrono::NaiveDate::from_ymd_opt(2026, 7, 16).unwrap(),
            training_rows: 100,
        };
        let mut closed = market(
            "KXMLBKS-26JUL211910AAABBB-AAAFACE1-6",
            "Fixture Ace: 6+",
            Some(40),
            Some(45),
        );
        closed.status = "closed".into();
        let entries = super::match_markets(&cards, &[closed], &model, "now", 0.05);
        assert!(entries.is_empty());
    }

    #[test]
    fn drops_the_weaker_leg_of_a_mutually_exclusive_pair() {
        let entry = |threshold: u16, side: super::Side, edge: f64| super::PaperEntry {
            market_ticker: format!("T-{threshold}"),
            event_date: "2026-07-22".into(),
            pitcher_id: 1,
            pitcher_name: "Test".into(),
            threshold,
            model_probability: 0.5,
            yes_bid: Some(40),
            yes_ask: Some(45),
            side: Some(side),
            entry_price_cents: Some(45),
            edge: Some(edge),
            recorded_at: "now".into(),
            actual_strikeouts: None,
            result_yes: None,
            profit_cents: None,
            voided: false,
            model_expected_strikeouts: Some(6.0),
            model_sd_strikeouts: Some(2.0),
            model_artifact_id: "m".into(),
        };

        // NO at 4+ cannot win alongside YES at 7+; the thinner edge is dropped.
        let mut clashing = vec![
            entry(4, Side::No, 0.052),
            entry(7, Side::Yes, 0.090),
            entry(8, Side::Yes, 0.056),
        ];
        super::reconcile_start_positions(&mut clashing);
        assert_eq!(clashing[0].side, None, "the 5.2pp NO should be demoted");
        assert_eq!(clashing[0].entry_price_cents, None);
        assert_eq!(clashing[0].edge, None);
        assert_eq!(clashing[1].side, Some(Side::Yes), "the 9.0pp YES survives");
        assert_eq!(clashing[2].side, Some(Side::Yes));

        // YES low with NO high is a coherent range bet — both legs can win.
        let mut coherent = vec![entry(4, Side::Yes, 0.06), entry(7, Side::No, 0.06)];
        super::reconcile_start_positions(&mut coherent);
        assert_eq!(coherent[0].side, Some(Side::Yes));
        assert_eq!(coherent[1].side, Some(Side::No));

        // Different pitchers never conflict with each other.
        let mut separate = vec![entry(4, Side::No, 0.05), entry(7, Side::Yes, 0.09)];
        separate[1].pitcher_id = 2;
        super::reconcile_start_positions(&mut separate);
        assert_eq!(separate[0].side, Some(Side::No));
        assert_eq!(separate[1].side, Some(Side::Yes));
    }

    #[test]
    fn normalizes_names_with_diacritics_and_punctuation() {
        assert_eq!(
            super::normalize_name("Cristopher Sánchez"),
            super::normalize_name("cristopher sanchez")
        );
        assert_eq!(super::normalize_name("Luis L. Ortiz"), "luis l ortiz");
    }

    #[test]
    fn decides_sides_only_with_a_real_edge_and_sane_quotes() {
        assert_eq!(
            super::decide(0.55, Some(40), Some(45), 0.05),
            Some((Side::Yes, 45, 0.55 - 0.45))
        );
        assert_eq!(
            super::decide(0.30, Some(40), Some(45), 0.05),
            Some((Side::No, 60, 0.40 - 0.30))
        );
        assert_eq!(super::decide(0.47, Some(40), Some(45), 0.05), None);
        assert_eq!(super::decide(0.60, None, Some(45), 0.05), None);
        assert_eq!(super::decide(0.90, Some(10), Some(45), 0.05), None);
        assert_eq!(super::decide(0.90, Some(0), Some(5), 0.05), None);
    }

    #[test]
    fn grading_pays_the_correct_side() {
        let mut entry = super::PaperEntry {
            market_ticker: "T".into(),
            event_date: "2026-07-20".into(),
            pitcher_id: 1,
            pitcher_name: "Test".into(),
            threshold: 6,
            model_probability: 0.55,
            yes_bid: Some(40),
            yes_ask: Some(45),
            side: Some(Side::Yes),
            entry_price_cents: Some(45),
            edge: Some(0.10),
            recorded_at: "now".into(),
            actual_strikeouts: None,
            result_yes: None,
            profit_cents: None,
            voided: false,
            model_expected_strikeouts: Some(6.0),
            model_sd_strikeouts: Some(2.0),
            model_artifact_id: "m".into(),
        };
        super::apply_result(&mut entry, 7.0);
        assert_eq!(entry.result_yes, Some(true));
        assert_eq!(entry.profit_cents, Some(55));

        entry.profit_cents = None;
        super::apply_result(&mut entry, 5.0);
        assert_eq!(entry.result_yes, Some(false));
        assert_eq!(entry.profit_cents, Some(-45));

        entry.side = Some(Side::No);
        entry.entry_price_cents = Some(60);
        super::apply_result(&mut entry, 5.0);
        assert_eq!(entry.profit_cents, Some(40));
    }

    #[test]
    fn matches_fixture_cards_to_markets_and_records_once() {
        let cards = crate::score_date_fixture("2026-07-21").unwrap();
        let model = crate::ModelInfo {
            artifact_id: "test-artifact".into(),
            schema_version: 2,
            trained_through: chrono::NaiveDate::from_ymd_opt(2026, 7, 16).unwrap(),
            training_rows: 100,
        };
        let markets = vec![
            market(
                "KXMLBKS-26JUL211910AAABBB-AAAFACE1-6",
                "Fixture Ace: 6+",
                Some(40),
                Some(45),
            ),
            market(
                "KXMLBKS-26JUL211910AAABBB-AAAFACE1-9",
                "Fixture Ace: 9+",
                None,
                None,
            ),
            market(
                "KXMLBKS-26JUL221910AAABBB-AAAFACE1-6",
                "Fixture Ace: 6+",
                Some(40),
                Some(45),
            ),
            market(
                "KXMLBKS-26JUL211910AAABBB-CCCSOMEONE9-6",
                "Someone Else: 6+",
                Some(40),
                Some(45),
            ),
        ];
        let entries = super::match_markets(&cards, &markets, &model, "now", 0.05);
        // The wrong-date and unknown-pitcher markets are skipped.
        assert_eq!(entries.len(), 2);
        assert!(
            entries
                .iter()
                .all(|entry| entry.pitcher_name == "Fixture Ace")
        );
        let quoted = entries.iter().find(|entry| entry.threshold == 6).unwrap();
        assert!(quoted.model_probability > 0.0 && quoted.model_probability < 1.0);
        let unquoted = entries.iter().find(|entry| entry.threshold == 9).unwrap();
        assert_eq!(unquoted.side, None);

        let root = std::env::temp_dir().join(format!("pns-paper-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        super::record_slate(&cards, &markets, &model, &root).unwrap();
        let first = super::read_log(&root).unwrap();
        super::record_slate(&cards, &markets, &model, &root).unwrap();
        let second = super::read_log(&root).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn record_slate_freezes_placed_trades_but_refreshes_open_markets() {
        let cards = crate::score_date_fixture("2026-07-21").unwrap();
        let model = crate::ModelInfo {
            artifact_id: "test-artifact".into(),
            schema_version: 2,
            trained_through: chrono::NaiveDate::from_ymd_opt(2026, 7, 16).unwrap(),
            training_rows: 100,
        };
        let root = std::env::temp_dir().join(format!("pns-paper-refresh-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();

        let placed_ticker = "KXMLBKS-26JUL211910AAABBB-AAAFACE1-6";
        let open_ticker = "KXMLBKS-26JUL211910AAABBB-AAAFACE1-9";
        let seed = vec![
            super::PaperEntry {
                market_ticker: placed_ticker.into(),
                event_date: "2026-07-21".into(),
                pitcher_id: 660271,
                pitcher_name: "Fixture Ace".into(),
                threshold: 6,
                model_probability: 0.5,
                yes_bid: Some(40),
                yes_ask: Some(45),
                side: Some(super::Side::Yes),
                entry_price_cents: Some(45),
                edge: Some(0.1),
                recorded_at: "frozen".into(),
                actual_strikeouts: None,
                result_yes: None,
                profit_cents: None,
                voided: false,
                model_expected_strikeouts: Some(6.0),
                model_sd_strikeouts: Some(2.0),
                model_artifact_id: "test-artifact".into(),
            },
            super::PaperEntry {
                market_ticker: open_ticker.into(),
                event_date: "2026-07-21".into(),
                pitcher_id: 660271,
                pitcher_name: "Fixture Ace".into(),
                threshold: 9,
                model_probability: 0.1,
                yes_bid: None,
                yes_ask: None,
                side: None,
                entry_price_cents: None,
                edge: None,
                recorded_at: "frozen".into(),
                actual_strikeouts: None,
                result_yes: None,
                profit_cents: None,
                voided: false,
                model_expected_strikeouts: Some(6.0),
                model_sd_strikeouts: Some(2.0),
                model_artifact_id: "test-artifact".into(),
            },
        ];
        super::write_log(&super::log_path(&root), &seed).unwrap();

        let fresh_markets = vec![
            market(placed_ticker, "Fixture Ace: 6+", Some(10), Some(15)),
            market(open_ticker, "Fixture Ace: 9+", Some(10), Some(15)),
        ];
        super::record_slate(&cards, &fresh_markets, &model, &root).unwrap();
        let entries = super::read_log(&root).unwrap();

        let placed = entries
            .iter()
            .find(|entry| entry.market_ticker == placed_ticker)
            .unwrap();
        assert_eq!(placed.recorded_at, "frozen");
        assert_eq!(placed.yes_bid, Some(40));
        assert_eq!(placed.entry_price_cents, Some(45));

        let refreshed = entries
            .iter()
            .find(|entry| entry.market_ticker == open_ticker)
            .unwrap();
        assert_ne!(refreshed.recorded_at, "frozen");
        assert_eq!(refreshed.yes_bid, Some(10));
        assert_eq!(refreshed.yes_ask, Some(15));

        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn summary_totals_are_consistent() {
        let cards = crate::score_date_fixture("2026-07-21").unwrap();
        let model = crate::ModelInfo {
            artifact_id: "test-artifact".into(),
            schema_version: 2,
            trained_through: chrono::NaiveDate::from_ymd_opt(2026, 7, 16).unwrap(),
            training_rows: 100,
        };
        let markets = vec![
            market(
                "KXMLBKS-26JUL211910AAABBB-AAAFACE1-4",
                "Fixture Ace: 4+",
                Some(30),
                Some(35),
            ),
            market(
                "KXMLBKS-26JUL211910AAABBB-AAAFACE1-8",
                "Fixture Ace: 8+",
                Some(60),
                Some(65),
            ),
        ];
        let mut entries = super::match_markets(&cards, &markets, &model, "now", 0.05);
        for entry in &mut entries {
            super::apply_result(entry, 5.0);
        }
        let summary = super::summarize(&entries);
        assert_eq!(summary.logged_markets, 2);
        assert_eq!(summary.graded_logged, 2);
        assert_eq!(summary.open_positions, 0);
        assert_eq!(
            summary.graded_bets,
            entries.iter().filter(|entry| entry.side.is_some()).count()
        );
        assert!(summary.observed_yes_rate.is_some());
    }

    #[test]
    fn summarize_excludes_voided_entries_from_every_stat() {
        let cards = crate::score_date_fixture("2026-07-21").unwrap();
        let model = crate::ModelInfo {
            artifact_id: "test-artifact".into(),
            schema_version: 2,
            trained_through: chrono::NaiveDate::from_ymd_opt(2026, 7, 16).unwrap(),
            training_rows: 100,
        };
        let markets = vec![market(
            "KXMLBKS-26JUL211910AAABBB-AAAFACE1-4",
            "Fixture Ace: 4+",
            Some(30),
            Some(35),
        )];
        let mut entries = super::match_markets(&cards, &markets, &model, "now", 0.05);
        super::apply_result(&mut entries[0], 5.0);
        let settled = super::summarize(&entries);
        assert_eq!(settled.logged_markets, 1);
        assert_eq!(settled.graded_logged, 1);

        entries[0].voided = true;
        let voided = super::summarize(&entries);
        assert_eq!(voided.logged_markets, 0);
        assert_eq!(voided.graded_logged, 0);
        assert_eq!(voided.graded_bets, 0);
        assert_eq!(voided.open_positions, 0);
        assert!(voided.mean_model_probability.is_none());
        assert!(voided.observed_yes_rate.is_none());
    }
}
