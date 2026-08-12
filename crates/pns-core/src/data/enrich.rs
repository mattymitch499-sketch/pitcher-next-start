use super::types::PitcherStartLog;
use crate::{PnsError, Result};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Default, Clone, Copy)]
struct RateStats {
    earned_runs: f64,
    innings: f64,
    strikeouts: f64,
    batters_faced: f64,
}
impl RateStats {
    fn add(&mut self, row: &PitcherStartLog) {
        self.earned_runs += row.earned_runs as f64;
        self.innings += row.innings_pitched;
        self.strikeouts += row.strikeouts as f64;
        self.batters_faced += row.batters_faced as f64;
    }
    fn rate(self, fallback: f64, prior_innings: f64) -> f64 {
        (self.earned_runs + fallback * prior_innings) / (self.innings + prior_innings)
    }
    /// Strikeouts per batter faced, shrunk toward `fallback` the same way
    /// [`Self::rate`] shrinks ER/IP. The denominator is batters faced rather than
    /// innings because a strikeout rate per *inning* would confound how often the
    /// opponent strikes out with how much traffic it puts on.
    fn k_rate(self, fallback: f64, prior_batters: f64) -> f64 {
        (self.strikeouts + fallback * prior_batters) / (self.batters_faced + prior_batters)
    }
}

/// League K/PA before any games are absorbed. Only reachable on the dataset's
/// very first date, where every factor collapses to exactly 1.0 anyway.
const PRIOR_LEAGUE_K_RATE: f64 = 0.22;

/// Shrinkage prior for an opponent's strikeout rate against one pitcher hand, in
/// batters faced. Chosen to match the existing 60-innings prior on
/// [`ContextAccumulators::opponent_offense_factor`] in scale — 60 IP is roughly
/// 258 batters — so the two opponent factors are shrunk comparably rather than
/// one being trusted far earlier in a season than the other.
const OPPONENT_K_PRIOR_BATTERS: f64 = 250.0;

/// Running ER/IP totals behind the historical context fold: league wide, by
/// (opponent, pitcher hand), and by park. Training enrichment and live as-of
/// lookups both read their factors from here, so the two cannot drift apart.
#[derive(Default, Clone)]
pub(crate) struct ContextAccumulators {
    league: RateStats,
    opponents: HashMap<(u32, char), RateStats>,
    parks: HashMap<u32, RateStats>,
}

impl ContextAccumulators {
    fn league_rate(&self) -> f64 {
        if self.league.innings > 0.0 {
            self.league.earned_runs / self.league.innings
        } else {
            0.48
        }
    }

    pub(crate) fn opponent_offense_factor(&self, opponent_id: u32, hand: char) -> f64 {
        let league_rate = self.league_rate();
        let opponent_rate = self
            .opponents
            .get(&(opponent_id, hand))
            .copied()
            .unwrap_or_default()
            .rate(league_rate, 60.0);
        (opponent_rate / league_rate).clamp(0.75, 1.30)
    }

    fn league_k_rate(&self) -> f64 {
        if self.league.batters_faced > 0.0 {
            self.league.strikeouts / self.league.batters_faced
        } else {
            PRIOR_LEAGUE_K_RATE
        }
    }

    /// How much more or less often this opponent strikes out against pitchers of
    /// `hand`, relative to the league.
    ///
    /// This is the strikeout analogue of [`Self::opponent_offense_factor`], which
    /// measures the same opponent in *earned runs* — a proxy that the v3
    /// artifact's coefficients show carries essentially nothing for the k-rate
    /// regression (-0.00111 per SD, against +0.01274 for swinging-strike rate).
    /// Clamped to the same bounds as the ER factor rather than new ones.
    pub(crate) fn opponent_k_factor(&self, opponent_id: u32, hand: char) -> f64 {
        let league_rate = self.league_k_rate();
        let opponent_rate = self
            .opponents
            .get(&(opponent_id, hand))
            .copied()
            .unwrap_or_default()
            .k_rate(league_rate, OPPONENT_K_PRIOR_BATTERS);
        (opponent_rate / league_rate).clamp(0.75, 1.30)
    }

    pub(crate) fn park_factor(&self, home_team_id: u32) -> f64 {
        let league_rate = self.league_rate();
        let park_rate = self
            .parks
            .get(&home_team_id)
            .copied()
            .unwrap_or_default()
            .rate(league_rate, 120.0);
        (park_rate / league_rate).clamp(0.85, 1.15)
    }

    /// Every factor this state can answer, as (opponent, hand) and park lookup
    /// tables. An id missing from a table resolves to 1.0, which is exactly what
    /// the accumulator returns for a team or park it holds no innings for.
    pub(crate) fn factor_tables(&self) -> (HashMap<(u32, char), f64>, HashMap<u32, f64>) {
        (
            self.opponents
                .keys()
                .map(|&(opponent_id, hand)| {
                    (
                        (opponent_id, hand),
                        self.opponent_offense_factor(opponent_id, hand),
                    )
                })
                .collect(),
            self.parks
                .keys()
                .map(|&home_team_id| (home_team_id, self.park_factor(home_team_id)))
                .collect(),
        )
    }

    fn add(&mut self, row: &PitcherStartLog, hand: char) {
        let home_team_id = if row.home {
            row.team_id
        } else {
            row.opponent_id
        };
        self.league.add(row);
        self.opponents
            .entry((row.opponent_id, hand))
            .or_default()
            .add(row);
        self.parks.entry(home_team_id).or_default().add(row);
    }
}

/// Adds context to each row using only games on earlier dates. Opponent strength
/// is starter ER/IP by opponent and pitcher hand; park environment is starter
/// ER/IP by home-team park proxy. Both are shrunk toward the prior league rate.
pub fn enrich_historical_context(rows: &mut [PitcherStartLog], hands: &HashMap<u32, char>) {
    accumulate_historical_context(rows, hands);
}

/// The fold behind [`enrich_historical_context`], additionally returning the
/// accumulator state once every row has been absorbed. Live scoring reuses that
/// final state so it serves the same statistic the model was trained on.
pub(crate) fn accumulate_historical_context(
    rows: &mut [PitcherStartLog],
    hands: &HashMap<u32, char>,
) -> ContextAccumulators {
    rows.sort_by_key(|row| row.game_date);
    let mut state = ContextAccumulators::default();
    let mut index = 0;
    while index < rows.len() {
        let date = rows[index].game_date;
        let end = rows[index..]
            .iter()
            .position(|row| row.game_date > date)
            .map(|offset| index + offset)
            .unwrap_or(rows.len());
        for row in &mut rows[index..end] {
            let hand = hands.get(&row.pitcher_id).copied().unwrap_or('R');
            row.pitch_hand = Some(hand);
            let home_team_id = if row.home {
                row.team_id
            } else {
                row.opponent_id
            };
            row.historical_opponent_offense =
                Some(state.opponent_offense_factor(row.opponent_id, hand));
            row.historical_park_factor = Some(state.park_factor(home_team_id));
        }
        for row in &rows[index..end] {
            let hand = hands.get(&row.pitcher_id).copied().unwrap_or('R');
            state.add(row, hand);
        }
        index = end;
    }
    rows.sort_by_key(|row| (row.pitcher_id, row.game_date));
    state
}

/// Opponent strikeout factors for every start, keyed the way a `FeatureRow`
/// identifies itself.
///
/// A **study** surface. It deliberately returns a map instead of writing a
/// column onto [`PitcherStartLog`], so measuring whether this feature earns its
/// place costs no dataset rewrite and no schema bump. If it clears its bar the
/// factor moves into [`accumulate_historical_context`] beside the ER one.
///
/// Leakage is prevented the same way that fold prevents it: every factor on a
/// date is read *before* any of that date's rows are absorbed, so a start can
/// see neither its own game nor another game played the same day.
pub fn opponent_k_factors(
    rows: &[PitcherStartLog],
    hands: &HashMap<u32, char>,
) -> HashMap<(u32, NaiveDate), f64> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&index| rows[index].game_date);
    let mut state = ContextAccumulators::default();
    let mut factors = HashMap::new();
    let mut index = 0;
    while index < order.len() {
        let date = rows[order[index]].game_date;
        let end = order[index..]
            .iter()
            .position(|&row_index| rows[row_index].game_date > date)
            .map(|offset| index + offset)
            .unwrap_or(order.len());
        for &row_index in &order[index..end] {
            let row = &rows[row_index];
            let hand = hands.get(&row.pitcher_id).copied().unwrap_or('R');
            factors.insert(
                (row.pitcher_id, row.game_date),
                state.opponent_k_factor(row.opponent_id, hand),
            );
        }
        for &row_index in &order[index..end] {
            let row = &rows[row_index];
            let hand = hands.get(&row.pitcher_id).copied().unwrap_or('R');
            state.add(row, hand);
        }
        index = end;
    }
    factors
}

#[derive(Deserialize)]
struct SavantPitch {
    pitch_type: String,
    game_date: String,
    release_speed: String,
}

pub fn normalize_savant_velocity(csv_text: &str) -> Result<HashMap<NaiveDate, f64>> {
    let cleaned = csv_text.trim_start_matches('\u{feff}');
    let mut reader = csv::Reader::from_reader(cleaned.as_bytes());
    let mut grouped: HashMap<NaiveDate, (f64, usize)> = HashMap::new();
    for record in reader.deserialize::<SavantPitch>() {
        let pitch = record?;
        if !matches!(pitch.pitch_type.as_str(), "FF" | "SI") {
            continue;
        }
        let date = NaiveDate::parse_from_str(&pitch.game_date, "%Y-%m-%d")
            .map_err(|error| PnsError::Normalize(error.to_string()))?;
        let velocity = match pitch.release_speed.parse::<f64>() {
            Ok(value) => value,
            Err(_) => continue,
        };
        let entry = grouped.entry(date).or_default();
        entry.0 += velocity;
        entry.1 += 1;
    }
    Ok(grouped
        .into_iter()
        .filter(|(_, (_, count))| *count > 0)
        .map(|(date, (sum, count))| (date, sum / count as f64))
        .collect())
}

/// Pitch-outcome counts for one start, from Statcast's per-pitch `description`.
///
/// Swinging-strike rate is the best-known leading indicator of future strikeout
/// rate — it stabilises faster than K rate itself — and the model has never seen
/// it. These are stored as raw counts rather than rates so a caller can build a
/// leakage-safe rolling rate by summing whole starts, which a pre-divided rate
/// cannot do correctly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PitchOutcomes {
    pub pitches: u32,
    pub swinging_strikes: u32,
    pub called_strikes: u32,
}

/// Statcast `description` values counted as a swing and miss.
///
/// `foul_tip` is included: by rule the batter swung and the catcher caught it
/// cleanly, which is a whiff, and it is strike three when it happens with two
/// strikes. `missed_bunt` is included for the same reason. Ordinary `foul` is
/// not — contact was made. This is the contestable line in the definition, so it
/// is written down rather than left implicit.
const SWINGING_STRIKE_DESCRIPTIONS: &[&str] = &[
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "bunt_foul_tip",
    "missed_bunt",
];

impl PitchOutcomes {
    fn add(&mut self, description: &str) {
        self.pitches += 1;
        if SWINGING_STRIKE_DESCRIPTIONS.contains(&description) {
            self.swinging_strikes += 1;
        } else if description == "called_strike" {
            self.called_strikes += 1;
        }
    }

    /// Swinging strikes per pitch. `None` when no pitch was recorded.
    pub fn swinging_strike_rate(&self) -> Option<f64> {
        (self.pitches > 0).then(|| f64::from(self.swinging_strikes) / f64::from(self.pitches))
    }

    /// Called-plus-swinging strikes per pitch, the usual "CSW%".
    ///
    /// Note this **contains** [`Self::swinging_strike_rate`]. Using both as
    /// model features puts two columns in the contract competing for one piece
    /// of variance; pair the swinging rate with [`Self::called_strike_rate`]
    /// instead when both are wanted.
    pub fn called_plus_swinging_rate(&self) -> Option<f64> {
        (self.pitches > 0).then(|| {
            f64::from(self.swinging_strikes + self.called_strikes) / f64::from(self.pitches)
        })
    }

    /// Called strikes per pitch — the half of CSW% that is *not* swings and
    /// misses, so it can sit alongside the swinging rate without overlapping it.
    pub fn called_strike_rate(&self) -> Option<f64> {
        (self.pitches > 0).then(|| f64::from(self.called_strikes) / f64::from(self.pitches))
    }

    /// Folds another start's counts in, for rolling windows.
    pub fn merge(&mut self, other: &PitchOutcomes) {
        self.pitches += other.pitches;
        self.swinging_strikes += other.swinging_strikes;
        self.called_strikes += other.called_strikes;
    }
}

#[derive(Deserialize)]
struct SavantOutcomePitch {
    game_date: String,
    description: String,
}

/// Groups a Statcast `type=details` export into per-date pitch-outcome counts.
///
/// Unlike [`normalize_savant_velocity`] this must be fed an export with **no**
/// `hfPT` pitch-type filter, or the denominator counts only fastballs and every
/// rate comes out wrong.
pub fn normalize_savant_pitch_outcomes(
    csv_text: &str,
) -> Result<HashMap<NaiveDate, PitchOutcomes>> {
    let cleaned = csv_text.trim_start_matches('\u{feff}');
    let mut reader = csv::Reader::from_reader(cleaned.as_bytes());
    let mut grouped: HashMap<NaiveDate, PitchOutcomes> = HashMap::new();
    for record in reader.deserialize::<SavantOutcomePitch>() {
        let pitch = record?;
        let date = NaiveDate::parse_from_str(&pitch.game_date, "%Y-%m-%d")
            .map_err(|error| PnsError::Normalize(error.to_string()))?;
        grouped.entry(date).or_default().add(&pitch.description);
    }
    Ok(grouped)
}

pub fn join_velocity(
    rows: &mut [PitcherStartLog],
    pitcher_id: u32,
    velocities: &HashMap<NaiveDate, f64>,
) {
    for row in rows.iter_mut().filter(|row| row.pitcher_id == pitcher_id) {
        row.avg_fb_velo = velocities.get(&row.game_date).copied();
    }
}

/// Attaches per-start pitch-outcome counts, the same shape as
/// [`join_velocity`].
///
/// A date with no Savant row is left as `None` rather than zeroed: zero pitches
/// is indistinguishable from a zero swinging-strike rate downstream, and the
/// feature builder treats absent counts as missing while treating a real zero as
/// a measurement.
pub fn join_pitch_outcomes(
    rows: &mut [PitcherStartLog],
    pitcher_id: u32,
    outcomes: &HashMap<NaiveDate, PitchOutcomes>,
) {
    for row in rows.iter_mut().filter(|row| row.pitcher_id == pitcher_id) {
        if let Some(outcome) = outcomes.get(&row.game_date) {
            row.pitches = Some(outcome.pitches);
            row.swinging_strikes = Some(outcome.swinging_strikes);
            row.called_strikes = Some(outcome.called_strikes);
        }
    }
}

#[cfg(test)]
mod tests {
    /// The opponent strikeout factor must be blind to its own date, and must
    /// move with how often the opponent actually strikes out.
    ///
    /// The same-day half matters most: two games against the same opponent on
    /// the same day must both read neutral. If the fold absorbed rows as it went
    /// rather than a date at a time, the second one would already "know" the
    /// first, which is the subtle form of leakage that inflates a study's gain
    /// without failing anything visible.
    #[test]
    fn opponent_k_factor_is_as_of_and_tracks_strikeout_tendency() {
        use crate::data::types::PitcherStartLog;
        use std::collections::HashMap;

        fn start(
            pitcher_id: u32,
            day: u32,
            opponent_id: u32,
            strikeouts: u16,
            batters_faced: u16,
        ) -> PitcherStartLog {
            PitcherStartLog {
                pitcher_id,
                pitcher_name: format!("Pitcher {pitcher_id}"),
                game_pk: pitcher_id * 100 + day,
                game_date: chrono::NaiveDate::from_ymd_opt(2026, 5, day)
                    .expect("valid fixture date"),
                team_id: 900 + pitcher_id,
                opponent_id,
                opponent_name: None,
                home: true,
                innings_pitched: 6.0,
                strikeouts,
                walks: 2,
                home_runs: 1,
                earned_runs: 3,
                hits: 5,
                batters_faced,
                pitch_count: Some(95),
                ground_ball_pct: Some(0.45),
                avg_fb_velo: Some(93.5),
                historical_park_factor: None,
                historical_opponent_offense: None,
                pitch_hand: None,
                opponent_batters: Vec::new(),
                pitches: None,
                swinging_strikes: None,
                called_strikes: None,
            }
        }

        let rows = vec![
            // Day 1: opponent 20 strikes out constantly, opponent 30 barely.
            start(1, 1, 20, 15, 30),
            start(5, 1, 20, 15, 30),
            start(2, 1, 30, 3, 30),
            // Day 2: fresh pitchers face those same two opponents.
            start(3, 2, 20, 9, 30),
            start(4, 2, 30, 9, 30),
        ];
        let factors = super::opponent_k_factors(&rows, &HashMap::new());
        let date =
            |day| chrono::NaiveDate::from_ymd_opt(2026, 5, day).expect("valid fixture date");

        // Nothing is absorbed yet on the first date, so every factor is exactly
        // neutral — including the *second* game against opponent 20.
        assert_eq!(factors[&(1, date(1))], 1.0);
        assert_eq!(factors[&(5, date(1))], 1.0);
        assert_eq!(factors[&(2, date(1))], 1.0);

        // By day 2 the league sits at 33/90 and the opponents have separated.
        let high = factors[&(3, date(2))];
        let low = factors[&(4, date(2))];
        assert!(
            high > 1.0,
            "an opponent that strikes out often must read above neutral, got {high}"
        );
        assert!(
            low < 1.0,
            "an opponent that rarely strikes out must read below neutral, got {low}"
        );
    }

    #[test]
    fn aggregates_fastballs_by_start_date() {
        let csv = "pitch_type,game_date,release_speed\nFF,2025-04-01,95.0\nSI,2025-04-01,93.0\nKC,2025-04-01,80.0\nFF,2025-04-07,96.0\n";
        let rows = super::normalize_savant_velocity(csv).unwrap();
        assert_eq!(
            rows[&chrono::NaiveDate::from_ymd_opt(2025, 4, 1).unwrap()],
            94.0
        );
    }

    /// Every Statcast `description` value must land in exactly one bucket, and
    /// the denominator must count *every* pitch. Getting this wrong inflates or
    /// deflates the rate silently, so each value is pinned by name.
    #[test]
    fn pitch_outcomes_bucket_every_description() {
        let csv = "game_date,description\n\
                   2025-04-01,swinging_strike\n\
                   2025-04-01,swinging_strike_blocked\n\
                   2025-04-01,foul_tip\n\
                   2025-04-01,missed_bunt\n\
                   2025-04-01,called_strike\n\
                   2025-04-01,called_strike\n\
                   2025-04-01,foul\n\
                   2025-04-01,ball\n\
                   2025-04-01,hit_into_play\n\
                   2025-04-01,hit_by_pitch\n\
                   2025-04-07,swinging_strike\n";
        let rows = super::normalize_savant_pitch_outcomes(csv).unwrap();

        let first = rows[&chrono::NaiveDate::from_ymd_opt(2025, 4, 1).unwrap()];
        assert_eq!(first.pitches, 10, "every row is a pitch");
        assert_eq!(
            first.swinging_strikes, 4,
            "swinging, blocked, foul tip and missed bunt are whiffs"
        );
        assert_eq!(first.called_strikes, 2);
        // foul / ball / hit_into_play / hit_by_pitch are pitches and nothing else.
        assert!((first.swinging_strike_rate().unwrap() - 0.4).abs() < 1e-12);
        assert!((first.called_plus_swinging_rate().unwrap() - 0.6).abs() < 1e-12);

        // Dates stay separate — a rolling window sums whole starts.
        let second = rows[&chrono::NaiveDate::from_ymd_opt(2025, 4, 7).unwrap()];
        assert_eq!(second.pitches, 1);
        assert_eq!(second.swinging_strikes, 1);
    }

    /// Rates come from summed counts, not averaged rates: a 100-pitch start and
    /// a 10-pitch start must not carry equal weight.
    #[test]
    fn merging_outcomes_weights_by_pitch_count() {
        let mut heavy = super::PitchOutcomes {
            pitches: 100,
            swinging_strikes: 10,
            called_strikes: 0,
        };
        let light = super::PitchOutcomes {
            pitches: 10,
            swinging_strikes: 5,
            called_strikes: 0,
        };
        heavy.merge(&light);
        assert_eq!(heavy.pitches, 110);
        assert_eq!(heavy.swinging_strikes, 15);
        // 15/110, not the 0.30 that averaging 0.10 and 0.50 would give.
        let rate = heavy.swinging_strike_rate().unwrap();
        assert!((rate - 15.0 / 110.0).abs() < 1e-12, "rate was {rate}");
    }

    #[test]
    fn empty_outcomes_have_no_rate_rather_than_a_zero() {
        let empty = super::PitchOutcomes::default();
        assert_eq!(empty.swinging_strike_rate(), None);
        assert_eq!(empty.called_plus_swinging_rate(), None);
    }

    #[test]
    fn historical_context_never_uses_same_date() {
        let mut rows = crate::data::pitcher_logs::fixture_starts().unwrap();
        super::enrich_historical_context(&mut rows, &std::collections::HashMap::new());
        assert!(rows.iter().all(|row| row.historical_park_factor.is_some()));
        let first_date = rows.iter().map(|row| row.game_date).min().unwrap();
        assert!(
            rows.iter()
                .filter(|row| row.game_date == first_date)
                .all(|row| (row.historical_park_factor.unwrap() - 1.0).abs() < 1e-9)
        );
    }
}
