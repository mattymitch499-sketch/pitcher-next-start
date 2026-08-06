//! As-of-date context engine.
//!
//! The model's coefficients were fit on the *training* opponent-offense and park
//! statistics — starter ER/IP by (opponent, pitcher hand) and by park, shrunk
//! toward the league rate. Live scoring must serve those same statistics, so
//! this module replays the exact accumulator fold from
//! [`super::enrich::accumulate_historical_context`] over every start played
//! before the slate date and keeps the resulting lookup tables.
//!
//! The accumulators are cumulative from the first training season, so the
//! current season alone is not equivalent: seeding starts from the shipped
//! training dataset, then tops up with starts played after the model artifact's
//! `data_through`.

use super::types::PitcherStartLog;
use super::{MlbStatsClient, dataset, enrich};
use crate::Result;
use chrono::{Datelike, Duration, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Fallback copy of the dataset the shipped model was fit on, for a binary run
/// outside the repo. Mirrors how the model artifact is embedded in `lib.rs`.
const EMBEDDED_TRAINING_DATASET: &str =
    include_str!("../../../../data/processed/training_starts.csv");

/// Matches the `season_starters` TTL: the derived tables move once a day.
const FACTOR_CACHE_TTL_SECONDS: u64 = 86_400;

/// Same starter threshold the training dataset was built with, so the top-up
/// rows have the same composition as the seeded ones.
const MINIMUM_SEASON_STARTS: u16 = 5;

/// Opponent-offense and park factors as of a slate date, computed by the
/// training accumulator rather than by live team splits.
#[derive(Debug, Clone)]
pub struct AsOfContext {
    opponent: HashMap<(u32, char), f64>,
    park: HashMap<u32, f64>,
}

/// JSON-friendly mirror of [`AsOfContext`]: JSON object keys cannot be tuples,
/// and factors are stored as raw f64 bits because a JSON decimal round trip can
/// move the value by one unit in the last place — enough to break the exact
/// train/serve equality this engine exists to guarantee.
#[derive(Serialize, Deserialize)]
struct CachedTables {
    opponent: Vec<(u32, char, u64)>,
    park: Vec<(u32, u64)>,
}

impl From<&AsOfContext> for CachedTables {
    fn from(context: &AsOfContext) -> Self {
        Self {
            opponent: context
                .opponent
                .iter()
                .map(|(&(opponent_id, hand), &factor)| (opponent_id, hand, factor.to_bits()))
                .collect(),
            park: context
                .park
                .iter()
                .map(|(&home_team_id, &factor)| (home_team_id, factor.to_bits()))
                .collect(),
        }
    }
}

impl From<CachedTables> for AsOfContext {
    fn from(cached: CachedTables) -> Self {
        Self {
            opponent: cached
                .opponent
                .into_iter()
                .map(|(opponent_id, hand, bits)| ((opponent_id, hand), f64::from_bits(bits)))
                .collect(),
            park: cached
                .park
                .into_iter()
                .map(|(home_team_id, bits)| (home_team_id, f64::from_bits(bits)))
                .collect(),
        }
    }
}

impl AsOfContext {
    /// Opponent offense factor for `opponent_id` facing a `hand`-handed starter.
    /// An unknown id resolves to 1.0, the value the training fold assigns a team
    /// it has no innings for.
    pub fn opponent_offense_factor(&self, opponent_id: u32, hand: char) -> f64 {
        self.opponent
            .get(&(opponent_id, hand))
            .copied()
            .unwrap_or(1.0)
    }

    /// Park factor for a home team id, 1.0 for an unseen park.
    pub fn park_factor(&self, home_team_id: u32) -> f64 {
        self.park.get(&home_team_id).copied().unwrap_or(1.0)
    }

    /// Folds `rows` with the training accumulator and keeps the final state's
    /// factor tables. Every row is absorbed, so the tables answer for a start on
    /// any later date — exactly what `enrich_historical_context` would write
    /// onto that start.
    pub fn from_starts(mut rows: Vec<PitcherStartLog>, fetched_hands: &HashMap<u32, char>) -> Self {
        let hands = hand_map(&rows, fetched_hands);
        let (opponent, park) =
            enrich::accumulate_historical_context(&mut rows, &hands).factor_tables();
        Self { opponent, park }
    }

    /// Builds the context for a slate on `target_date`: the training dataset
    /// seeded first, then every start played after `data_through` and before the
    /// slate. The derived tables are cached, so the per-pitcher top-up fetch runs
    /// at most once a day per slate.
    pub fn build(
        client: &MlbStatsClient,
        target_date: NaiveDate,
        data_through: NaiveDate,
    ) -> Result<Self> {
        let cache_file = super::cache_path(&format!("asof-factors-{data_through}-{target_date}"));
        if super::cache_is_fresh(&cache_file, FACTOR_CACHE_TTL_SECONDS)
            && let Ok(text) = std::fs::read_to_string(&cache_file)
            && let Ok(cached) = serde_json::from_str::<CachedTables>(&text)
        {
            return Ok(cached.into());
        }
        let mut rows = training_dataset()?;
        rows.retain(|row| row.game_date < target_date);
        let fresh = starts_between(
            client,
            data_through + Duration::days(1),
            target_date - Duration::days(1),
        )?;
        let hands = if fresh.is_empty() {
            HashMap::new()
        } else {
            fetch_pitcher_hands(client, &fresh)?
        };
        // A dataset CSV refreshed past the artifact's `data_through` overlaps the
        // fetched window, so drop duplicates the way the dataset builder does.
        rows.extend(fresh);
        let deduplicated: BTreeMap<_, _> = rows
            .into_iter()
            .map(|row| ((row.pitcher_id, row.game_pk), row))
            .collect();
        let context = Self::from_starts(deduplicated.into_values().collect(), &hands);
        if let Some(parent) = cache_file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &cache_file,
            serde_json::to_string(&CachedTables::from(&context))?,
        )?;
        Ok(context)
    }
}

/// The dataset the shipped model was fit on. A disk copy wins so a refreshed
/// `data/processed/training_starts.csv` takes effect without a rebuild.
fn training_dataset() -> Result<Vec<PitcherStartLog>> {
    if let Ok(path) = std::env::var("PNS_TRAINING_DATASET") {
        return dataset::read_csv(path);
    }
    if let Ok(current) = std::env::current_dir() {
        let candidate = [
            current.join("data/processed/training_starts.csv"),
            current.join("../data/processed/training_starts.csv"),
        ]
        .into_iter()
        .find(|path| path.exists());
        if let Some(path) = candidate {
            return dataset::read_csv(path);
        }
    }
    dataset::read_csv_text(EMBEDDED_TRAINING_DATASET)
}

/// Hands used for opponent attribution. A row that already carries a hand (every
/// training CSV row does) keeps it, so seeded rows are attributed exactly as they
/// were during training; freshly fetched rows fall back to the API answer.
fn hand_map(rows: &[PitcherStartLog], fetched: &HashMap<u32, char>) -> HashMap<u32, char> {
    let mut hands: HashMap<u32, char> = rows
        .iter()
        .filter_map(|row| row.pitch_hand.map(|hand| (row.pitcher_id, hand)))
        .collect();
    for (&pitcher_id, &hand) in fetched {
        hands.entry(pitcher_id).or_insert(hand);
    }
    hands
}

/// Starts played in `[start, end]`, fetched the way the dataset builder fetches a
/// season. A pitcher with no start in the window makes `pitcher_starts_range`
/// return an error, which is normal for a short window and simply skipped.
fn starts_between(
    client: &MlbStatsClient,
    start: NaiveDate,
    end: NaiveDate,
) -> Result<Vec<PitcherStartLog>> {
    if start > end {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for season in start.year()..=end.year() {
        let ids = client.season_starters(season, MINIMUM_SEASON_STARTS)?;
        let window_start =
            start.max(NaiveDate::from_ymd_opt(season, 1, 1).expect("valid season start"));
        let window_end =
            end.min(NaiveDate::from_ymd_opt(season, 12, 31).expect("valid season end"));
        for chunk in ids.chunks(12) {
            let results: Vec<_> = std::thread::scope(|scope| {
                let handles: Vec<_> = chunk
                    .iter()
                    .map(|&pitcher_id| {
                        let client = client.clone();
                        scope.spawn(move || {
                            client.pitcher_starts_range(pitcher_id, window_start, window_end)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("as-of context worker panicked"))
                    .collect()
            });
            rows.extend(results.into_iter().flatten().flatten());
        }
    }
    Ok(rows)
}

/// The people endpoint takes ids in the query string, so ask in chunks rather
/// than building one URL out of every starter in the window.
fn fetch_pitcher_hands(
    client: &MlbStatsClient,
    rows: &[PitcherStartLog],
) -> Result<HashMap<u32, char>> {
    let mut ids: Vec<_> = rows.iter().map(|row| row.pitcher_id).collect();
    ids.sort_unstable();
    ids.dedup();
    let mut hands = HashMap::new();
    for chunk in ids.chunks(50) {
        hands.extend(client.pitcher_hands(chunk)?);
    }
    Ok(hands)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(
        pitcher_id: u32,
        day: u32,
        team_id: u32,
        opponent_id: u32,
        home: bool,
        earned_runs: u16,
        innings_pitched: f64,
    ) -> PitcherStartLog {
        PitcherStartLog {
            pitcher_id,
            pitcher_name: format!("Pitcher {pitcher_id}"),
            game_pk: pitcher_id * 100 + day,
            game_date: NaiveDate::from_ymd_opt(2026, 5, day).expect("valid fixture date"),
            team_id,
            opponent_id,
            opponent_name: None,
            home,
            innings_pitched,
            strikeouts: 5,
            walks: 2,
            home_runs: 1,
            earned_runs,
            hits: 5,
            batters_faced: 24,
            pitch_count: Some(92),
            ground_ball_pct: Some(0.45),
            avg_fb_velo: Some(93.5),
            historical_park_factor: None,
            historical_opponent_offense: None,
            pitch_hand: None,
            opponent_batters: Vec::new(),
        }
    }

    /// Two history dates, then a third date whose factors the engine must
    /// reproduce. Opponent 20 hammers right-handers in park 10; opponent 30 does
    /// not, so the factors land away from neutral.
    fn fixture_rows() -> Vec<PitcherStartLog> {
        vec![
            start(1, 1, 10, 20, true, 8, 5.0),
            start(2, 1, 20, 10, false, 0, 7.0),
            start(3, 2, 30, 20, false, 6, 6.0),
            start(4, 2, 10, 30, true, 1, 6.0),
            start(1, 3, 10, 20, true, 3, 6.0),
            start(2, 3, 20, 10, false, 2, 6.0),
            start(5, 3, 40, 50, false, 4, 5.0),
        ]
    }

    fn fixture_hands() -> HashMap<u32, char> {
        [(1, 'R'), (2, 'L'), (3, 'R'), (4, 'R'), (5, 'R')]
            .into_iter()
            .collect()
    }

    #[test]
    fn as_of_lookups_equal_training_enrichment() {
        let hands = fixture_hands();
        let rows = fixture_rows();
        let scored_date = NaiveDate::from_ymd_opt(2026, 5, 3).unwrap();

        let mut enriched = rows.clone();
        enrich::enrich_historical_context(&mut enriched, &hands);
        let history: Vec<_> = rows
            .into_iter()
            .filter(|row| row.game_date < scored_date)
            .collect();
        let context = AsOfContext::from_starts(history, &hands);

        let mut checked = 0;
        let mut non_neutral = 0;
        for row in enriched.iter().filter(|row| row.game_date == scored_date) {
            let hand = hands[&row.pitcher_id];
            let home_team_id = if row.home {
                row.team_id
            } else {
                row.opponent_id
            };
            assert_eq!(
                context.opponent_offense_factor(row.opponent_id, hand),
                row.historical_opponent_offense.unwrap(),
                "opponent factor drifted for pitcher {}",
                row.pitcher_id
            );
            assert_eq!(
                context.park_factor(home_team_id),
                row.historical_park_factor.unwrap(),
                "park factor drifted for pitcher {}",
                row.pitcher_id
            );
            checked += 1;
            if (row.historical_opponent_offense.unwrap() - 1.0).abs() > 1e-9
                || (row.historical_park_factor.unwrap() - 1.0).abs() > 1e-9
            {
                non_neutral += 1;
            }
        }
        assert_eq!(checked, 3);
        assert!(non_neutral > 0, "fixture must exercise non-neutral factors");
    }

    #[test]
    fn row_carried_hands_seed_the_same_tables_as_fetched_hands() {
        let hands = fixture_hands();
        let history: Vec<_> = fixture_rows()
            .into_iter()
            .filter(|row| row.game_date.day() < 3)
            .collect();
        let mut carried = history.clone();
        for row in &mut carried {
            row.pitch_hand = Some(hands[&row.pitcher_id]);
        }
        let fetched = AsOfContext::from_starts(history, &hands);
        let seeded = AsOfContext::from_starts(carried, &HashMap::new());
        assert_eq!(
            seeded.opponent_offense_factor(20, 'R'),
            fetched.opponent_offense_factor(20, 'R')
        );
        assert_eq!(seeded.park_factor(10), fetched.park_factor(10));
    }

    #[test]
    fn cached_tables_round_trip() {
        let context = AsOfContext::from_starts(fixture_rows(), &fixture_hands());
        let text = serde_json::to_string(&CachedTables::from(&context)).unwrap();
        let restored: AsOfContext = serde_json::from_str::<CachedTables>(&text).unwrap().into();
        assert_eq!(
            restored.opponent_offense_factor(20, 'R'),
            context.opponent_offense_factor(20, 'R')
        );
        assert_eq!(restored.park_factor(10), context.park_factor(10));
        assert_eq!(restored.park_factor(999), 1.0);
    }
}
