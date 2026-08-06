//! Per-batter strikeout rates, split by the hand of the starting pitcher faced.
//!
//! A batter-by-batter simulator needs one number per lineup slot: the chance this
//! hitter strikes out against this arm. Raw same-hand samples are small — a season
//! gives a regular a few hundred plate appearances against righties and far fewer
//! against lefties — so the rate is shrunk toward the league rate against that same
//! hand, exactly the way [`super::enrich`] shrinks opponent and park rates toward
//! the league ER/IP.
//!
//! The fold is leakage-safe by construction: [`BatterKRates::as_of`] only absorbs
//! lines from games played strictly before the date being served, so a batter's
//! own game can never inform his rate for that game.
//!
//! The accumulator itself never fetches; it consumes
//! [`super::lineups::normalize_boxscore`] output. The one live-path exception is
//! [`live_batter_rates`], which seeds from the shipped lines file and tops up
//! from recent box scores.

use super::lineups::GameBoxscore;
use crate::Result;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Strength of the league-rate prior, in plate appearances. Strikeout rate is one
/// of the fastest-stabilizing batting statistics (roughly 60 PA to half signal), so
/// the prior stays light: a full season of same-hand plate appearances (~200) still
/// leaves about two thirds of the weight on the batter's own line.
const PRIOR_PLATE_APPEARANCES: f64 = 100.0;

/// League K/PA used before any line has been absorbed, when there is no league to
/// average yet. Mirrors the 0.48 ER/IP seed in [`super::enrich`]; ~22% is the
/// modern league-wide strikeout rate.
const DEFAULT_LEAGUE_K_RATE: f64 = 0.22;

/// One batter's line from one game, attributed to the hand of the starting pitcher
/// he faced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatterGameLine {
    pub game_date: NaiveDate,
    pub batter_id: u32,
    pub strikeouts: u16,
    pub plate_appearances: u16,
    /// `'L'` or `'R'` — the opposing *starter*, not every arm the batter saw. A
    /// starter takes most of a lineup's plate appearances, so his hand is the
    /// available approximation until pitch-level data is worth its cost.
    pub opposing_hand: char,
}

impl BatterGameLine {
    /// Every starting batter's line from one completed game. Each side is attributed
    /// to the *opposing* side's starting pitcher; a side whose opponent's hand is
    /// unknown is skipped rather than guessed, because a wrong hand pollutes both
    /// splits at once.
    pub fn from_boxscore(boxscore: &GameBoxscore, game_date: NaiveDate) -> Vec<Self> {
        let mut lines = Vec::new();
        for (team, opponent) in [
            (&boxscore.away, &boxscore.home),
            (&boxscore.home, &boxscore.away),
        ] {
            let Some(opposing_hand) = opponent.starting_pitcher_hand else {
                continue;
            };
            lines.extend(team.batting_order.iter().map(|line| Self {
                game_date,
                batter_id: line.batter_id,
                strikeouts: line.strikeouts,
                plate_appearances: line.plate_appearances,
                opposing_hand,
            }));
        }
        lines
    }
}

/// One row of `data/processed/batter_lines.csv`: a [`BatterGameLine`] plus the
/// game it came from. The line itself does not carry `game_pk` because a rate fold
/// never needs it, but the stored file does: both starters of a game share one box
/// score, so a rebuild dedupes on (`game_pk`, `batter_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatterLineRow {
    pub game_date: NaiveDate,
    pub game_pk: u32,
    pub batter_id: u32,
    pub strikeouts: u16,
    pub plate_appearances: u16,
    pub opposing_hand: char,
}

impl BatterLineRow {
    pub fn new(game_pk: u32, line: BatterGameLine) -> Self {
        Self {
            game_date: line.game_date,
            game_pk,
            batter_id: line.batter_id,
            strikeouts: line.strikeouts,
            plate_appearances: line.plate_appearances,
            opposing_hand: line.opposing_hand,
        }
    }

    /// The fold-ready line, dropping the provenance column.
    pub fn line(self) -> BatterGameLine {
        BatterGameLine {
            game_date: self.game_date,
            batter_id: self.batter_id,
            strikeouts: self.strikeouts,
            plate_appearances: self.plate_appearances,
            opposing_hand: self.opposing_hand,
        }
    }
}

pub fn write_batter_lines(rows: &[BatterLineRow], path: impl AsRef<Path>) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_batter_lines(path: impl AsRef<Path>) -> Result<Vec<BatterLineRow>> {
    let mut reader = csv::Reader::from_path(path)?;
    reader
        .deserialize()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Batter K-rate tables for a live slate on `date`: the shipped
/// `batter_lines.csv` seeds the fold, then box scores of final games played
/// after the file ends (and strictly before `date`) top it up — the same
/// seed-plus-top-up shape as [`super::asof::AsOfContext::build`]. This is the
/// one function in this module that fetches.
///
/// No embedded fallback: the lines file is ~6 MB, so a missing file is an error
/// the caller downgrades to a data warning rather than a reason to bloat every
/// binary.
pub fn live_batter_rates(client: &super::MlbStatsClient, date: NaiveDate) -> Result<BatterKRates> {
    let rows = read_batter_lines(batter_lines_path()?)?;
    let seeded_through = rows
        .iter()
        .map(|row| row.game_date)
        .max()
        .ok_or_else(|| crate::PnsError::Data("batter_lines.csv is empty".into()))?;
    let mut lines: Vec<BatterGameLine> = rows.into_iter().map(BatterLineRow::line).collect();
    let fetch_from = seeded_through + chrono::Duration::days(1);
    let fetch_to = date - chrono::Duration::days(1);
    if fetch_from <= fetch_to {
        for entry in client.game_index(fetch_from, fetch_to)? {
            if !entry.is_final {
                continue;
            }
            // A game with no batting order (suspended, data gap) is skipped the
            // same way the historical backfill skips it.
            let Ok(boxscore) = client.game_boxscore(entry.game_pk) else {
                continue;
            };
            lines.extend(BatterGameLine::from_boxscore(&boxscore, entry.game_date));
        }
    }
    Ok(BatterKRates::as_of(&lines, date))
}

/// `PNS_BATTER_LINES` overrides; otherwise the repo-relative path, matching how
/// [`super::asof`] resolves the training dataset.
fn batter_lines_path() -> Result<std::path::PathBuf> {
    if let Ok(path) = std::env::var("PNS_BATTER_LINES") {
        return Ok(std::path::PathBuf::from(path));
    }
    let current = std::env::current_dir()
        .map_err(|error| crate::PnsError::Data(format!("no working directory: {error}")))?;
    [
        current.join("data/processed/batter_lines.csv"),
        current.join("../data/processed/batter_lines.csv"),
    ]
    .into_iter()
    .find(|path| path.exists())
    .ok_or_else(|| {
        crate::PnsError::Data(
            "batter_lines.csv not found; run the build_lineup_dataset example \
             or set PNS_BATTER_LINES"
                .into(),
        )
    })
}

/// Running K/PA totals, matching [`super::enrich::RateStats`]'s shape.
#[derive(Default, Clone, Copy)]
struct KStats {
    strikeouts: f64,
    plate_appearances: f64,
}

impl KStats {
    fn add(&mut self, line: &BatterGameLine) {
        self.strikeouts += line.strikeouts as f64;
        self.plate_appearances += line.plate_appearances as f64;
    }

    fn rate(self, fallback: f64, prior_plate_appearances: f64) -> f64 {
        (self.strikeouts + fallback * prior_plate_appearances)
            / (self.plate_appearances + prior_plate_appearances)
    }
}

/// K/PA totals by (batter, opposing hand) and league-wide by hand, as of some date.
#[derive(Default, Clone)]
pub struct BatterKRates {
    league: HashMap<char, KStats>,
    batters: HashMap<(u32, char), KStats>,
}

impl BatterKRates {
    /// The state a lookup on `date` may use: every line played strictly earlier.
    /// Lines from `date` itself are excluded, so a game can never inform its own
    /// projection — the same guarantee the date-block fold in
    /// [`super::enrich::accumulate_historical_context`] provides.
    pub fn as_of(lines: &[BatterGameLine], date: NaiveDate) -> Self {
        let mut state = Self::default();
        for line in lines.iter().filter(|line| line.game_date < date) {
            state.absorb(line);
        }
        state
    }

    /// Folds one line in. A caller walking dates in order absorbs a date's lines
    /// only after serving every lookup for that date.
    pub fn absorb(&mut self, line: &BatterGameLine) {
        self.league.entry(line.opposing_hand).or_default().add(line);
        self.batters
            .entry((line.batter_id, line.opposing_hand))
            .or_default()
            .add(line);
    }

    /// League K/PA against `hand`.
    pub fn league_k_rate(&self, hand: char) -> f64 {
        let league = self.league.get(&hand).copied().unwrap_or_default();
        if league.plate_appearances > 0.0 {
            league.strikeouts / league.plate_appearances
        } else {
            DEFAULT_LEAGUE_K_RATE
        }
    }

    /// A batter's K/PA against `hand`, shrunk toward the league rate against that
    /// same hand. An unknown batter has no plate appearances to weigh against the
    /// prior, so this returns the league rate exactly — the same "no data resolves
    /// to neutral" answer [`super::asof::AsOfContext`] gives an unknown team id.
    pub fn k_rate(&self, batter_id: u32, hand: char) -> f64 {
        let league_rate = self.league_k_rate(hand);
        self.batters
            .get(&(batter_id, hand))
            .copied()
            .unwrap_or_default()
            .rate(league_rate, PRIOR_PLATE_APPEARANCES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Day 1 of a fixture season, counting past a month boundary.
    fn date(day: i64) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 5, 1).expect("valid fixture date")
            + chrono::Duration::days(day - 1)
    }

    fn line(
        day: i64,
        batter_id: u32,
        strikeouts: u16,
        plate_appearances: u16,
        hand: char,
    ) -> BatterGameLine {
        BatterGameLine {
            game_date: date(day),
            batter_id,
            strikeouts,
            plate_appearances,
            opposing_hand: hand,
        }
    }

    /// Ten days against right-handers: 50 strikeouts in 160 plate appearances, so
    /// the league rate is 0.3125. Batter 1 is far worse than that, batter 4 never
    /// strikes out.
    fn fixture_lines() -> Vec<BatterGameLine> {
        let mut lines = Vec::new();
        for day in 1..=10 {
            lines.push(line(day, 1, 3, 4, 'R'));
            lines.push(line(day, 2, 1, 4, 'R'));
            lines.push(line(day, 3, 1, 4, 'R'));
            lines.push(line(day, 4, 0, 4, 'R'));
        }
        lines
    }

    #[test]
    fn attributes_each_side_to_the_opposing_starter_hand() {
        let boxscore = crate::data::lineups::fixture_boxscore().unwrap();
        let lines = BatterGameLine::from_boxscore(&boxscore, date(1));
        // Away starter throws left, home starter throws right.
        let away_batter = lines.iter().find(|line| line.batter_id == 607208).unwrap();
        let home_batter = lines.iter().find(|line| line.batter_id == 672640).unwrap();
        assert_eq!(away_batter.opposing_hand, 'R');
        assert_eq!(home_batter.opposing_hand, 'L');
        assert_eq!(away_batter.game_date, date(1));
        assert_eq!(lines.len(), 6);
        // The substitute never enters the accumulator.
        assert!(!lines.iter().any(|line| line.batter_id == 665052));
    }

    #[test]
    fn a_side_without_a_known_opposing_hand_contributes_nothing() {
        let mut boxscore = crate::data::lineups::fixture_boxscore().unwrap();
        boxscore.home.starting_pitcher_hand = None;
        let lines = BatterGameLine::from_boxscore(&boxscore, date(1));
        // Only the home batters remain: they faced the away starter, whose hand is
        // still known.
        assert_eq!(lines.len(), 3);
        assert!(lines.iter().all(|line| line.opposing_hand == 'L'));
    }

    #[test]
    fn batter_lines_round_trip_through_the_csv() {
        let boxscore = crate::data::lineups::fixture_boxscore().unwrap();
        let rows: Vec<_> = BatterGameLine::from_boxscore(&boxscore, date(1))
            .into_iter()
            .map(|line| BatterLineRow::new(823837, line))
            .collect();
        let path = std::env::temp_dir().join("pns-batter-lines-roundtrip.csv");
        write_batter_lines(&rows, &path).unwrap();
        let loaded = read_batter_lines(&path).unwrap();
        assert_eq!(loaded, rows);
        // And the stored rows still fold exactly like the lines they came from.
        let folded: Vec<_> = loaded.into_iter().map(BatterLineRow::line).collect();
        assert_eq!(folded, BatterGameLine::from_boxscore(&boxscore, date(1)));
    }

    #[test]
    fn lookups_never_see_games_on_or_after_their_own_date() {
        let history = fixture_lines();
        let mut with_today = history.clone();
        with_today.push(line(11, 1, 4, 4, 'R'));
        let lookup_date = date(11);
        assert_eq!(
            BatterKRates::as_of(&with_today, lookup_date).k_rate(1, 'R'),
            BatterKRates::as_of(&history, lookup_date).k_rate(1, 'R'),
            "a game on the lookup date leaked into the rate"
        );
        // The same line does move the rate once its date is in the past.
        assert!(
            BatterKRates::as_of(&with_today, date(12)).k_rate(1, 'R')
                > BatterKRates::as_of(&history, date(12)).k_rate(1, 'R')
        );
    }

    #[test]
    fn an_unknown_batter_gets_exactly_the_league_rate() {
        let rates = BatterKRates::as_of(&fixture_lines(), date(11));
        let league = rates.league_k_rate('R');
        assert!((league - 0.3125).abs() < 1e-9, "league rate was {league}");
        assert_eq!(rates.k_rate(9999, 'R'), league);
        // No lines at all: the seed prior, not a divide by zero.
        assert_eq!(
            BatterKRates::default().k_rate(9999, 'L'),
            DEFAULT_LEAGUE_K_RATE
        );
    }

    #[test]
    fn a_high_strikeout_batter_lands_between_his_own_rate_and_the_league() {
        let rates = BatterKRates::as_of(&fixture_lines(), date(11));
        let league = rates.league_k_rate('R');
        let shrunk = rates.k_rate(1, 'R');
        let raw = 30.0 / 40.0;
        assert!(
            league < shrunk && shrunk < raw,
            "expected {league} < {shrunk} < {raw}"
        );
        // 40 plate appearances against a 100 PA prior keep most of the weight on
        // the league.
        assert!((shrunk - (30.0 + league * 100.0) / 140.0).abs() < 1e-9);
    }

    #[test]
    fn each_hand_accumulates_separately() {
        // Mirror-image platoon splits: batter 1 whiffs against lefties, batter 2
        // against righties, and the two hands share the same league rate so only
        // the per-batter split can move a lookup.
        let mut lines = Vec::new();
        for day in 1..=20 {
            lines.push(line(day, 1, 3, 4, 'L'));
            lines.push(line(day, 2, 1, 4, 'L'));
            lines.push(line(day + 20, 1, 1, 4, 'R'));
            lines.push(line(day + 20, 2, 3, 4, 'R'));
        }
        let rates = BatterKRates::as_of(&lines, date(41));
        assert!((rates.league_k_rate('L') - 0.5).abs() < 1e-9);
        assert!((rates.league_k_rate('R') - 0.5).abs() < 1e-9);
        assert!(
            rates.k_rate(1, 'L') > rates.league_k_rate('L')
                && rates.k_rate(1, 'R') < rates.league_k_rate('R'),
            "batter 1's platoon split collapsed"
        );
        assert!(
            rates.k_rate(2, 'R') > rates.league_k_rate('R')
                && rates.k_rate(2, 'L') < rates.league_k_rate('L'),
            "batter 2's platoon split collapsed"
        );
    }
}
