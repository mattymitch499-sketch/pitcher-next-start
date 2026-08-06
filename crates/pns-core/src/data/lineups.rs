//! Lineup surfaces for a batter-by-batter simulator.
//!
//! Two shapes, one module because both answer "who batted, in what order, against
//! whom":
//!
//! * **Today** — the schedule feed hydrated with `lineups` carries each side's
//!   posted card. MLB posts cards a few hours before first pitch, so an unposted
//!   card is a normal answer and is reported as `None` rather than as an empty
//!   lineup.
//! * **History** — a completed game's box score carries every batter's line and
//!   each side's starting pitcher. Box scores are the only feed that ties a
//!   batter's strikeouts to the arm he faced, which is what
//!   [`super::batters`] accumulates.

use crate::{PnsError, Result};
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;

const BOXSCORE_FIXTURE: &str = include_str!("../../../../data/fixtures/mlb_boxscore.json");
const SCHEDULE_FIXTURE: &str = include_str!("../../../../data/fixtures/mlb_schedule_lineups.json");

/// One scheduled game's lineup card, posted or not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameLineup {
    pub game_pk: u32,
    pub game_date: NaiveDate,
    pub away: TeamLineup,
    pub home: TeamLineup,
}

/// One side of a scheduled game. `batters` is `None` while MLB has not posted the
/// card yet — a different answer from "this team has no batters", which callers
/// must be able to tell apart before they simulate anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeamLineup {
    pub team_id: u32,
    /// Starting batter ids in lineup order, once posted.
    pub batters: Option<Vec<u32>>,
    pub probable_pitcher_id: Option<u32>,
}

/// Both sides of a completed game's box score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameBoxscore {
    pub away: BoxscoreTeam,
    pub home: BoxscoreTeam,
}

/// One side of a completed game: who started, how they hit, and which arm the
/// *other* side had to bat against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoxscoreTeam {
    pub team_id: u32,
    /// The nine batters who started, in lineup-slot order, with this game's line.
    pub batting_order: Vec<BoxscoreBatterLine>,
    pub starting_pitcher_id: Option<u32>,
    /// `'L'` or `'R'`, matching the hand codes the rest of the crate carries.
    /// `None` when the box score was not hydrated with `person`.
    pub starting_pitcher_hand: Option<char>,
}

/// One batter's line in one game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxscoreBatterLine {
    pub batter_id: u32,
    pub strikeouts: u16,
    pub plate_appearances: u16,
}

#[derive(Deserialize)]
struct ScheduleResponse {
    #[serde(default)]
    dates: Vec<ScheduleDate>,
}
#[derive(Deserialize)]
struct ScheduleDate {
    #[serde(default)]
    games: Vec<ScheduleGame>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScheduleGame {
    game_pk: u32,
    teams: ScheduleTeams,
    /// Absent on a future date and `{}` on the morning of a slate; both mean the
    /// card is not posted yet.
    #[serde(default)]
    lineups: Option<Lineups>,
}
#[derive(Deserialize)]
struct ScheduleTeams {
    away: ScheduleSide,
    home: ScheduleSide,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScheduleSide {
    team: Team,
    #[serde(default)]
    probable_pitcher: Option<Person>,
}
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct Lineups {
    #[serde(default)]
    away_players: Vec<Person>,
    #[serde(default)]
    home_players: Vec<Person>,
}
#[derive(Deserialize)]
struct Team {
    id: u32,
}
#[derive(Deserialize)]
struct Person {
    id: u32,
}

#[derive(Deserialize)]
struct BoxscoreResponse {
    teams: BoxscoreSides,
}
#[derive(Deserialize)]
struct BoxscoreSides {
    away: BoxscoreSide,
    home: BoxscoreSide,
}
#[derive(Deserialize)]
struct BoxscoreSide {
    team: Team,
    /// Pitcher ids in appearance order, so the starter is the first entry.
    #[serde(default)]
    pitchers: Vec<u32>,
    /// Keyed `ID{player_id}`.
    #[serde(default)]
    players: HashMap<String, BoxscorePlayer>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxscorePlayer {
    person: BoxscorePerson,
    /// `slot * 100 + substitution index`: "500" started fifth, "501" replaced him.
    #[serde(default)]
    batting_order: Option<String>,
    #[serde(default)]
    stats: PlayerStats,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoxscorePerson {
    id: u32,
    /// Only present when the box score is fetched with `hydrate=person`.
    #[serde(default)]
    pitch_hand: Option<Hand>,
}
#[derive(Deserialize)]
struct Hand {
    code: String,
}
#[derive(Deserialize, Default)]
struct PlayerStats {
    /// `{}` for a player who did not bat, so every field needs its own default.
    #[serde(default)]
    batting: BattingStat,
    #[serde(default)]
    pitching: PitchingStat,
}
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct BattingStat {
    #[serde(default)]
    strike_outs: u16,
    #[serde(default)]
    plate_appearances: u16,
}
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct PitchingStat {
    #[serde(default)]
    games_started: u16,
}

/// Posted lineups for every game on `date`. An empty slate is a legitimate answer
/// (off day, or cards simply not out yet), so this returns an empty list rather
/// than an error.
pub fn normalize_lineups(json: &str, date: NaiveDate) -> Result<Vec<GameLineup>> {
    let response: ScheduleResponse = serde_json::from_str(json)?;
    Ok(response
        .dates
        .into_iter()
        .flat_map(|day| day.games)
        .map(|game| {
            let lineups = game.lineups.unwrap_or_default();
            GameLineup {
                game_pk: game.game_pk,
                game_date: date,
                away: TeamLineup {
                    team_id: game.teams.away.team.id,
                    batters: posted(lineups.away_players),
                    probable_pitcher_id: game.teams.away.probable_pitcher.map(|person| person.id),
                },
                home: TeamLineup {
                    team_id: game.teams.home.team.id,
                    batters: posted(lineups.home_players),
                    probable_pitcher_id: game.teams.home.probable_pitcher.map(|person| person.id),
                },
            }
        })
        .collect())
}

/// A missing player list and an empty one both mean "not posted yet".
fn posted(players: Vec<Person>) -> Option<Vec<u32>> {
    (!players.is_empty()).then(|| players.into_iter().map(|person| person.id).collect())
}

/// Both sides of a completed game. Substitutes are dropped: a pinch hitter's plate
/// appearance is almost never against the starting pitcher, so folding his line
/// into a "versus that hand" rate would be attributing it to an arm he never saw.
pub fn normalize_boxscore(json: &str) -> Result<GameBoxscore> {
    let response: BoxscoreResponse = serde_json::from_str(json)?;
    let boxscore = GameBoxscore {
        away: normalize_side(response.teams.away),
        home: normalize_side(response.teams.home),
    };
    if boxscore.away.batting_order.is_empty() && boxscore.home.batting_order.is_empty() {
        return Err(PnsError::Data(
            "MLB box score has no batting order posted".into(),
        ));
    }
    Ok(boxscore)
}

fn normalize_side(side: BoxscoreSide) -> BoxscoreTeam {
    let starting_pitcher_id = starting_pitcher(&side);
    let starting_pitcher_hand = starting_pitcher_id
        .and_then(|id| side.players.get(&format!("ID{id}")))
        .and_then(|player| player.person.pitch_hand.as_ref())
        .and_then(|hand| hand.code.chars().next());
    let mut ordered: Vec<_> = side
        .players
        .values()
        .filter_map(|player| {
            let slot = starting_slot(player.batting_order.as_deref()?)?;
            Some((
                slot,
                BoxscoreBatterLine {
                    batter_id: player.person.id,
                    strikeouts: player.stats.batting.strike_outs,
                    plate_appearances: player.stats.batting.plate_appearances,
                },
            ))
        })
        .collect();
    ordered.sort_by_key(|&(slot, _)| slot);
    BoxscoreTeam {
        team_id: side.team.id,
        batting_order: ordered.into_iter().map(|(_, line)| line).collect(),
        starting_pitcher_id,
        starting_pitcher_hand,
    }
}

/// Lineup slot for a batter who *started* there. A substitution index other than
/// zero means the player took the slot over later in the game.
fn starting_slot(batting_order: &str) -> Option<u16> {
    let code: u16 = batting_order.parse().ok()?;
    (code > 0 && code.is_multiple_of(100)).then_some(code / 100)
}

/// The arm credited with the game started. MLB lists `pitchers` in appearance
/// order, so the first entry is the fallback when no stat line claims the start.
fn starting_pitcher(side: &BoxscoreSide) -> Option<u32> {
    side.pitchers
        .iter()
        .copied()
        .find(|id| {
            side.players
                .get(&format!("ID{id}"))
                .is_some_and(|player| player.stats.pitching.games_started == 1)
        })
        .or_else(|| side.pitchers.first().copied())
}

pub fn fixture_lineups(date: NaiveDate) -> Result<Vec<GameLineup>> {
    normalize_lineups(SCHEDULE_FIXTURE, date)
}

pub fn fixture_boxscore() -> Result<GameBoxscore> {
    normalize_boxscore(BOXSCORE_FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slate_date() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 7, 29).expect("valid fixture date")
    }

    #[test]
    fn normalizes_a_posted_lineup_card() {
        let games = fixture_lineups(slate_date()).unwrap();
        let posted = &games[0];
        assert_eq!(posted.game_pk, 823837);
        assert_eq!(posted.game_date, slate_date());
        assert_eq!(posted.away.team_id, 143);
        assert_eq!(posted.home.team_id, 146);
        assert_eq!(posted.away.batters.as_ref().unwrap().len(), 9);
        assert_eq!(posted.home.batters.as_ref().unwrap().len(), 9);
        assert_eq!(posted.away.batters.as_ref().unwrap()[0], 607208);
        assert_eq!(posted.away.probable_pitcher_id, Some(666200));
        assert_eq!(posted.home.probable_pitcher_id, Some(687473));
    }

    #[test]
    fn unposted_lineups_are_none_not_empty() {
        let games = fixture_lineups(slate_date()).unwrap();
        assert_eq!(games.len(), 3);
        // Empty `lineups` object: slate date, card not out yet.
        assert_eq!(games[1].away.batters, None);
        assert_eq!(games[1].home.batters, None);
        assert!(games[1].away.probable_pitcher_id.is_some());
        // No `lineups` key at all: future date.
        assert_eq!(games[2].away.batters, None);
        assert_eq!(games[2].home.batters, None);
        assert_eq!(games[2].away.probable_pitcher_id, None);
    }

    #[test]
    fn one_side_posting_first_does_not_hide_the_other() {
        let json = r#"{"dates":[{"games":[{"gamePk":1,"teams":
            {"away":{"team":{"id":143}},"home":{"team":{"id":146}}},
            "lineups":{"awayPlayers":[],"homePlayers":[{"id":5},{"id":6}]}}]}]}"#;
        let games = normalize_lineups(json, slate_date()).unwrap();
        assert_eq!(games[0].away.batters, None);
        assert_eq!(games[0].home.batters, Some(vec![5, 6]));
    }

    #[test]
    fn normalizes_boxscore_batters_starters_and_hands() {
        let boxscore = fixture_boxscore().unwrap();
        assert_eq!(boxscore.away.team_id, 143);
        assert_eq!(boxscore.away.starting_pitcher_id, Some(666200));
        assert_eq!(boxscore.away.starting_pitcher_hand, Some('L'));
        assert_eq!(boxscore.home.starting_pitcher_id, Some(687473));
        assert_eq!(boxscore.home.starting_pitcher_hand, Some('R'));
        // Lineup-slot order, not the arbitrary order of the `players` map.
        assert_eq!(
            boxscore
                .away
                .batting_order
                .iter()
                .map(|line| line.batter_id)
                .collect::<Vec<_>>(),
            vec![607208, 681082, 547180]
        );
        let harper = boxscore.away.batting_order[2];
        assert_eq!(harper.strikeouts, 2);
        assert_eq!(harper.plate_appearances, 4);
    }

    #[test]
    fn boxscore_keeps_the_starter_of_a_slot_not_his_replacement() {
        let boxscore = fixture_boxscore().unwrap();
        let ids: Vec<_> = boxscore
            .home
            .batting_order
            .iter()
            .map(|line| line.batter_id)
            .collect();
        // 640459 started the eighth slot ("800"); 665052 relieved him ("801") and
        // appears ahead of him in the feed's team-level `battingOrder` array.
        assert!(ids.contains(&640459));
        assert!(!ids.contains(&665052));
    }

    #[test]
    fn boxscore_without_a_batting_order_is_an_error() {
        let json = r#"{"teams":{"away":{"team":{"id":1},"players":{}},
                                "home":{"team":{"id":2},"players":{}}}}"#;
        assert!(normalize_boxscore(json).is_err());
    }
}
