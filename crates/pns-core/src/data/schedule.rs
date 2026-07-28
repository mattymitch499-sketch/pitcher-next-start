use super::types::NextStartContext;
use crate::{PnsError, Result};
use chrono::NaiveDate;
use serde::Deserialize;

#[derive(Deserialize)]
struct ScheduleResponse {
    #[serde(default)]
    dates: Vec<ScheduleDate>,
}
#[derive(Deserialize)]
struct ScheduleDate {
    #[serde(default)]
    games: Vec<Game>,
}
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Game {
    teams: Teams,
    game_date: Option<String>,
    venue: Option<Venue>,
    #[serde(default)]
    status: Option<Status>,
    #[serde(default)]
    reschedule_game_date: Option<String>,
}
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Status {
    #[serde(default)]
    detailed_state: String,
}
#[derive(Deserialize, Clone)]
struct Teams {
    away: Side,
    home: Side,
}
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Side {
    team: Team,
    probable_pitcher: Option<Person>,
}
#[derive(Deserialize, Clone)]
struct Team {
    id: u32,
    name: String,
}
#[derive(Deserialize, Clone)]
struct Venue {
    name: String,
}
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Person {
    id: u32,
    full_name: String,
}

pub fn normalize_schedule(json: &str, date: NaiveDate) -> Result<Vec<NextStartContext>> {
    let response: ScheduleResponse = serde_json::from_str(json)?;
    let mut contexts = Vec::new();
    for game in response.dates.into_iter().flat_map(|day| day.games) {
        let venue_name = game
            .venue
            .as_ref()
            .map(|venue| venue.name.clone())
            .unwrap_or_else(|| "Venue TBD".into());
        if let Some(pitcher) = game.teams.away.probable_pitcher {
            contexts.push(NextStartContext {
                pitcher_id: pitcher.id,
                pitcher_name: pitcher.full_name,
                game_date: date,
                opponent_id: game.teams.home.team.id,
                home_team_id: game.teams.home.team.id,
                home: false,
                park_factor: 1.0,
                opponent_offense: 1.0,
                pitcher_team_name: game.teams.away.team.name.clone(),
                opponent_team_name: game.teams.home.team.name.clone(),
                venue_name: venue_name.clone(),
                game_time: game.game_date.clone(),
                pitcher_hand: None,
            });
        }
        if let Some(pitcher) = game.teams.home.probable_pitcher {
            contexts.push(NextStartContext {
                pitcher_id: pitcher.id,
                pitcher_name: pitcher.full_name,
                game_date: date,
                opponent_id: game.teams.away.team.id,
                home_team_id: game.teams.home.team.id,
                home: true,
                park_factor: 1.0,
                opponent_offense: 1.0,
                pitcher_team_name: game.teams.home.team.name.clone(),
                opponent_team_name: game.teams.away.team.name.clone(),
                venue_name,
                game_time: game.game_date,
                pitcher_hand: None,
            });
        }
    }
    if contexts.is_empty() {
        return Err(PnsError::Data(format!(
            "MLB has no probable starters listed for {date}"
        )));
    }
    contexts.sort_by_key(|context| (context.pitcher_name.clone(), context.pitcher_id));
    Ok(contexts)
}

/// If `pitcher_id` was the probable starter in a postponed game on this
/// schedule date, returns the date MLB rescheduled it to (once announced).
pub fn find_reschedule(json: &str, pitcher_id: u32) -> Result<Option<NaiveDate>> {
    let response: ScheduleResponse = serde_json::from_str(json)?;
    for game in response.dates.into_iter().flat_map(|day| day.games) {
        let is_pitcher = game
            .teams
            .away
            .probable_pitcher
            .as_ref()
            .is_some_and(|pitcher| pitcher.id == pitcher_id)
            || game
                .teams
                .home
                .probable_pitcher
                .as_ref()
                .is_some_and(|pitcher| pitcher.id == pitcher_id);
        let postponed = game
            .status
            .as_ref()
            .is_some_and(|status| status.detailed_state == "Postponed");
        if is_pitcher
            && postponed
            && let Some(reschedule) = game.reschedule_game_date
        {
            return Ok(NaiveDate::parse_from_str(&reschedule, "%Y-%m-%d").ok());
        }
    }
    Ok(None)
}

pub fn fixture_contexts(date: NaiveDate) -> Result<Vec<NextStartContext>> {
    normalize_schedule(
        include_str!("../../../../data/fixtures/mlb_schedule.json"),
        date,
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalizes_both_probable_starters() {
        let date = chrono::NaiveDate::from_ymd_opt(2025, 7, 12).unwrap();
        let rows = super::fixture_contexts(date).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|row| row.home && row.pitcher_id == 608331));
        assert!(rows.iter().any(|row| !row.home && row.opponent_id == 147));
        assert!(rows.iter().all(|row| !row.pitcher_team_name.is_empty()));
    }

    #[test]
    fn finds_reschedule_date_for_a_postponed_starter() {
        let json = r#"{"dates":[{"games":[
            {"gameDate":"2026-07-21T23:05:00Z","status":{"detailedState":"Postponed"},"rescheduleGameDate":"2026-07-22",
             "teams":{"away":{"team":{"id":134,"name":"Pittsburgh Pirates"},"probablePitcher":{"id":696149,"fullName":"Bubba Chandler"}},
                      "home":{"team":{"id":147,"name":"New York Yankees"},"probablePitcher":{"id":701542,"fullName":"Will Warren"}}},
             "venue":{"name":"Yankee Stadium"}}
        ]}]}"#;
        assert_eq!(
            super::find_reschedule(json, 701542).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2026, 7, 22)
        );
        assert_eq!(
            super::find_reschedule(json, 696149).unwrap(),
            chrono::NaiveDate::from_ymd_opt(2026, 7, 22)
        );
        assert_eq!(super::find_reschedule(json, 999999).unwrap(), None);
    }
}
