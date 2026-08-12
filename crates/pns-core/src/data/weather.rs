//! Game-time weather from the MLB schedule feed.
//!
//! `/schedule?...&hydrate=weather` carries `condition`, `temp` and `wind` for
//! every game, and a date *range* hydrates in one request — the whole dataset
//! span is a few dozen calls rather than one per game.
//!
//! **This is observed weather, not a forecast.** MLB populates the field at game
//! time; a future date returns an empty object. So this module can backfill
//! history for free, but it cannot serve live scoring — that would need a
//! forecast source, which is a separate decision.
//!
//! One thing worth knowing before parsing: MLB already reports wind direction
//! **relative to the ballpark** (`"8 mph, In From RF"`, `"6 mph, L To R"`), so
//! bucketing is a matter of reading their words, not computing park orientation.

use crate::Result;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Which way the wind blows relative to the park.
///
/// `Calm` is a real reading (`"0 mph, None"`, overwhelmingly indoor games) and
/// is kept distinct from `Unknown`, which means the string did not parse. A
/// study that merged the two would treat a parser miss as still air.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WindDirection {
    /// Blowing in from an outfield position — suppresses carry.
    In,
    /// Blowing out toward an outfield position — aids carry.
    Out,
    /// Left-to-right or right-to-left across the field.
    Cross,
    /// Explicitly no wind.
    Calm,
    /// Unparsed. Never treat as calm.
    Unknown,
}

impl WindDirection {
    /// Buckets MLB's direction phrase. Matching is done on the leading word
    /// because the suffix names an outfield position (`In From RF`, `In From
    /// CF`) that carries no extra information about carry.
    pub fn parse(raw: &str) -> Self {
        let trimmed = raw.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower == "none" || lower.is_empty() {
            return Self::Calm;
        }
        if lower.starts_with("in") {
            return Self::In;
        }
        if lower.starts_with("out") {
            return Self::Out;
        }
        if lower.contains("to r") || lower.contains("to l") {
            return Self::Cross;
        }
        Self::Unknown
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::Cross => "cross",
            Self::Calm => "calm",
            Self::Unknown => "unknown",
        }
    }
}

/// One game's conditions at first pitch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameWeather {
    pub game_pk: u32,
    pub game_date: NaiveDate,
    pub home_team_id: u32,
    /// Fahrenheit. `None` when MLB reported no temperature.
    pub temperature_f: Option<f64>,
    pub condition: String,
    pub wind_speed_mph: Option<f64>,
    pub wind_direction: WindDirection,
    /// A dome or a closed roof. Indoor games still report a temperature, but it
    /// is a thermostat reading — wind and sky carry no information there, which
    /// is exactly why this flag exists rather than being inferred downstream.
    pub indoor: bool,
}

#[derive(Deserialize)]
struct ScheduleResponse {
    #[serde(default)]
    dates: Vec<ScheduleDate>,
}
#[derive(Deserialize)]
struct ScheduleDate {
    #[serde(default)]
    date: Option<String>,
    #[serde(default)]
    games: Vec<Game>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Game {
    #[serde(default)]
    game_pk: u32,
    #[serde(default)]
    teams: Option<Teams>,
    #[serde(default)]
    weather: Option<Weather>,
}
#[derive(Deserialize)]
struct Teams {
    home: Side,
}
#[derive(Deserialize)]
struct Side {
    team: Team,
}
#[derive(Deserialize)]
struct Team {
    id: u32,
}
#[derive(Deserialize)]
struct Weather {
    #[serde(default)]
    condition: Option<String>,
    #[serde(default)]
    temp: Option<String>,
    #[serde(default)]
    wind: Option<String>,
}

/// Splits `"8 mph, In From RF"` into speed and bucketed direction.
///
/// A missing or unparseable speed yields `None` rather than 0.0 — zero is a real
/// reading and must not be manufactured.
fn parse_wind(raw: &str) -> (Option<f64>, WindDirection) {
    let (speed_part, direction_part) = match raw.split_once(',') {
        Some((speed, direction)) => (speed, direction),
        None => (raw, ""),
    };
    let speed = speed_part
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<f64>().ok());
    (speed, WindDirection::parse(direction_part))
}

/// Every game carrying weather in a hydrated schedule response.
///
/// Games with an empty `weather` object are skipped rather than emitted with
/// blanks: an unplayed game has no observed conditions, and a row of `None`s
/// would be indistinguishable from a game played in unrecorded weather.
pub fn normalize_schedule_weather(json: &str) -> Result<Vec<GameWeather>> {
    let response: ScheduleResponse = serde_json::from_str(json)?;
    let mut rows = Vec::new();
    for day in response.dates {
        let Some(date) = day
            .date
            .as_deref()
            .and_then(|date| NaiveDate::parse_from_str(date, "%Y-%m-%d").ok())
        else {
            continue;
        };
        for game in day.games {
            if game.game_pk == 0 {
                continue;
            }
            let Some(weather) = game.weather else {
                continue;
            };
            let condition = weather.condition.unwrap_or_default();
            if condition.is_empty() && weather.temp.is_none() && weather.wind.is_none() {
                continue;
            }
            let Some(home_team_id) = game.teams.map(|teams| teams.home.team.id) else {
                continue;
            };
            let (wind_speed_mph, wind_direction) = match weather.wind.as_deref() {
                Some(raw) => parse_wind(raw),
                None => (None, WindDirection::Unknown),
            };
            let indoor = {
                let lower = condition.to_ascii_lowercase();
                lower.contains("dome") || lower.contains("roof closed")
            };
            rows.push(GameWeather {
                game_pk: game.game_pk,
                game_date: date,
                home_team_id,
                temperature_f: weather
                    .temp
                    .as_deref()
                    .and_then(|value| value.trim().parse::<f64>().ok()),
                condition,
                wind_speed_mph,
                wind_direction,
                indoor,
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::WindDirection;

    /// Every direction phrase MLB actually emits, pinned to its bucket. A
    /// mis-bucketed wind would flip the sign of the feature it feeds, so each
    /// one is named rather than covered by a pattern.
    #[test]
    fn wind_phrases_bucket_by_their_leading_word() {
        for phrase in ["In From RF", "In From CF", "In From LF", " in from rf "] {
            assert_eq!(WindDirection::parse(phrase), WindDirection::In, "{phrase}");
        }
        for phrase in ["Out To CF", "Out To LF", "Out To RF"] {
            assert_eq!(WindDirection::parse(phrase), WindDirection::Out, "{phrase}");
        }
        for phrase in ["L To R", "R To L"] {
            assert_eq!(
                WindDirection::parse(phrase),
                WindDirection::Cross,
                "{phrase}"
            );
        }
        assert_eq!(WindDirection::parse("None"), WindDirection::Calm);
        assert_eq!(WindDirection::parse(""), WindDirection::Calm);
        // Anything unrecognised must NOT masquerade as still air.
        assert_eq!(WindDirection::parse("Varies"), WindDirection::Unknown);
    }

    #[test]
    fn parses_speed_and_direction_together() {
        assert_eq!(
            super::parse_wind("8 mph, In From RF"),
            (Some(8.0), WindDirection::In)
        );
        assert_eq!(super::parse_wind("0 mph, None"), (Some(0.0), WindDirection::Calm));
        // No speed at all must stay None rather than becoming a manufactured 0.
        assert_eq!(super::parse_wind(", Out To CF"), (None, WindDirection::Out));
    }

    /// A dome and a closed roof are indoor; an open-air game is not, and an
    /// unplayed game contributes no row at all.
    #[test]
    fn indoor_games_are_flagged_and_empty_weather_is_skipped() {
        let json = r#"{"dates":[{"date":"2026-06-01","games":[
            {"gamePk":1,"teams":{"home":{"team":{"id":147}}},
             "weather":{"condition":"Dome","temp":"72","wind":"0 mph, None"}},
            {"gamePk":2,"teams":{"home":{"team":{"id":112}}},
             "weather":{"condition":"Roof Closed","temp":"73","wind":"0 mph, None"}},
            {"gamePk":3,"teams":{"home":{"team":{"id":138}}},
             "weather":{"condition":"Partly Cloudy","temp":"81","wind":"8 mph, Out To CF"}},
            {"gamePk":4,"teams":{"home":{"team":{"id":121}}},"weather":{}}
        ]}]}"#;
        let rows = super::normalize_schedule_weather(json).unwrap();
        assert_eq!(rows.len(), 3, "the unplayed game must not produce a row");
        assert!(rows[0].indoor);
        assert!(rows[1].indoor);
        assert!(!rows[2].indoor);
        assert_eq!(rows[2].temperature_f, Some(81.0));
        assert_eq!(rows[2].wind_speed_mph, Some(8.0));
        assert_eq!(rows[2].wind_direction, WindDirection::Out);
        assert_eq!(rows[2].home_team_id, 138);
    }
}
