pub mod context;
pub mod dataset;
pub mod enrich;
pub mod kalshi;
pub mod pitcher_logs;
pub mod schedule;
pub mod types;

use crate::{PnsError, Result};
use chrono::{Duration, NaiveDate};
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{Duration as StdDuration, SystemTime};

const BASE_URL: &str = "https://statsapi.mlb.com/api/v1";

#[derive(Clone)]
pub struct MlbStatsClient {
    http: Client,
    base_url: String,
}

impl MlbStatsClient {
    pub fn new() -> Result<Self> {
        let http = Client::builder()
            .timeout(StdDuration::from_secs(20))
            .user_agent("pitcher-next-start/0.1")
            .build()?;
        Ok(Self {
            http,
            base_url: BASE_URL.into(),
        })
    }

    pub fn probable_starters(&self, date: NaiveDate) -> Result<Vec<types::NextStartContext>> {
        let response = self.request_text(
            self.http
                .get(format!("{}/schedule", self.base_url))
                .query(&[
                    ("sportId", "1"),
                    ("date", &date.to_string()),
                    ("hydrate", "probablePitcher"),
                ]),
            &format!("schedule-{date}"),
            300,
        )?;
        schedule::normalize_schedule(&response, date)
    }

    pub fn pitcher_starts(
        &self,
        pitcher_id: u32,
        target_date: NaiveDate,
    ) -> Result<Vec<types::PitcherStartLog>> {
        self.pitcher_starts_range(
            pitcher_id,
            target_date - Duration::days(400),
            target_date - Duration::days(1),
        )
    }

    pub fn pitcher_starts_range(
        &self,
        pitcher_id: u32,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<Vec<types::PitcherStartLog>> {
        // A "no games yet" answer (queried before MLB posts a just-finished
        // box score) must not be cached for the full TTL, or a real result
        // minutes later would stay hidden behind a stale empty cache entry.
        let response = self.request_text_if(
            self.http
                .get(format!("{}/people/{pitcher_id}/stats", self.base_url))
                .query(&[
                    ("stats", "gameLog"),
                    ("group", "pitching"),
                    ("gameType", "R"),
                    ("startDate", &start.to_string()),
                    ("endDate", &end.to_string()),
                ]),
            &format!("logs-{pitcher_id}-{start}-{end}"),
            21_600,
            has_game_log_splits,
        )?;
        let starts = pitcher_logs::normalize_game_log(&response)?;
        if starts.is_empty() {
            return Err(PnsError::Data(format!(
                "MLB returned no prior starts for pitcher {pitcher_id}"
            )));
        }
        Ok(starts)
    }

    /// If `pitcher_id` was probable in a postponed game on `date`, returns
    /// the date MLB rescheduled it to (once announced).
    pub fn postponed_reschedule_date(
        &self,
        pitcher_id: u32,
        date: NaiveDate,
    ) -> Result<Option<NaiveDate>> {
        let response = self.request_text(
            self.http.get(format!("{}/schedule", self.base_url)).query(&[
                ("sportId", "1"),
                ("date", &date.to_string()),
                ("hydrate", "probablePitcher"),
            ]),
            &format!("reschedule-{pitcher_id}-{date}"),
            3_600,
        )?;
        schedule::find_reschedule(&response, pitcher_id)
    }

    pub fn season_starters(&self, season: i32, minimum_starts: u16) -> Result<Vec<u32>> {
        let response = self.request_text(
            self.http.get(format!("{}/stats", self.base_url)).query(&[
                ("stats", "season"),
                ("group", "pitching"),
                ("season", &season.to_string()),
                ("gameType", "R"),
                ("playerPool", "ALL"),
                ("limit", "1000"),
                ("sportIds", "1"),
            ]),
            &format!("season-starters-{season}-{minimum_starts}"),
            86_400,
        )?;
        dataset::normalize_season_starters(&response, minimum_starts)
    }

    pub fn pitcher_hands(&self, pitcher_ids: &[u32]) -> Result<HashMap<u32, char>> {
        let ids = pitcher_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let response = self.request_text(
            self.http
                .get(format!("{}/people", self.base_url))
                .query(&[("personIds", ids.as_str())]),
            &format!("hands-{ids}"),
            604_800,
        )?;
        context::normalize_pitcher_hands(&response)
    }

    pub fn slate_context(&self, season: i32) -> Result<context::SlateContext> {
        let fetch = |code: &str| -> Result<String> {
            self.request_text(
                self.http
                    .get(format!("{}/teams/stats", self.base_url))
                    .query(&[
                        ("stats", "statSplits"),
                        ("group", "hitting"),
                        ("season", &season.to_string()),
                        ("gameType", "R"),
                        ("sportIds", "1"),
                        ("sitCodes", code),
                    ]),
                &format!("team-context-{season}-{code}"),
                3_600,
            )
        };
        context::normalize_slate_context(&fetch("vl")?, &fetch("vr")?, &fetch("h")?, &fetch("a")?)
    }

    pub fn savant_fastball_velocity(
        &self,
        pitcher_id: u32,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Result<HashMap<NaiveDate, f64>> {
        let response = self.request_text(
            self.http
                .get("https://baseballsavant.mlb.com/statcast_search/csv")
                .query(&[
                    ("all", "true"),
                    ("player_type", "pitcher"),
                    ("pitchers_lookup[]", &pitcher_id.to_string()),
                    ("game_date_gt", &start.to_string()),
                    ("game_date_lt", &end.to_string()),
                    ("hfGT", "R|"),
                    ("hfPT", "FF|SI|"),
                    ("type", "details"),
                    ("group_by", "name"),
                ]),
            &format!("savant-{pitcher_id}-{start}-{end}"),
            21_600,
        )?;
        enrich::normalize_savant_velocity(&response)
    }

    fn request_text(
        &self,
        request: reqwest::blocking::RequestBuilder,
        cache_key: &str,
        ttl_seconds: u64,
    ) -> Result<String> {
        self.request_text_if(request, cache_key, ttl_seconds, |_| true)
    }

    /// Like [`Self::request_text`], but a freshly-fetched response is only
    /// written to the cache when `cacheable` returns true for it.
    fn request_text_if(
        &self,
        request: reqwest::blocking::RequestBuilder,
        cache_key: &str,
        ttl_seconds: u64,
        cacheable: impl Fn(&str) -> bool,
    ) -> Result<String> {
        let cache_path = cache_path(cache_key);
        if cache_is_fresh(&cache_path, ttl_seconds) {
            return Ok(std::fs::read_to_string(cache_path)?);
        }
        let mut last_error = String::new();
        for attempt in 0..3 {
            let candidate = request.try_clone().ok_or_else(|| {
                PnsError::Data("MLB request could not be cloned for retry".into())
            })?;
            match candidate
                .send()
                .and_then(|response| response.error_for_status())
                .and_then(|response| response.text())
            {
                Ok(text) => {
                    if cacheable(&text) {
                        if let Some(parent) = cache_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::write(&cache_path, &text)?;
                    }
                    return Ok(text);
                }
                Err(error) => {
                    last_error = error.to_string();
                    if attempt < 2 {
                        std::thread::sleep(StdDuration::from_millis(250 * (attempt + 1) as u64));
                    }
                }
            }
        }
        if cache_path.exists() {
            return Ok(std::fs::read_to_string(cache_path)?);
        }
        Err(PnsError::Data(format!(
            "MLB feed request failed after 3 attempts: {last_error}"
        )))
    }
}

/// True when a `stats?stats=gameLog` response actually contains at least
/// one game (i.e. the box score has been posted), so an empty "not final
/// yet" answer never gets locked into the cache.
fn has_game_log_splits(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| value["stats"][0]["splits"].as_array().map(|splits| !splits.is_empty()))
        .unwrap_or(false)
}

fn cache_path(key: &str) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    std::env::temp_dir()
        .join("pitcher-next-start/cache")
        .join(format!("{:016x}.cache", hasher.finish()))
}

fn cache_is_fresh(path: &Path, ttl_seconds: u64) -> bool {
    path.metadata()
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|age| age.as_secs() <= ttl_seconds)
}

#[cfg(test)]
mod tests {
    #[test]
    fn game_log_splits_detects_empty_vs_populated_responses() {
        let empty = r#"{"stats":[{"type":{"displayName":"gameLog"},"splits":[]}]}"#;
        let populated = r#"{"stats":[{"type":{"displayName":"gameLog"},"splits":[{"date":"2026-07-21"}]}]}"#;
        assert!(!super::has_game_log_splits(empty));
        assert!(super::has_game_log_splits(populated));
        assert!(!super::has_game_log_splits("not json"));
    }
}
