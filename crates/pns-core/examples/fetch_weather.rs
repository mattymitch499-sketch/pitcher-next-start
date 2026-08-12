//! Backfills observed game-time weather for every date the dataset covers and
//! writes the compact `data/processed/game_weather.csv`.
//!
//! The schedule feed hydrates a whole date *range* in one request, so a full
//! backfill is a few dozen calls. Responses for ranges safely in the past cache
//! for 30 days — completed weather never changes — which makes a re-run free.
//!
//! Never touches `training_starts*.csv`. Like `fetch_pitch_outcomes`, the
//! compact CSV is the durable artifact and the HTTP cache stays disposable.
//!
//! **Observed, not forecast.** MLB fills the field at game time. This can build
//! history; it cannot serve live scoring.
//!
//! Usage:
//! `cargo run --release -p pns-core --example fetch_weather -- [--dataset PATH] [--out PATH] [--min-coverage F]`

use chrono::{Datelike, NaiveDate};
use pns_core::data::MlbStatsClient;
use pns_core::data::weather::GameWeather;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

struct Args {
    dataset: PathBuf,
    out: PathBuf,
    /// Refuse to write below this share of dataset starts matched to a game with
    /// weather. A silently partial backfill would look like a real null result.
    min_coverage: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    if starts.is_empty() {
        return Err("dataset has no rows".into());
    }
    let first = starts
        .iter()
        .map(|start| start.game_date)
        .min()
        .expect("nonempty");
    let last = starts
        .iter()
        .map(|start| start.game_date)
        .max()
        .expect("nonempty");
    eprintln!(
        "{} starts spanning {first} .. {last}",
        starts.len()
    );

    let client = MlbStatsClient::new()?;
    let mut rows: Vec<GameWeather> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for (start, end) in month_chunks(first, last) {
        // A chunk that fails aborts the run. A partial backfill that still wrote
        // its file would shift every rate computed from it.
        let chunk = client.schedule_weather(start, end).map_err(|error| {
            format!("weather fetch failed for {start} .. {end}: {error}")
        })?;
        eprintln!("  {start} .. {end}: {} games with weather", chunk.len());
        for row in chunk {
            if seen.insert(row.game_pk) {
                rows.push(row);
            }
        }
    }
    if rows.is_empty() {
        return Err("no weather rows fetched".into());
    }

    // Coverage is measured against the starts that actually matter, by game_pk.
    let by_game: HashMap<u32, &GameWeather> =
        rows.iter().map(|row| (row.game_pk, row)).collect();
    let matched = starts
        .iter()
        .filter(|start| by_game.contains_key(&start.game_pk))
        .count();
    let coverage = matched as f64 / starts.len() as f64;
    eprintln!(
        "{} weather rows; {matched}/{} dataset starts matched ({:.2}% coverage)",
        rows.len(),
        starts.len(),
        coverage * 100.0
    );

    let indoor = rows.iter().filter(|row| row.indoor).count();
    let with_temp = rows.iter().filter(|row| row.temperature_f.is_some()).count();
    let temps: Vec<f64> = rows.iter().filter_map(|row| row.temperature_f).collect();
    let mean_temp = temps.iter().sum::<f64>() / temps.len().max(1) as f64;
    eprintln!(
        "  indoor {indoor}, with temperature {with_temp}, mean temperature {mean_temp:.1}F"
    );

    if coverage < args.min_coverage {
        return Err(format!(
            "coverage {:.2}% is below the {:.2}% floor; refusing to write a partial backfill",
            coverage * 100.0,
            args.min_coverage * 100.0
        )
        .into());
    }

    rows.sort_by_key(|row| (row.game_date, row.game_pk));
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut writer = csv::Writer::from_path(&args.out)?;
    for row in &rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    eprintln!("wrote {} rows to {}", rows.len(), args.out.display());
    Ok(())
}

/// Calendar-month ranges covering `[first, last]` inclusive.
///
/// Month sized rather than season sized so a single failed chunk costs one
/// month, and so no request asks for a range large enough to be truncated.
fn month_chunks(first: NaiveDate, last: NaiveDate) -> Vec<(NaiveDate, NaiveDate)> {
    let mut chunks = Vec::new();
    let mut year = first.year();
    let mut month = first.month();
    loop {
        let start = NaiveDate::from_ymd_opt(year, month, 1).expect("valid month start");
        let (next_year, next_month) = if month == 12 {
            (year + 1, 1)
        } else {
            (year, month + 1)
        };
        let end = NaiveDate::from_ymd_opt(next_year, next_month, 1)
            .expect("valid next month")
            .pred_opt()
            .expect("valid month end");
        chunks.push((start.max(first), end.min(last)));
        if end >= last {
            break;
        }
        year = next_year;
        month = next_month;
    }
    chunks
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dataset: root.join("data/processed/training_starts_v4.csv"),
        out: root.join("data/processed/game_weather.csv"),
        min_coverage: 0.90,
    };
    let mut raw = std::env::args().skip(1);
    while let Some(argument) = raw.next() {
        let (name, inline) = match argument.split_once('=') {
            Some((name, value)) => (name.to_string(), Some(value.to_string())),
            None => (argument, None),
        };
        let mut value = || -> Result<String, Box<dyn std::error::Error>> {
            inline
                .clone()
                .or_else(|| raw.next())
                .ok_or_else(|| format!("{name} needs a value").into())
        };
        match name.as_str() {
            "--dataset" => args.dataset = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            "--min-coverage" => args.min_coverage = value()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    /// Chunks must tile the span exactly: no gap, no overlap, clipped to the
    /// requested endpoints. A gap would silently drop a month of games.
    #[test]
    fn month_chunks_tile_the_span_without_gaps() {
        let first = NaiveDate::from_ymd_opt(2023, 3, 30).unwrap();
        let last = NaiveDate::from_ymd_opt(2023, 6, 15).unwrap();
        let chunks = super::month_chunks(first, last);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0], (first, NaiveDate::from_ymd_opt(2023, 3, 31).unwrap()));
        assert_eq!(chunks[3].1, last);
        for pair in chunks.windows(2) {
            assert_eq!(
                pair[1].0,
                pair[0].1.succ_opt().unwrap(),
                "chunks must be contiguous"
            );
        }
    }

    /// A span inside one month is a single chunk, not an infinite loop.
    #[test]
    fn a_single_month_span_yields_one_chunk() {
        let first = NaiveDate::from_ymd_opt(2026, 7, 2).unwrap();
        let last = NaiveDate::from_ymd_opt(2026, 7, 16).unwrap();
        assert_eq!(super::month_chunks(first, last), vec![(first, last)]);
    }
}
