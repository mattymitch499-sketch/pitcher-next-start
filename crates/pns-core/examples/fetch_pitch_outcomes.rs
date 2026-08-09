//! Downloads per-start pitch-outcome counts (pitches, swinging strikes, called
//! strikes) and writes the compact `data/processed/pitch_outcomes.csv`.
//!
//! Swinging-strike rate is the best-known leading indicator of future strikeout
//! rate, and the model has never seen it — it predicts strikeouts from past
//! strikeouts. This fetches the raw material for [`whiff_signal`], which decides
//! whether the signal is worth a feature-contract bump and a retrain.
//!
//! # Why this is expensive, and what it does about it
//!
//! [`MlbStatsClient::savant_pitch_outcomes`] drops the `hfPT=FF|SI|` filter the
//! velocity path uses, because a swinging-strike *rate* needs every pitch in the
//! denominator. That makes each response several times larger. Two mitigations:
//!
//! - Only pitchers who actually appear in the study's evaluation window are
//!   fetched, rather than every pitcher in the dataset.
//! - The compact CSV is the durable artifact. Once it exists the raw Statcast
//!   cache can be deleted and nothing downstream notices.
//!
//! **This example never writes to `training_starts*.csv`.** The dataset rebuild
//! path (`enrich_dataset`) rewrites that file in place; this one deliberately
//! stays out of its way.
//!
//! Usage:
//! `cargo run --release -p pns-core --example fetch_pitch_outcomes -- [--dataset PATH] [--from YYYY-MM-DD] [--eval-from YYYY-MM-DD] [--through YYYY-MM-DD] [--limit N] [--out PATH]`

use chrono::NaiveDate;
use pns_core::data::MlbStatsClient;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Concurrent Savant requests. The responses are large; stay well under what the
/// velocity path uses.
const CHUNK: usize = 4;

struct Args {
    dataset: PathBuf,
    /// Earliest pitch date to fetch. History before the evaluation window is
    /// needed because the rolling feature looks back five starts.
    from: NaiveDate,
    /// Only pitchers with a start on or after this date are fetched.
    eval_from: NaiveDate,
    through: NaiveDate,
    limit: Option<usize>,
    out: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    let pitchers: Vec<u32> = starts
        .iter()
        .filter(|row| row.game_date >= args.eval_from && row.game_date <= args.through)
        .map(|row| row.pitcher_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let pitchers: Vec<u32> = match args.limit {
        Some(limit) => pitchers.into_iter().take(limit).collect(),
        None => pitchers,
    };
    eprintln!(
        "{} pitchers with a start in {}..={}; fetching pitches from {}",
        pitchers.len(),
        args.eval_from,
        args.through,
        args.from
    );

    let client = MlbStatsClient::new()?;
    let mut rows: Vec<(u32, NaiveDate, pns_core::data::enrich::PitchOutcomes)> = Vec::new();
    let mut failures = Vec::new();
    let batches = pitchers.len().div_ceil(CHUNK);

    for (index, chunk) in pitchers.chunks(CHUNK).enumerate() {
        let results: Vec<_> = std::thread::scope(|scope| {
            chunk
                .iter()
                .map(|&pitcher_id| {
                    let client = client.clone();
                    scope.spawn(move || {
                        (
                            pitcher_id,
                            client.savant_pitch_outcomes(pitcher_id, args.from, args.through),
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("Savant worker panicked"))
                .collect()
        });
        for (pitcher_id, result) in results {
            match result {
                Ok(by_date) => {
                    for (date, outcomes) in by_date {
                        rows.push((pitcher_id, date, outcomes));
                    }
                }
                Err(error) => failures.push(format!("{pitcher_id}: {error}")),
            }
        }
        if index % 10 == 9 || index + 1 == batches {
            eprintln!(
                "  batch {}/{batches}  ({} start-rows so far, {} failures)",
                index + 1,
                rows.len(),
                failures.len()
            );
        }
    }

    // Same discipline as `enrich_dataset`: a partial download that looks
    // complete is worse than no download, because every rate silently shifts.
    if !failures.is_empty() {
        return Err(format!(
            "refusing to write partial outcomes; {} requests failed: {}",
            failures.len(),
            failures
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ")
        )
        .into());
    }

    rows.sort_by_key(|(pitcher_id, date, _)| (*pitcher_id, *date));
    let mut writer = csv::Writer::from_path(&args.out)?;
    writer.write_record([
        "pitcher_id",
        "game_date",
        "pitches",
        "swinging_strikes",
        "called_strikes",
    ])?;
    for (pitcher_id, date, outcomes) in &rows {
        writer.write_record([
            pitcher_id.to_string(),
            date.to_string(),
            outcomes.pitches.to_string(),
            outcomes.swinging_strikes.to_string(),
            outcomes.called_strikes.to_string(),
        ])?;
    }
    writer.flush()?;

    let pitches: u64 = rows.iter().map(|(_, _, o)| u64::from(o.pitches)).sum();
    let whiffs: u64 = rows
        .iter()
        .map(|(_, _, o)| u64::from(o.swinging_strikes))
        .sum();
    eprintln!(
        "wrote {} start-rows ({} pitches, league SwStr {:.4}) to {}",
        rows.len(),
        pitches,
        whiffs as f64 / pitches.max(1) as f64,
        args.out.display()
    );
    Ok(())
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dataset: root.join("data/processed/training_starts_v3.csv"),
        from: NaiveDate::from_ymd_opt(2025, 3, 1).expect("valid default fetch start"),
        eval_from: NaiveDate::from_ymd_opt(2026, 5, 1).expect("valid default eval start"),
        through: NaiveDate::from_ymd_opt(2026, 7, 16).expect("valid default end"),
        limit: None,
        out: root.join("data/processed/pitch_outcomes.csv"),
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
            "--from" => args.from = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--eval-from" => args.eval_from = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--through" => args.through = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--limit" => args.limit = Some(value()?.parse()?),
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}
