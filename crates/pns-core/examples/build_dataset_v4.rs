//! Joins pitch-outcome counts onto the v3 dataset and writes v4.
//!
//! v4 = v3 plus `pitches` / `swinging_strikes` / `called_strikes` per start, the
//! raw material for the `recent_swinging_strike_rate` and
//! `recent_called_strike_rate` features added in feature schema 2.
//!
//! Reads `data/processed/pitch_outcomes.csv` rather than re-fetching, and
//! **never writes to `training_starts_v3.csv` or `training_starts.csv`**.
//! `examples/enrich_dataset.rs` rewrites its input in place; this deliberately
//! does not, so the frozen datasets the older artifacts were trained on stay
//! byte-identical.
//!
//! `--through` truncates the output, which is how the historical training set
//! for `pns-model-hist-*` is produced.
//!
//! Usage:
//! `cargo run --release -p pns-core --example build_dataset_v4 -- [--dataset PATH] [--outcomes PATH] [--through YYYY-MM-DD] [--out PATH]`

use chrono::NaiveDate;
use std::collections::HashMap;
use std::path::PathBuf;

/// Below this share of starts carrying pitch outcomes the join is broken rather
/// than merely incomplete, and writing would bake a silent hole into every
/// downstream rate.
const MIN_COVERAGE: f64 = 0.90;

#[derive(serde::Deserialize)]
struct OutcomeRow {
    pitcher_id: u32,
    game_date: NaiveDate,
    pitches: u32,
    swinging_strikes: u32,
    called_strikes: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut dataset = root.join("data/processed/training_starts_v3.csv");
    let mut outcomes = root.join("data/processed/pitch_outcomes.csv");
    let mut out = root.join("data/processed/training_starts_v4.csv");
    let mut through: Option<NaiveDate> = None;

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
            "--dataset" => dataset = PathBuf::from(value()?),
            "--outcomes" => outcomes = PathBuf::from(value()?),
            "--through" => through = Some(NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?),
            "--out" => out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if out == dataset {
        return Err("refusing to overwrite the input dataset".into());
    }

    let mut rows = pns_core::data::dataset::read_csv(&dataset)?;
    let mut by_start: HashMap<(u32, NaiveDate), OutcomeRow> = HashMap::new();
    for record in csv::Reader::from_path(&outcomes)?.deserialize::<OutcomeRow>() {
        let row = record?;
        by_start.insert((row.pitcher_id, row.game_date), row);
    }
    eprintln!(
        "{} starts, {} pitch-outcome rows available",
        rows.len(),
        by_start.len()
    );

    if let Some(through) = through {
        rows.retain(|row| row.game_date <= through);
        eprintln!("truncated to {} starts through {through}", rows.len());
    }

    let mut joined = 0usize;
    for row in &mut rows {
        if let Some(outcome) = by_start.get(&(row.pitcher_id, row.game_date)) {
            row.pitches = Some(outcome.pitches);
            row.swinging_strikes = Some(outcome.swinging_strikes);
            row.called_strikes = Some(outcome.called_strikes);
            joined += 1;
        }
    }
    let coverage = joined as f64 / rows.len().max(1) as f64;
    eprintln!(
        "joined {joined} of {} starts ({:.2}% coverage)",
        rows.len(),
        coverage * 100.0
    );
    if coverage < MIN_COVERAGE {
        return Err(format!(
            "coverage {:.2}% is below the {:.0}% floor; refusing to write a dataset \
             whose whiff features would be mostly sentinel",
            coverage * 100.0,
            MIN_COVERAGE * 100.0
        )
        .into());
    }

    // Sanity check the joined rates against the league figure the fetch reported.
    let pitches: u64 = rows
        .iter()
        .filter_map(|row| row.pitches)
        .map(u64::from)
        .sum();
    let whiffs: u64 = rows
        .iter()
        .filter_map(|row| row.swinging_strikes)
        .map(u64::from)
        .sum();
    eprintln!(
        "joined league SwStr {:.4} over {pitches} pitches",
        whiffs as f64 / pitches.max(1) as f64
    );

    pns_core::data::dataset::write_csv(&rows, &out)?;
    println!("wrote {} starts to {}", rows.len(), out.display());
    Ok(())
}
