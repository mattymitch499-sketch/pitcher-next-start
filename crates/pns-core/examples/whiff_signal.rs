//! Does swinging-strike rate add anything the model's 16 features miss?
//!
//! A study, not a build. It fits the k-rate regression twice on identical rows —
//! once with the shipped feature set, once with rolling SwStr% and CSW% appended
//! — and scores both on a *later* window. Nothing here changes the feature
//! contract, the artifact schema, or any shipped file.
//!
//! The bar, fixed before the numbers existed: the added features must improve
//! **held-out predictive log-likelihood**, with at least one coefficient clearly
//! separated from zero. A better point estimate alone does not count. That is
//! exactly the trap the removal-hazard work fell into, where a 2.25x in-sample
//! gain produced slightly *worse* out-of-sample forecasting.
//!
//! Requires `data/processed/pitch_outcomes.csv` from `fetch_pitch_outcomes`.
//!
//! Usage:
//! `cargo run --release -p pns-core --example whiff_signal -- [--dataset PATH] [--outcomes PATH] [--split YYYY-MM-DD] [--through YYYY-MM-DD] [--out PATH]`

use chrono::NaiveDate;
use pns_core::data::enrich::PitchOutcomes;
use pns_core::data::types::PitcherStartLog;
use pns_core::features;
use pns_core::model::whiff::{WhiffComparison, compare_k_rate};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

/// Rolling window, matching what the model is trained on.
const HISTORY_WINDOW: usize = 5;

/// Minimum pitches across the prior window before a whiff rate is trusted.
/// Roughly two starts. Below this the rate is mostly sampling noise, and a row
/// that cannot support the feature is dropped from *both* arms rather than
/// filled with a neutral value — a filled row would dilute the very signal the
/// study is trying to detect.
const MIN_PRIOR_PITCHES: u32 = 200;

const ADDED_FEATURES: [&str; 2] = ["recent_swinging_strike_rate", "recent_called_plus_swinging_rate"];

struct Args {
    dataset: PathBuf,
    outcomes: PathBuf,
    split: NaiveDate,
    through: NaiveDate,
    out: PathBuf,
}

#[derive(Serialize)]
struct Report {
    dataset: String,
    split: NaiveDate,
    through: NaiveDate,
    history_window: usize,
    min_prior_pitches: u32,
    rows_with_whiff_history: usize,
    rows_dropped_for_thin_history: usize,
    league_swinging_strike_rate: f64,
    comparison: WhiffComparison,
    verdict: String,
}

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
    let args = parse_args(&root)?;

    let mut starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    // `features::historical_rows` walks each pitcher's starts in slice order, so
    // the same sort the training path uses has to be applied first or "prior"
    // is not chronological.
    starts.sort_by_key(|start| (start.pitcher_id, start.game_date));

    let mut by_start: HashMap<(u32, NaiveDate), PitchOutcomes> = HashMap::new();
    let mut reader = csv::Reader::from_path(&args.outcomes)?;
    for record in reader.deserialize::<OutcomeRow>() {
        let row = record?;
        by_start.insert(
            (row.pitcher_id, row.game_date),
            PitchOutcomes {
                pitches: row.pitches,
                swinging_strikes: row.swinging_strikes,
                called_strikes: row.called_strikes,
            },
        );
    }
    let league_pitches: u64 = by_start.values().map(|o| u64::from(o.pitches)).sum();
    let league_whiffs: u64 = by_start
        .values()
        .map(|o| u64::from(o.swinging_strikes))
        .sum();
    eprintln!(
        "{} starts with pitch outcomes ({} pitches, league SwStr {:.4})",
        by_start.len(),
        league_pitches,
        league_whiffs as f64 / league_pitches.max(1) as f64
    );

    // Rolling whiff rates keyed the same way a FeatureRow identifies its start.
    let rolling = rolling_rates(&starts, &by_start);
    eprintln!("{} starts have a usable prior-window rate", rolling.len());

    let rows = features::historical_rows(&starts, HISTORY_WINDOW)?;
    eprintln!("{} feature rows built", rows.len());

    let (mut baseline_fit, mut baseline_eval) = (Vec::new(), Vec::new());
    let (mut augmented_fit, mut augmented_eval) = (Vec::new(), Vec::new());
    let mut dropped = 0usize;
    for row in &rows {
        if row.as_of > args.through {
            continue;
        }
        let Some((swstr, csw)) = rolling.get(&(row.pitcher_id, row.as_of)).copied() else {
            dropped += 1;
            continue;
        };
        let mut augmented = row.clone();
        augmented.values.push(swstr);
        augmented.values.push(csw);
        if row.as_of < args.split {
            baseline_fit.push(row.clone());
            augmented_fit.push(augmented);
        } else {
            baseline_eval.push(row.clone());
            augmented_eval.push(augmented);
        }
    }

    eprintln!(
        "fit {} rows before {}, evaluate {} rows {}..={} ({} rows dropped for thin history)",
        baseline_fit.len(),
        args.split,
        baseline_eval.len(),
        args.split,
        args.through,
        dropped
    );
    if baseline_eval.is_empty() {
        return Err("no evaluation rows; check --split and --through".into());
    }

    let names: Vec<String> = ADDED_FEATURES.iter().map(|name| name.to_string()).collect();
    let comparison = compare_k_rate(
        &baseline_fit,
        &baseline_eval,
        &augmented_fit,
        &augmented_eval,
        &names,
    )?;

    // The bar, applied mechanically rather than by eye.
    let separated = comparison
        .added_coefficients
        .iter()
        .any(|coefficient| coefficient.z > 2.0);
    let pre_registered = comparison.log_likelihood_gain > 0.0 && separated;
    // Stricter than pre-registered: the interval was added after the fact, and
    // only ever makes the bar harder to clear, never easier.
    let interval_excludes_zero = comparison
        .log_likelihood_gain_interval
        .as_ref()
        .and_then(|interval| interval.low)
        .is_some_and(|low| low > 0.0);
    let verdict = match (pre_registered, interval_excludes_zero) {
        (true, true) => {
            "CLEARS THE BAR: held-out likelihood improved, a coefficient is separated from zero, \
             and the gain's cluster-bootstrap interval excludes zero"
        }
        (true, false) => {
            "CLEARS THE PRE-REGISTERED BAR, but the gain's interval includes zero — the direction \
             is right and the size is not established"
        }
        (false, _) => "does not clear the pre-registered bar",
    };

    println!("\nk-rate held-out comparison ({} eval rows)", comparison.baseline.rows);
    println!(
        "{:<12} {:>16} {:>12} {:>12}",
        "", "mean log-lik", "MAE", "RMSE"
    );
    for (label, fit) in [
        ("baseline", &comparison.baseline),
        ("+ whiff", &comparison.augmented),
    ] {
        println!(
            "{label:<12} {:>16.5} {:>12.6} {:>12.6}",
            fit.mean_log_likelihood, fit.mae, fit.rmse
        );
    }
    println!(
        "{:<12} {:>+16.5} {:>+12.6}",
        "difference", comparison.log_likelihood_gain, comparison.mae_change
    );
    match comparison
        .log_likelihood_gain_interval
        .as_ref()
        .and_then(|interval| interval.low.zip(interval.high))
    {
        Some((low, high)) => println!(
            "  log-likelihood gain 95% CI (clustered by pitcher): [{low:+.5}, {high:+.5}]"
        ),
        None => println!("  log-likelihood gain interval: suppressed (too few pitchers)"),
    }
    println!("\nadded coefficients (standardized units)");
    println!("{:<38} {:>12} {:>12} {:>8}", "feature", "mean", "sd", "z");
    for coefficient in &comparison.added_coefficients {
        println!(
            "{:<38} {:>12.6} {:>12.6} {:>8.2}",
            coefficient.name, coefficient.mean, coefficient.sd, coefficient.z
        );
    }
    println!("\n{verdict}");

    let report = Report {
        dataset: args.dataset.display().to_string(),
        split: args.split,
        through: args.through,
        history_window: HISTORY_WINDOW,
        min_prior_pitches: MIN_PRIOR_PITCHES,
        rows_with_whiff_history: baseline_fit.len() + baseline_eval.len(),
        rows_dropped_for_thin_history: dropped,
        league_swinging_strike_rate: league_whiffs as f64 / league_pitches.max(1) as f64,
        comparison,
        verdict: verdict.to_string(),
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    println!("wrote {}", args.out.display());
    Ok(())
}

/// Rolling SwStr% and CSW% over the same prior window `historical_rows` uses.
///
/// Rates are built by summing whole starts and dividing once, never by averaging
/// per-start rates: a 100-pitch start and a 10-pitch start do not carry equal
/// evidence. Only starts strictly before the current one contribute, which is
/// the same leakage rule the rest of the feature pipeline follows.
fn rolling_rates(
    starts: &[PitcherStartLog],
    by_start: &HashMap<(u32, NaiveDate), PitchOutcomes>,
) -> HashMap<(u32, NaiveDate), (f64, f64)> {
    let mut groups: BTreeMap<u32, Vec<&PitcherStartLog>> = BTreeMap::new();
    for start in starts {
        groups.entry(start.pitcher_id).or_default().push(start);
    }
    let mut out = HashMap::new();
    for (id, group) in groups {
        // Mirrors `historical_rows`: rows begin at index 3, prior window is the
        // preceding `HISTORY_WINDOW` starts.
        for index in 3..group.len() {
            let from = index.saturating_sub(HISTORY_WINDOW);
            let mut window = PitchOutcomes::default();
            for prior in &group[from..index] {
                if let Some(outcomes) = by_start.get(&(prior.pitcher_id, prior.game_date)) {
                    window.merge(outcomes);
                }
            }
            if window.pitches < MIN_PRIOR_PITCHES {
                continue;
            }
            if let (Some(swstr), Some(csw)) = (
                window.swinging_strike_rate(),
                window.called_plus_swinging_rate(),
            ) {
                out.insert((id, group[index].game_date), (swstr, csw));
            }
        }
    }
    out
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dataset: root.join("data/processed/training_starts_v3.csv"),
        outcomes: root.join("data/processed/pitch_outcomes.csv"),
        split: NaiveDate::from_ymd_opt(2026, 5, 1).expect("valid default split"),
        through: NaiveDate::from_ymd_opt(2026, 7, 16).expect("valid default end"),
        out: root.join("outputs/whiff_signal.json"),
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
            "--outcomes" => args.outcomes = PathBuf::from(value()?),
            "--split" => args.split = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--through" => args.through = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}
