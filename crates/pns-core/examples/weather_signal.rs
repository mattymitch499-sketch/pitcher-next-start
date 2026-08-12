//! Does game-time weather add anything the model's 18 features miss?
//!
//! A study, not a build. Nothing here changes the feature contract, the artifact
//! schema, or any shipped file.
//!
//! ## Why this question
//!
//! The model has never seen weather in any form. Temperature and wind are the
//! two candidates with a physical story: warm thin air carries, wind off the bat
//! helps or suppresses, and both change how long a pitcher lasts.
//!
//! Unlike the rolling whiff rates, these are properties of *tonight's game*
//! rather than of the prior window — the same shape as `home` and `park_factor`.
//! That is legitimate as a feature (conditions are known at first pitch) but it
//! is exactly why shipping it would need a **forecast** source: MLB publishes
//! observed weather only, and a future date returns nothing.
//!
//! ## Clustering, fixed in advance because the default is wrong here
//!
//! `compare_k_rate` bootstraps clustered by **pitcher**. That is right for a
//! pitcher property like swinging-strike rate and wrong for weather, which is a
//! property of the **park and the date** — every start in the same venue shares
//! whatever the feature gets wrong about that venue's climate and roof.
//!
//! The opponent-strikeout study measured that error at **3.4x**: its
//! pitcher-clustered interval excluded zero, and both correct units did not. So
//! this study reports park, park-season and date clusterings from the start and
//! is judged on the strictest, with the pitcher-clustered figure kept only as
//! the reference showing how much it flatters.
//!
//! ## The bar, fixed before the numbers exist
//!
//! Held-out predictive log-likelihood must improve on the **primary** split
//! (fit <= 2024, evaluate 2025-01-01 -> 2026-07-16), a coefficient must be
//! separated from zero, and the gain's interval must exclude zero under **every**
//! park-based clustering. Identical in substance to the bar swinging-strike rate
//! cleared, with the cluster unit named rather than left to the default.
//!
//! Requires `data/processed/game_weather.csv` from `fetch_weather`.
//!
//! Usage:
//! `cargo run --release -p pns-core --example weather_signal -- [--dataset PATH] [--weather PATH] [--features temp|temp-wind] [--split YYYY-MM-DD] [--through YYYY-MM-DD] [--out PATH]`

use chrono::{Datelike, NaiveDate};
use pns_core::data::weather::{GameWeather, WindDirection};
use pns_core::features;
use pns_core::model::whiff::{TargetPriors, WhiffComparison, compare_target};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

const HISTORY_WINDOW: usize = 5;

/// Which regression to test the feature against.
///
/// **This choice is the whole point.** The first pass ran against `k-rate` and
/// found nothing, which was the wrong question: a strikeout is by definition a
/// ball *not* put in play, and wind acts on balls that *were* hit. The
/// coefficient audit says so too — `park_factor`, the same kind of
/// run-environment variable, is -0.00107 on k-rate and **+0.16208 on
/// batters_faced**. Total strikeouts are `k_rate x batters_faced`, so the second
/// factor is where a weather effect would live.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    /// Strikeouts per batter faced. Kept runnable so the published null result
    /// stays reproducible.
    KRate,
    /// Batters faced — the workload channel.
    BattersFaced,
}

impl Target {
    fn parse(raw: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match raw {
            "k-rate" => Ok(Self::KRate),
            "batters-faced" => Ok(Self::BattersFaced),
            other => {
                Err(format!("--target must be k-rate or batters-faced, got {other}").into())
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::KRate => "k_rate",
            Self::BattersFaced => "batters_faced",
        }
    }
}

/// Which weather parameterisation to test. Two, both named in the plan before
/// any number existed, so this is not a search over encodings.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Parameterisation {
    /// Temperature alone.
    Temp,
    /// Temperature plus the signed wind component along the in/out axis.
    TempWind,
}

impl Parameterisation {
    fn parse(raw: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match raw {
            "temp" => Ok(Self::Temp),
            "temp-wind" => Ok(Self::TempWind),
            other => Err(format!("--features must be temp or temp-wind, got {other}").into()),
        }
    }

    fn names(self) -> Vec<String> {
        match self {
            Self::Temp => vec!["game_temperature_f".into()],
            Self::TempWind => vec![
                "game_temperature_f".into(),
                "game_wind_out_component_mph".into(),
            ],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Temp => "temp",
            Self::TempWind => "temp-wind",
        }
    }

    fn values(self, weather: &GameWeather) -> Option<Vec<f64>> {
        let temperature = weather.temperature_f?;
        Some(match self {
            Self::Temp => vec![temperature],
            Self::TempWind => vec![temperature, wind_out_component(weather)],
        })
    }
}

/// Wind resolved onto the single axis that has a physical story: positive blowing
/// out, negative blowing in, zero across or calm.
///
/// Collapsing to one signed number rather than dummying four directions keeps
/// the added width at one column and states the hypothesis directly — a
/// crosswind is not half an out-wind, it is no out-wind.
fn wind_out_component(weather: &GameWeather) -> f64 {
    let speed = weather.wind_speed_mph.unwrap_or(0.0);
    match weather.wind_direction {
        WindDirection::Out => speed,
        WindDirection::In => -speed,
        WindDirection::Cross | WindDirection::Calm | WindDirection::Unknown => 0.0,
    }
}

struct Args {
    dataset: PathBuf,
    weather: PathBuf,
    features: Parameterisation,
    target: Target,
    split: NaiveDate,
    through: NaiveDate,
    out: PathBuf,
}

#[derive(Serialize)]
struct ClusterInterval {
    unit: String,
    clusters: usize,
    low: Option<f64>,
    high: Option<f64>,
}

#[derive(Serialize)]
struct Report {
    dataset: String,
    weather_source: String,
    /// Which regression this was measured against. The k-rate null and the
    /// workload result are different questions and must never be read as one.
    target: String,
    parameterisation: String,
    split: NaiveDate,
    through: NaiveDate,
    history_window: usize,
    rows_scored: usize,
    rows_dropped_without_weather: usize,
    indoor_share: f64,
    mean_temperature_f: f64,
    /// Pitcher-clustered, kept only as the reference that shows how much the
    /// default flatters a non-pitcher feature.
    comparison: WhiffComparison,
    intervals: Vec<ClusterInterval>,
    verdict: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let mut starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    starts.sort_by_key(|start| (start.pitcher_id, start.game_date));

    let mut by_game: HashMap<u32, GameWeather> = HashMap::new();
    let mut reader = csv::Reader::from_path(&args.weather)?;
    for record in reader.deserialize::<GameWeather>() {
        let row = record?;
        by_game.insert(row.game_pk, row);
    }
    eprintln!("{} games with weather", by_game.len());

    // Per-start lookups: the extra feature values, and the park that hosted it.
    let mut extras: HashMap<(u32, NaiveDate), Vec<f64>> = HashMap::new();
    let mut park_of: HashMap<(u32, NaiveDate), u32> = HashMap::new();
    let mut indoor = 0usize;
    let mut temperatures: Vec<f64> = Vec::new();
    for start in &starts {
        let home_team_id = if start.home {
            start.team_id
        } else {
            start.opponent_id
        };
        park_of.insert((start.pitcher_id, start.game_date), home_team_id);
        let Some(weather) = by_game.get(&start.game_pk) else {
            continue;
        };
        if weather.indoor {
            indoor += 1;
        }
        if let Some(temperature) = weather.temperature_f {
            temperatures.push(temperature);
        }
        if let Some(values) = args.features.values(weather) {
            extras.insert((start.pitcher_id, start.game_date), values);
        }
    }
    let indoor_share = indoor as f64 / starts.len().max(1) as f64;
    let mean_temperature =
        temperatures.iter().sum::<f64>() / temperatures.len().max(1) as f64;
    eprintln!(
        "{} starts have weather features; indoor {:.1}%, mean temperature {:.1}F",
        extras.len(),
        indoor_share * 100.0,
        mean_temperature
    );

    let rows = features::historical_rows(&starts, HISTORY_WINDOW)?;
    eprintln!("{} feature rows built", rows.len());

    let (mut baseline_fit, mut baseline_eval) = (Vec::new(), Vec::new());
    let (mut augmented_fit, mut augmented_eval) = (Vec::new(), Vec::new());
    let mut dropped = 0usize;
    let mut scored = 0usize;
    for row in &rows {
        if row.as_of > args.through {
            continue;
        }
        scored += 1;
        let Some(extra) = extras.get(&(row.pitcher_id, row.as_of)) else {
            dropped += 1;
            continue;
        };
        let mut augmented = row.clone();
        augmented.values.extend(extra.iter().copied());
        if row.as_of < args.split {
            baseline_fit.push(row.clone());
            augmented_fit.push(augmented);
        } else {
            baseline_eval.push(row.clone());
            augmented_eval.push(augmented);
        }
    }
    // The backfill measured 100% coverage, so a material drop here means the
    // join broke rather than the data being thin.
    if scored > 0 && (dropped as f64 / scored as f64) > 0.01 {
        return Err(format!(
            "{dropped} of {scored} rows had no weather; the game_pk join is broken"
        )
        .into());
    }
    eprintln!(
        "fit {} rows before {}, evaluate {} rows {}..={} ({dropped} dropped)",
        baseline_fit.len(),
        args.split,
        baseline_eval.len(),
        args.split,
        args.through
    );
    if baseline_eval.is_empty() {
        return Err("no evaluation rows; check --split and --through".into());
    }

    let names = args.features.names();
    // Priors must match what `BayesianLinearModel::fit` uses for the same target
    // (`model/mod.rs:186-187`) or the study is not measuring the production
    // estimator.
    let compare = |baseline_fit: &[features::FeatureRow],
                   baseline_eval: &[features::FeatureRow],
                   augmented_fit: &[features::FeatureRow],
                   augmented_eval: &[features::FeatureRow]|
     -> Result<WhiffComparison, Box<dyn std::error::Error>> {
        Ok(match args.target {
            Target::KRate => compare_target(
                baseline_fit,
                baseline_eval,
                augmented_fit,
                augmented_eval,
                &names,
                |row: &features::FeatureRow| row.target_k_rate,
                TargetPriors::K_RATE,
            )?,
            Target::BattersFaced => compare_target(
                baseline_fit,
                baseline_eval,
                augmented_fit,
                augmented_eval,
                &names,
                |row: &features::FeatureRow| row.target_batters_faced,
                TargetPriors::BATTERS_FACED,
            )?,
        })
    };
    let comparison = compare(
        &baseline_fit,
        &baseline_eval,
        &augmented_fit,
        &augmented_eval,
    )?;

    // Re-key the cluster identifier without touching the fit. `compare_k_rate`
    // reads only `values` and the target, so the point estimate must come out
    // identical — that identity is the proof the swap was inert.
    let rekey = |source: &[features::FeatureRow],
                 key: &dyn Fn(u32, NaiveDate) -> u32|
     -> Result<Vec<features::FeatureRow>, String> {
        source
            .iter()
            .map(|row| {
                let park = park_of
                    .get(&(row.pitcher_id, row.as_of))
                    .copied()
                    .ok_or_else(|| {
                        format!("no park for pitcher {} on {}", row.pitcher_id, row.as_of)
                    })?;
                let mut copy = row.clone();
                copy.pitcher_id = key(park, row.as_of);
                Ok(copy)
            })
            .collect()
    };
    let clustered = |key: &dyn Fn(u32, NaiveDate) -> u32| -> Result<WhiffComparison, Box<dyn std::error::Error>> {
        let arm = compare(
            &rekey(&baseline_fit, key)?,
            &rekey(&baseline_eval, key)?,
            &rekey(&augmented_fit, key)?,
            &rekey(&augmented_eval, key)?,
        )?;
        if (arm.log_likelihood_gain - comparison.log_likelihood_gain).abs() > 1e-12 {
            return Err(
                "re-keying the cluster identifier changed the fit, so the intervals are not \
                 measuring the same quantity"
                    .into(),
            );
        }
        Ok(arm)
    };

    let by_park = clustered(&|park, _| park)?;
    let by_park_season = clustered(&|park, date| park * 10_000 + date.year() as u32)?;
    let by_date = clustered(&|_, date| {
        date.year() as u32 * 1_000 + date.ordinal()
    })?;

    let distinct = |key: &dyn Fn(u32, NaiveDate) -> u32| -> usize {
        baseline_eval
            .iter()
            .filter_map(|row| {
                park_of
                    .get(&(row.pitcher_id, row.as_of))
                    .map(|&park| key(park, row.as_of))
            })
            .collect::<std::collections::BTreeSet<u32>>()
            .len()
    };

    let arms: Vec<(&str, usize, &WhiffComparison)> = vec![
        (
            "pitcher (reference, wrong unit)",
            baseline_eval
                .iter()
                .map(|row| row.pitcher_id)
                .collect::<std::collections::BTreeSet<u32>>()
                .len(),
            &comparison,
        ),
        ("park", distinct(&|park, _| park), &by_park),
        (
            "park-season",
            distinct(&|park, date| park * 10_000 + date.year() as u32),
            &by_park_season,
        ),
        (
            "date",
            distinct(&|_, date| date.year() as u32 * 1_000 + date.ordinal()),
            &by_date,
        ),
    ];

    // The bar: judged on every park-based clustering, never on the pitcher one.
    let separated = comparison
        .added_coefficients
        .iter()
        .any(|coefficient| coefficient.z > 2.0);
    let pre_registered = comparison.log_likelihood_gain > 0.0 && separated;
    let interval_excludes_zero = [&by_park, &by_park_season, &by_date].iter().all(|arm| {
        arm.log_likelihood_gain_interval
            .as_ref()
            .and_then(|interval| interval.low)
            .is_some_and(|low| low > 0.0)
    });
    let verdict = match (pre_registered, interval_excludes_zero) {
        (true, true) => {
            "CLEARS THE BAR: held-out likelihood improved, a coefficient is separated from zero, \
             and the gain's interval excludes zero under every park-based clustering"
        }
        (true, false) => {
            "CLEARS THE PRE-REGISTERED BAR, but a correctly clustered interval includes zero — \
             the direction is right and the size is not established"
        }
        (false, _) => "does not clear the pre-registered bar",
    };

    println!(
        "\n{} held-out comparison [{}] ({} eval rows)",
        args.target.label(),
        args.features.label(),
        comparison.baseline.rows
    );
    println!(
        "{:<14} {:>16} {:>12} {:>12}",
        "", "mean log-lik", "MAE", "RMSE"
    );
    for (label, fit) in [
        ("baseline", &comparison.baseline),
        ("+ weather", &comparison.augmented),
    ] {
        println!(
            "{label:<14} {:>16.5} {:>12.6} {:>12.6}",
            fit.mean_log_likelihood, fit.mae, fit.rmse
        );
    }
    println!(
        "{:<14} {:>+16.5} {:>+12.6}",
        "difference", comparison.log_likelihood_gain, comparison.mae_change
    );
    let mut intervals = Vec::new();
    for (unit, clusters, arm) in &arms {
        let bounds = arm
            .log_likelihood_gain_interval
            .as_ref()
            .and_then(|interval| interval.low.zip(interval.high));
        match bounds {
            Some((low, high)) => println!(
                "  gain 95% CI by {unit:<32} ({clusters:>4} clusters) [{low:+.5}, {high:+.5}]"
            ),
            None => println!(
                "  gain 95% CI by {unit:<32} ({clusters:>4} clusters) suppressed"
            ),
        }
        intervals.push(ClusterInterval {
            unit: (*unit).to_string(),
            clusters: *clusters,
            low: bounds.map(|(low, _)| low),
            high: bounds.map(|(_, high)| high),
        });
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
        weather_source: args.weather.display().to_string(),
        target: args.target.label().to_string(),
        parameterisation: args.features.label().to_string(),
        split: args.split,
        through: args.through,
        history_window: HISTORY_WINDOW,
        rows_scored: baseline_fit.len() + baseline_eval.len(),
        rows_dropped_without_weather: dropped,
        indoor_share,
        mean_temperature_f: mean_temperature,
        comparison,
        intervals,
        verdict: verdict.to_string(),
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    println!("wrote {}", args.out.display());
    Ok(())
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        dataset: root.join("data/processed/training_starts_v4.csv"),
        weather: root.join("data/processed/game_weather.csv"),
        features: Parameterisation::Temp,
        // Workload is the open question; k-rate has already been answered.
        target: Target::BattersFaced,
        split: NaiveDate::from_ymd_opt(2025, 1, 1).expect("valid default split"),
        through: NaiveDate::from_ymd_opt(2026, 7, 16).expect("valid default end"),
        out: PathBuf::new(),
    };
    // Resolved after parsing so the default output path carries the target, and
    // a workload run cannot silently overwrite the published k-rate result.
    let mut explicit_out: Option<PathBuf> = None;
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
            "--weather" => args.weather = PathBuf::from(value()?),
            "--features" => args.features = Parameterisation::parse(&value()?)?,
            "--target" => args.target = Target::parse(&value()?)?,
            "--split" => args.split = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--through" => args.through = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--out" => explicit_out = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    args.out = explicit_out.unwrap_or_else(|| {
        root.join(format!(
            "outputs/weather_signal_{}_{}.json",
            args.target.label(),
            args.features.label()
        ))
    });
    Ok(args)
}
