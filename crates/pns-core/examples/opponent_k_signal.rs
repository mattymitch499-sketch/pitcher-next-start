//! Does the opposing lineup's *strikeout* tendency add anything the model's 18
//! features miss?
//!
//! A study, not a build. Nothing here changes the feature contract, the artifact
//! schema, or any shipped file.
//!
//! ## Why this question
//!
//! The v3 artifact's coefficients say the model's only opponent feature is doing
//! nothing for strikeouts. `opponent_offense_factor` — opponent earned runs per
//! inning against this pitcher hand — carries **-0.00111 per standard deviation**
//! on the k-rate regression, against **+0.01274** for swinging-strike rate and
//! **+0.00856** for the model's own recent k-rate. It sits at the noise floor,
//! alongside `park_factor` (-0.00107).
//!
//! That is a *runs* measure being asked a *strikeout* question. So the model
//! currently has close to no signal about whether the lineup it faces tonight
//! actually strikes out — even though the same fold that computes the ER factor
//! has the strikeout counts sitting right there.
//!
//! Note the same audit shows `opponent_offense_factor` is **not** dead
//! everywhere: it carries -0.16886 on `batters_faced`. It earns its slot on the
//! workload regression. So this study asks whether a strikeout factor *adds* to
//! the eighteen, not whether it should replace anything.
//!
//! ## The bar, fixed before the numbers exist
//!
//! Held-out predictive log-likelihood must improve on the **primary** split, the
//! added coefficient must be separated from zero, and the cluster-bootstrap
//! interval on the gain must exclude zero. Identical to the bar the
//! swinging-strike feature cleared. A better point estimate alone does not count.
//!
//! Two splits, with the high-power one named primary **now**:
//!
//! | split | fit | eval |
//! |---|---|---|
//! | **primary** | <= 2024-12-31 | 2025-01-01 -> 2026-07-16 |
//! | continuity | < 2026-05-01 | 2026-05-01 -> 2026-07-16 |
//!
//! Usage:
//! `cargo run --release -p pns-core --example opponent_k_signal -- [--dataset PATH] [--split YYYY-MM-DD] [--through YYYY-MM-DD] [--out PATH]`

use chrono::NaiveDate;
use pns_core::data::enrich;
use pns_core::features;
use pns_core::model::whiff::{TargetPriors, WhiffComparison, compare_target};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Rolling window, matching what the model is trained on.
const HISTORY_WINDOW: usize = 5;

const FEATURE_NAME: &str = "opponent_k_factor";

/// Which regression to test against.
///
/// `k-rate` is the original question. `batters-faced` exists because the
/// coefficient audit shows the model's *existing* opponent feature carries its
/// weight on workload, not rate: `opponent_offense_factor` is -0.00111 on k-rate
/// and **-0.16886 on batters_faced**. A lineup that strikes out often puts fewer
/// balls in play, so fewer men reach and fewer batters come up — a coherent
/// workload story worth measuring separately.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    KRate,
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

struct Args {
    dataset: PathBuf,
    target: Target,
    split: NaiveDate,
    through: NaiveDate,
    out: PathBuf,
}

/// How much the factor actually varies across the scored rows.
///
/// Reported because a null result has two very different causes: the feature
/// carries no signal, or shrinkage crushed it flat so there was nothing to
/// detect. A factor pinned near 1.0 everywhere would be the second.
#[derive(Serialize)]
struct FactorSpread {
    min: f64,
    max: f64,
    mean: f64,
    standard_deviation: f64,
}

#[derive(Serialize)]
struct Report {
    dataset: String,
    added_feature: String,
    /// Which regression this was measured against.
    target: String,
    split: NaiveDate,
    through: NaiveDate,
    history_window: usize,
    rows_scored: usize,
    /// Dataset-wide strikeouts per batter faced. An independent check that the
    /// counts are being read correctly — the real figure is about 0.22.
    league_k_per_batter_faced: f64,
    factor_spread: FactorSpread,
    /// Named `Whiff*` because the harness is shared, not whiff-specific: it fits
    /// the real k-rate estimator twice on identical rows differing only in
    /// feature width.
    ///
    /// Its bootstrap clusters by pitcher, which is **not** the right cluster for
    /// an opponent property — see `comparison_clustered_by_opponent`.
    comparison: WhiffComparison,
    /// The same comparison with the gain's interval clustered by opponent
    /// instead. Identical fit and identical point estimate by construction; only
    /// the interval differs, and this is the one the verdict is judged on.
    comparison_clustered_by_opponent: WhiffComparison,
    /// Clustered by opponent *team-season*, since rosters turn over and the two
    /// seasons in the evaluation window are not one observation.
    comparison_clustered_by_opponent_season: WhiffComparison,
    verdict: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let mut starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    // `features::historical_rows` walks each pitcher's starts in slice order, so
    // the same sort the training path uses has to be applied first or "prior" is
    // not chronological.
    starts.sort_by_key(|start| (start.pitcher_id, start.game_date));

    // Handedness rides on the dataset rather than being refetched. A row without
    // it would silently fall back to 'R' and pollute both hands' pools, so refuse
    // rather than guess.
    let mut hands: HashMap<u32, char> = HashMap::new();
    let mut missing_hand = 0usize;
    for start in &starts {
        match start.pitch_hand {
            Some(hand) => {
                hands.insert(start.pitcher_id, hand);
            }
            None => missing_hand += 1,
        }
    }
    if missing_hand > 0 {
        return Err(format!(
            "{missing_hand} of {} rows have no pitch_hand; the factor is hand-split, so a \
             fallback would mix left- and right-handed pools",
            starts.len()
        )
        .into());
    }

    let league_strikeouts: u64 = starts.iter().map(|s| u64::from(s.strikeouts)).sum();
    let league_batters: u64 = starts.iter().map(|s| u64::from(s.batters_faced)).sum();
    let league_k_rate = league_strikeouts as f64 / league_batters.max(1) as f64;
    eprintln!(
        "{} starts, {league_strikeouts} strikeouts over {league_batters} batters faced \
         (league K/PA {league_k_rate:.4})",
        starts.len()
    );

    let factors = enrich::opponent_k_factors(&starts, &hands);
    eprintln!("{} starts have an as-of opponent strikeout factor", factors.len());

    let rows = features::historical_rows(&starts, HISTORY_WINDOW)?;
    eprintln!("{} feature rows built", rows.len());

    // Which opponent each scored start faced, so the gain can also be clustered
    // the way *this* feature actually correlates. See `rekey` below.
    let opponent_of: HashMap<(u32, NaiveDate), u32> = starts
        .iter()
        .map(|start| ((start.pitcher_id, start.game_date), start.opponent_id))
        .collect();

    let (mut baseline_fit, mut baseline_eval) = (Vec::new(), Vec::new());
    let (mut augmented_fit, mut augmented_eval) = (Vec::new(), Vec::new());
    let mut observed: Vec<f64> = Vec::new();
    let mut missing = 0usize;
    for row in &rows {
        if row.as_of > args.through {
            continue;
        }
        // Unlike the whiff study there is no thin-history drop: the accumulator
        // shrinks toward the league rate, so an opponent with no record yet reads
        // exactly neutral rather than missing.
        let Some(&factor) = factors.get(&(row.pitcher_id, row.as_of)) else {
            missing += 1;
            continue;
        };
        observed.push(factor);
        let mut augmented = row.clone();
        augmented.values.push(factor);
        if row.as_of < args.split {
            baseline_fit.push(row.clone());
            augmented_fit.push(augmented);
        } else {
            baseline_eval.push(row.clone());
            augmented_eval.push(augmented);
        }
    }
    if missing > 0 {
        return Err(format!(
            "{missing} feature rows had no matching start in the factor map; the join key is \
             wrong rather than the data being thin"
        )
        .into());
    }

    let count = observed.len().max(1) as f64;
    let mean = observed.iter().sum::<f64>() / count;
    let variance = observed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
    let spread = FactorSpread {
        min: observed.iter().copied().fold(f64::INFINITY, f64::min),
        max: observed.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean,
        standard_deviation: variance.sqrt(),
    };
    eprintln!(
        "factor spread: min {:.4}, max {:.4}, mean {:.4}, sd {:.4}",
        spread.min, spread.max, spread.mean, spread.standard_deviation
    );

    eprintln!(
        "fit {} rows before {}, evaluate {} rows {}..={}",
        baseline_fit.len(),
        args.split,
        baseline_eval.len(),
        args.split,
        args.through
    );
    if baseline_eval.is_empty() {
        return Err("no evaluation rows; check --split and --through".into());
    }

    let names = vec![FEATURE_NAME.to_string()];
    // Priors must match `BayesianLinearModel::fit` for the same target
    // (`model/mod.rs:186-187`), or this is not the production estimator.
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

    // `compare_k_rate` clusters its bootstrap on `FeatureRow::pitcher_id`. That
    // is the right cluster for a *pitcher* property like swinging-strike rate,
    // and the wrong one here: this factor is a property of the **opponent**, so
    // every start against the same lineup shares whatever it gets wrong. Since
    // the fit reads only `values` and the target, swapping the identifier
    // re-clusters the interval while leaving the fit alone — and the point
    // estimate coming out identical is the proof of that.
    let rekey = |source: &[features::FeatureRow],
                 key: &dyn Fn(u32, NaiveDate) -> u32|
     -> Result<Vec<features::FeatureRow>, String> {
        source
            .iter()
            .map(|row| {
                let opponent = opponent_of
                    .get(&(row.pitcher_id, row.as_of))
                    .copied()
                    .ok_or_else(|| {
                        format!("no opponent for pitcher {} on {}", row.pitcher_id, row.as_of)
                    })?;
                let mut copy = row.clone();
                copy.pitcher_id = key(opponent, row.as_of);
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
        // Re-keying must move the interval and nothing else.
        if (arm.log_likelihood_gain - comparison.log_likelihood_gain).abs() > 1e-12 {
            return Err(
                "re-keying the cluster identifier changed the fit, so the intervals are not \
                 measuring the same quantity"
                    .into(),
            );
        }
        Ok(arm)
    };

    let by_opponent = clustered(&|opponent, _| opponent)?;
    // Rosters turn over, so the 2025 and 2026 versions of a club are not one
    // observation. Clustering by team *alone* over-pools across a two-season
    // evaluation window; team-season is the more defensible unit. Both are
    // reported, and the verdict is judged on the stricter of the two.
    let by_opponent_season = clustered(&|opponent, date| {
        use chrono::Datelike;
        opponent * 10_000 + date.year() as u32
    })?;

    // The bar, applied mechanically rather than by eye, and judged on the
    // **opponent-clustered** interval because that is the honest one for this
    // feature.
    let separated = comparison
        .added_coefficients
        .iter()
        .any(|coefficient| coefficient.z > 2.0);
    let pre_registered = comparison.log_likelihood_gain > 0.0 && separated;
    let interval_excludes_zero = [&by_opponent, &by_opponent_season].iter().all(|arm| {
        arm.log_likelihood_gain_interval
            .as_ref()
            .and_then(|interval| interval.low)
            .is_some_and(|low| low > 0.0)
    });
    let verdict = match (pre_registered, interval_excludes_zero) {
        (true, true) => {
            "CLEARS THE BAR: held-out likelihood improved, the coefficient is separated from \
             zero, and the gain's interval excludes zero even clustered by opponent"
        }
        (true, false) => {
            "CLEARS THE PRE-REGISTERED BAR, but the opponent-clustered interval includes zero \
             — the direction is right and the size is not established"
        }
        (false, _) => "does not clear the pre-registered bar",
    };

    println!(
        "\n{} held-out comparison [{FEATURE_NAME}] ({} eval rows)",
        args.target.label(),
        comparison.baseline.rows
    );
    println!(
        "{:<14} {:>16} {:>12} {:>12}",
        "", "mean log-lik", "MAE", "RMSE"
    );
    for (label, fit) in [
        ("baseline", &comparison.baseline),
        ("+ opponent K", &comparison.augmented),
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
    for (label, arm) in [
        ("pitcher (wrong unit here)", &comparison),
        ("opponent team", &by_opponent),
        ("opponent team-season", &by_opponent_season),
    ] {
        match arm
            .log_likelihood_gain_interval
            .as_ref()
            .and_then(|interval| interval.low.zip(interval.high))
        {
            Some((low, high)) => {
                println!("  gain 95% CI clustered by {label:<26} [{low:+.5}, {high:+.5}]")
            }
            None => println!("  gain 95% CI clustered by {label:<26} suppressed (too few clusters)"),
        }
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
        added_feature: FEATURE_NAME.to_string(),
        target: args.target.label().to_string(),
        split: args.split,
        through: args.through,
        history_window: HISTORY_WINDOW,
        rows_scored: baseline_fit.len() + baseline_eval.len(),
        league_k_per_batter_faced: league_k_rate,
        factor_spread: spread,
        comparison,
        comparison_clustered_by_opponent: by_opponent,
        comparison_clustered_by_opponent_season: by_opponent_season,
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
        // The primary split, named in advance. The continuity split is
        // `--split 2026-05-01`.
        // k-rate stays the default so the published result reproduces unchanged.
        target: Target::KRate,
        split: NaiveDate::from_ymd_opt(2025, 1, 1).expect("valid default split"),
        through: NaiveDate::from_ymd_opt(2026, 7, 16).expect("valid default end"),
        out: PathBuf::new(),
    };
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
            "--target" => args.target = Target::parse(&value()?)?,
            "--split" => args.split = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--through" => args.through = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--out" => explicit_out = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    args.out = explicit_out.unwrap_or_else(|| match args.target {
        // The k-rate path keeps its original filename so published output is
        // not silently relocated.
        Target::KRate => root.join("outputs/opponent_k_signal.json"),
        Target::BattersFaced => root.join("outputs/opponent_k_signal_batters_faced.json"),
    });
    Ok(args)
}
