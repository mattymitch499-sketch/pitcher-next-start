//! Replays a historical window and grades the per-plate-appearance simulator
//! against the incumbent v2 model on the same starts.
//!
//! The question is narrow and answerable: given the same pitcher, the same
//! lineup and the same day, which of the two priced the strikeout ladder closer
//! to what actually happened? Both forecasters emit a PMF over exact strikeout
//! counts, so both are asked `P(K >= 2)` through `P(K >= 12)` on every start and
//! graded with Brier, log score and ECE — and the difference between them
//! carries a bootstrap interval that resamples *starts*, because the eleven
//! rungs of one start resolve as a block.
//!
//! Everything the replay depends on is dated before the window opens:
//!
//! - the model artifact is trained through the day before `--from`, so
//!   [`pns_core::model::artifact::ModelArtifact::score`]'s own leakage guard is
//!   what admits every row rather than being bypassed;
//! - the sim parameters were fitted with `--through` set to `--from`;
//! - the batter K-rate fold and the five-start feature window are resolved by
//!   [`pns_core::backtest::replay`], which has a regression test asserting that
//!   rows dated after a start cannot move that start's inputs.
//!
//! Nothing here touches the network: both the starts and the box-score lines
//! come off disk.
//!
//! Usage:
//! `cargo run --release -p pns-core --example backtest_sim -- [--model PATH] [--sim-params PATH] [--from YYYY-MM-DD] [--through YYYY-MM-DD] [--draws N] [--out PATH]`

use chrono::NaiveDate;
use pns_core::backtest::{self, SkipReason, StartForecast};
use pns_core::model::artifact::ModelArtifact;
use pns_core::sim::{DEFAULT_DRAWS, SimParams, expected_strikeouts, simulate_start};
use serde::Serialize;
use std::path::PathBuf;

struct Args {
    model: PathBuf,
    sim_params: PathBuf,
    dataset: PathBuf,
    lines: PathBuf,
    from: NaiveDate,
    through: NaiveDate,
    draws: usize,
    out: PathBuf,
}

/// Provenance, so a reader of the JSON can tell what produced the numbers
/// without re-deriving it from file names.
#[derive(Serialize)]
struct Provenance {
    model_artifact_id: String,
    model_trained_through: NaiveDate,
    model_training_rows: usize,
    sim_params_fit_through: Option<NaiveDate>,
    sim_params_model_artifact_id: String,
    sim_params_caveat: String,
    dataset: String,
    batter_lines: String,
    window_from: NaiveDate,
    window_through: NaiveDate,
    draws_per_start: usize,
    generated_at: String,
}

#[derive(Serialize)]
struct Population {
    starts_in_window: usize,
    starts_replayed: usize,
    starts_skipped: usize,
    skipped_short_history: usize,
    skipped_unknown_hand: usize,
    skipped_ambiguous_date: usize,
    /// Replayed starts whose opposing nine is recorded. A start without one
    /// falls back to league-average batters, which strips the simulator of the
    /// only signal the incumbent does not already have.
    starts_with_known_lineup: usize,
    mean_actual_strikeouts: f64,
    mean_model_expected_strikeouts: f64,
    mean_sim_expected_strikeouts: f64,
}

#[derive(Serialize)]
struct Report {
    provenance: Provenance,
    population: Population,
    evaluation: backtest::Evaluation,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let artifact = ModelArtifact::load(&args.model)?;
    let params = SimParams::load_path(&args.sim_params)?;
    let starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    let lines: Vec<_> = pns_core::data::batters::read_batter_lines(&args.lines)?
        .into_iter()
        .map(|row| row.line())
        .collect();
    eprintln!(
        "model {} through {}; sim params fitted through {:?}",
        artifact.artifact_id, artifact.training.data_through, params.fit.through
    );
    if artifact.training.data_through >= args.from {
        return Err(format!(
            "model artifact is trained through {}, which is inside the {} window; \
             train a historical artifact first",
            artifact.training.data_through, args.from
        )
        .into());
    }
    eprintln!(
        "{} starts and {} batter lines on disk",
        starts.len(),
        lines.len()
    );

    let replay = backtest::replay(&starts, &lines, args.from, args.through)?;
    let starts_in_window = starts
        .iter()
        .filter(|start| start.game_date >= args.from && start.game_date <= args.through)
        .count();
    eprintln!(
        "replaying {} of {} starts in window ({} skipped)",
        replay.cases.len(),
        starts_in_window,
        replay.skipped.len()
    );

    let mut forecasts = Vec::with_capacity(replay.cases.len());
    let mut model_expected = 0.0;
    let mut sim_expected = 0.0;
    let mut actual = 0.0;
    for (index, case) in replay.cases.iter().enumerate() {
        // The guarded path, not `artifact.model.score`: the artifact's own
        // cutoff check is the last line of defence against a leaky window, so
        // the backtest goes through it rather than around it.
        let projection = artifact.score(&case.row)?;
        let sim_pmf = simulate_start(
            &case.inputs(projection.k_rate_mean, projection.batters_faced_mean),
            &params,
            args.draws,
            case.seed(),
        );
        model_expected += projection.expected_strikeouts_compositional;
        sim_expected += expected_strikeouts(&sim_pmf);
        actual += f64::from(case.actual_strikeouts);
        forecasts.push(StartForecast {
            pitcher_id: case.pitcher_id,
            game_date: case.game_date,
            model_pmf: projection.strikeout_probabilities,
            sim_pmf,
            actual_strikeouts: case.actual_strikeouts,
        });
        if index % 250 == 249 {
            eprintln!("  scored {} / {}", index + 1, replay.cases.len());
        }
    }
    if forecasts.is_empty() {
        return Err("no start in the window could be replayed".into());
    }
    let count = forecasts.len() as f64;

    let evaluation = backtest::evaluate(&forecasts);
    let report = Report {
        provenance: Provenance {
            model_artifact_id: artifact.artifact_id.clone(),
            model_trained_through: artifact.training.data_through,
            model_training_rows: artifact.training.training_row_count,
            sim_params_fit_through: params.fit.through,
            sim_params_model_artifact_id: params.fit.model_artifact_id.clone(),
            sim_params_caveat: params.fit.caveat.clone(),
            dataset: args.dataset.display().to_string(),
            batter_lines: args.lines.display().to_string(),
            window_from: args.from,
            window_through: args.through,
            draws_per_start: args.draws,
            generated_at: chrono::Utc::now().to_rfc3339(),
        },
        population: Population {
            starts_in_window,
            starts_replayed: replay.cases.len(),
            starts_skipped: replay.skipped.len(),
            skipped_short_history: replay.skipped_by_reason(SkipReason::ShortHistory),
            skipped_unknown_hand: replay.skipped_by_reason(SkipReason::UnknownHand),
            skipped_ambiguous_date: replay.skipped_by_reason(SkipReason::AmbiguousDate),
            starts_with_known_lineup: replay.cases.iter().filter(|case| case.lineup_known).count(),
            mean_actual_strikeouts: actual / count,
            mean_model_expected_strikeouts: model_expected / count,
            mean_sim_expected_strikeouts: sim_expected / count,
        },
        evaluation,
    };

    print_report(&report);
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(
        &args.out,
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    println!("\nwrote {}", args.out.display());
    Ok(())
}

fn print_report(report: &Report) {
    let population = &report.population;
    let evaluation = &report.evaluation;
    println!(
        "\nbacktest {} .. {}  ({} draws per start)",
        report.provenance.window_from,
        report.provenance.window_through,
        report.provenance.draws_per_start
    );
    println!(
        "model {} trained through {} on {} rows",
        report.provenance.model_artifact_id,
        report.provenance.model_trained_through,
        report.provenance.model_training_rows
    );
    println!(
        "sim params fitted through {:?} against {}",
        report.provenance.sim_params_fit_through, report.provenance.sim_params_model_artifact_id
    );
    println!(
        "\n{} of {} starts replayed; {} skipped ({} short history, {} unknown hand, {} ambiguous date)",
        population.starts_replayed,
        population.starts_in_window,
        population.starts_skipped,
        population.skipped_short_history,
        population.skipped_unknown_hand,
        population.skipped_ambiguous_date
    );
    println!(
        "{} of {} replayed starts carry a recorded opposing lineup",
        population.starts_with_known_lineup, population.starts_replayed
    );
    println!(
        "mean K  actual {:.3}  model {:.3}  sim {:.3}",
        population.mean_actual_strikeouts,
        population.mean_model_expected_strikeouts,
        population.mean_sim_expected_strikeouts
    );

    println!("\nsurvival ladder: P(K >= t), and the Brier each forecaster paid for it");
    println!(
        "{:>4} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "t", "starts", "model", "sim", "observed", "mdl err", "sim err", "mdl Brier", "sim Brier"
    );
    for rung in &evaluation.rungs {
        println!(
            "{:>4} {:>7} {:>9.4} {:>9.4} {:>9.4} {:>+9.4} {:>+9.4} {:>9.4} {:>9.4}",
            rung.threshold,
            rung.starts,
            rung.model_probability,
            rung.sim_probability,
            rung.observed_rate,
            rung.model_probability - rung.observed_rate,
            rung.sim_probability - rung.observed_rate,
            rung.model_brier,
            rung.sim_brier,
        );
    }

    println!(
        "\npooled over {} rung-legs on {} starts",
        evaluation.legs, evaluation.starts
    );
    println!("{:<24} {:>12} {:>12}", "", "model", "sim");
    let line = |label: &str, model: f64, sim: f64| {
        println!("{label:<24} {model:>12.5} {sim:>12.5}");
    };
    line(
        "pooled Brier",
        evaluation.model.pooled_brier,
        evaluation.sim.pooled_brier,
    );
    line(
        "rung log loss",
        evaluation.model.rung_log_loss,
        evaluation.sim.rung_log_loss,
    );
    line("ECE (10 bins)", evaluation.model.ece, evaluation.sim.ece);
    line(
        "exact-count log score",
        evaluation.model.exact_log_score,
        evaluation.sim.exact_log_score,
    );

    if let Some(difference) = &evaluation.brier_difference {
        let interval = match (difference.low, difference.high) {
            (Some(low), Some(high)) => format!("[{low:+.5}, {high:+.5}]"),
            _ => "interval suppressed (too few starts)".into(),
        };
        println!(
            "\nsim minus model pooled Brier: {:+.5}  95% cluster-bootstrap CI {interval}",
            difference.point
        );
        println!("(negative favours the simulator; clustered by pitcher-start)");
    }
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        model: root.join("models/pns-model-hist-2026-04-30.json"),
        sim_params: root.join("models/pns-sim-params-hist-2026-04-30.json"),
        dataset: root.join("data/processed/training_starts_v3.csv"),
        lines: root.join("data/processed/batter_lines.csv"),
        from: NaiveDate::from_ymd_opt(2026, 5, 1).expect("valid default window start"),
        through: NaiveDate::from_ymd_opt(2026, 7, 16).expect("valid default window end"),
        draws: DEFAULT_DRAWS,
        out: root.join("outputs/backtest_sim_2026-04-30.json"),
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
            "--model" => args.model = PathBuf::from(value()?),
            "--sim-params" => args.sim_params = PathBuf::from(value()?),
            "--dataset" => args.dataset = PathBuf::from(value()?),
            "--lines" => args.lines = PathBuf::from(value()?),
            "--from" => args.from = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--through" => args.through = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--draws" => args.draws = value()?.parse()?,
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if args.draws == 0 {
        return Err("--draws must be positive".into());
    }
    Ok(args)
}
