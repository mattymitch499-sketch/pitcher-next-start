//! Fits the removal hazard behind the per-plate-appearance simulator, and writes
//! `models/pns-sim-params-v1.json`.
//!
//! The simulator's strikeout side is fixed by the v2 model and the batter rate
//! fold; the only free part is *when the starter gets pulled*. So the fit asks
//! exactly one question: replaying every historical start, do the simulated
//! batters-faced quantiles match the observed ones, bucket by bucket of what the
//! batters-faced regression predicted? Four hazard parameters are searched by
//! coordinate descent over a shrinking grid — no optimizer, no dependency, and
//! common random numbers across candidates so the search follows the loss
//! surface instead of Monte Carlo noise.
//!
//! Leakage discipline, in two parts:
//!
//! - The batter rate fold absorbs a date's box-score lines only *after* every
//!   start on that date has been served, the same as
//!   [`pns_core::data::enrich::accumulate_historical_context`]'s date-block fold.
//! - `--through YYYY-MM-DD` fits on starts played strictly before that date, so a
//!   backtest can refit with a boundary that predates its test window.
//!
//! One caveat applies to the shipped fit and is written into the artifact's
//! `caveat` field: `models/pns-model-v2.json` was trained through 2026-07-16, so
//! its `k_rate_mean` / `batters_faced_mean` predictions on earlier starts are
//! in-sample. The hazard is therefore fitted against slightly-too-good workload
//! predictions. That is acceptable for live pricing because the hazard is a
//! *conditional* calibration (given the prediction, how does the pitcher's
//! actual workload spread?). A backtest must not accept it, so `--model` points
//! the fit at a historical artifact whose training cutoff also predates the test
//! window; `--model` plus `--through` closes the caveat completely, and the
//! `caveat` field then says so.
//!
//! **Which objective to minimise is an open question, and the default is the
//! conservative answer.** `--objective workload` (default) matches simulated
//! batters-faced quantiles to observed ones; `--objective forecast` scores the
//! pooled survival-ladder Brier the backtest reports, on the fit window. Fitted
//! properly (`--restarts 8`) the two disagree sharply — the workload optimum
//! reaches a quantile loss of 0.80 against the forecast optimum's 5.00, while
//! the forecast optimum moves `traffic_slope` in the opposite direction. Neither
//! moved out-of-sample Brier by more than 0.0002 (measured 2026-08-07:
//! shipped 0.11333, workload 0.11338, forecast 0.11325), which is the useful
//! finding: **the hazard is not where the simulator's remaining error lives.**
//! Both settings are kept so that conclusion can be re-tested rather than
//! taken on trust.
//!
//! Usage:
//! `cargo run --release -p pns-core --example fit_sim_params -- [--model PATH] [--through YYYY-MM-DD] [--draws N] [--seed N] [--restarts N] [--objective workload|forecast] [--out PATH]`

use chrono::NaiveDate;
use pns_core::backtest::RUNGS;
use pns_core::data::batters::BatterKRates;
use pns_core::model::artifact::ModelArtifact;
use pns_core::sim::{
    BucketFit, DEFAULT_DRAWS, FitMetadata, MAX_BATTERS_FACED, SIM_PARAMS_SCHEMA_VERSION, SimParams,
    StartInputs, expected_strikeouts, simulate_draws, simulate_start,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// History window the shipped model was trained with.
const HISTORY_WINDOW: usize = 5;

/// Predicted-workload buckets. Five is enough to expose a slope in the fit
/// without leaving any bucket too thin to read a p10 off.
const BUCKETS: usize = 5;

/// Starts the coordinate search simulates. The full set is used for the report;
/// the search runs on a deterministic stride through it so a sweep costs seconds
/// rather than minutes, and the stride keeps every bucket and every season in.
const SEARCH_STARTS: usize = 5_000;

/// Quantiles the loss matches. p10/p90 pin the tails the hook creates, the
/// quartiles and median pin the body.
const QUANTILE_COUNT: usize = 5;
const QUANTILES: [f64; QUANTILE_COUNT] = [0.10, 0.25, 0.50, 0.75, 0.90];

/// Coordinate-descent axes: (name, low, high). Rounds halve the span each pass.
const AXES: [(&str, f64, f64); 4] = [
    ("intercept", -3.0, 1.0),
    ("pa_slope", 0.15, 1.20),
    ("traffic_slope", 0.0, 1.50),
    ("pivot_scale", 0.60, 1.40),
];
const ROUNDS: usize = 3;
const GRID_POINTS: usize = 7;

/// Deterministic restarts, on top of the one that begins at the defaults.
///
/// Coordinate descent only accepts a move that improves the loss on its own
/// axis, so it halts wherever no single axis helps — even when a joint move
/// would. That is not hypothetical here: the shipped historical fit
/// (`models/pns-sim-params-hist-2026-04-30.json`) left `intercept`,
/// `traffic_slope` and `pivot_scale` at *exactly* their defaults, moving only
/// `pa_slope`. Restarting from a spread of points is the cheapest escape that
/// keeps the search fully deterministic.
///
/// **Default 0 on purpose.** Multi-start works — it found a workload loss of
/// 0.80 where the single start stalled at 1.80 — but those better-fitting
/// parameters *forecast worse* out of sample (backtest Brier 0.11338 against
/// the shipped 0.11333, measured 2026-08-07). Until the objective question is
/// settled, the default has to reproduce the artifact that is actually shipped,
/// or regenerating params silently downgrades them. Pass `--restarts 8` to
/// search properly.
const DEFAULT_RESTARTS: usize = 0;

/// Halton bases, one per axis. Coprime bases stop the four coordinates from
/// advancing in lockstep the way a shared base (or a uniform grid) would.
const HALTON_BASES: [usize; 4] = [2, 3, 5, 7];

/// Radical inverse of `index` in `base` — a low-discrepancy point in (0, 1).
///
/// Restart positions have to be reproducible run to run, and every other
/// randomness source in this project is hand-rolled for exactly that reason
/// (`sim_seed`, the cluster bootstrap). A Halton sequence spreads the restarts
/// across the box with no RNG and no hand-picked table to drift out of date.
fn halton(mut index: usize, base: usize) -> f64 {
    let (mut fraction, mut result) = (1.0, 0.0);
    while index > 0 {
        fraction /= base as f64;
        result += fraction * (index % base) as f64;
        index /= base;
    }
    result
}

/// Where restart `index` begins. Restart 0 is the defaults, so the search
/// strictly contains the single-start version it replaces: rounds share a
/// schedule that depends only on the round index, so restart 0 replays the old
/// trajectory move for move and the best-of pass can only improve on it.
fn restart_hazard(index: usize) -> pns_core::sim::RemovalHazard {
    let mut hazard = SimParams::defaults().hazard;
    if index > 0 {
        for (axis, (_, low, high)) in AXES.iter().enumerate() {
            set_axis(
                &mut hazard,
                axis,
                low + (high - low) * halton(index, HALTON_BASES[axis]),
            );
        }
    }
    hazard
}

/// Starts sampled for the simulator-versus-model strikeout comparison.
const COMPARISON_STARTS: usize = 1_000;

struct Args {
    through: Option<NaiveDate>,
    draws: usize,
    seed: u64,
    model: PathBuf,
    out: PathBuf,
    /// Restarts beyond the one that begins at the defaults. `0` reproduces the
    /// single-start search this replaces.
    restarts: usize,
    objective: Objective,
}

/// One historical start, resolved into everything the simulator needs plus the
/// truth it is judged against.
struct Fitted {
    inputs: StartInputs,
    observed_batters_faced: usize,
    observed_strikeouts: u16,
    model_expected_strikeouts: f64,
    bucket: usize,
    seed: u64,
}

/// What the search minimises.
///
/// `Workload` matches simulated batters-faced quantiles to observed ones. It is
/// the original objective and a *proxy*: it never looks at a strikeout. Fitting
/// it 2.25x better (loss 1.80 -> 0.80, measured 2026-08-07) made out-of-sample
/// strikeout Brier slightly **worse** (0.11333 -> 0.11338), because the hazard
/// bought its workload fit by keeping struggling pitchers in longer — and the
/// outcome alphabet is league-average for everyone, so those extra batters
/// strike out at a rate reality does not reproduce.
///
/// `Forecast` scores candidates on the pooled survival-ladder Brier the backtest
/// actually reports, computed on the fit window. It optimises the target rather
/// than a proxy for it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Objective {
    Workload,
    Forecast,
}

impl Objective {
    fn parse(raw: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match raw {
            "workload" => Ok(Objective::Workload),
            "forecast" => Ok(Objective::Forecast),
            other => Err(format!("--objective must be workload or forecast, got {other}").into()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Objective::Workload => "workload",
            Objective::Forecast => "forecast",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(
        root.join("models/pns-model-v2.json"),
        root.join("models/pns-sim-params-v1.json"),
    )?;

    let artifact = ModelArtifact::load(&args.model)?;
    eprintln!(
        "model {} trained through {}",
        artifact.artifact_id, artifact.training.data_through
    );
    let mut starts =
        pns_core::data::dataset::read_csv(root.join("data/processed/training_starts_v3.csv"))?;
    let mut lines: Vec<_> =
        pns_core::data::batters::read_batter_lines(root.join("data/processed/batter_lines.csv"))?
            .into_iter()
            .map(|row| row.line())
            .collect();
    if let Some(through) = args.through {
        starts.retain(|start| start.game_date < through);
        lines.retain(|line| line.game_date < through);
    }
    starts.sort_by_key(|start| (start.pitcher_id, start.game_date));
    lines.sort_by_key(|line| line.game_date);
    eprintln!(
        "{} starts and {} batter lines{}",
        starts.len(),
        lines.len(),
        args.through
            .map(|date| format!(" before {date}"))
            .unwrap_or_default()
    );

    // The model's per-start predictions. `historical_rows` builds the same
    // rolling five-start windows the artifact was trained on; the row is keyed
    // back to its start by (pitcher, date), which is unique for a starter.
    let rows = pns_core::features::historical_rows(&starts, HISTORY_WINDOW)?;
    let mut predictions = HashMap::new();
    for row in &rows {
        // `artifact.score` refuses rows on or before the training cutoff, which
        // is every row here. Going through the model directly is deliberate: the
        // in-sample caveat above is the price, and it is documented, not hidden.
        let projection = artifact.model.score(row)?;
        predictions.insert(
            (row.pitcher_id, row.as_of),
            (
                projection.k_rate_mean,
                projection.batters_faced_mean,
                projection.expected_strikeouts_compositional,
            ),
        );
    }
    eprintln!("scored {} feature rows", predictions.len());

    // Replay in date order, absorbing each date's batter lines only once every
    // start on that date has been served.
    let mut by_date: Vec<_> = starts.iter().collect();
    by_date.sort_by_key(|start| start.game_date);
    let mut rates = BatterKRates::default();
    let mut cursor = 0usize;
    let mut fitted = Vec::new();
    for start in by_date {
        while cursor < lines.len() && lines[cursor].game_date < start.game_date {
            rates.absorb(&lines[cursor]);
            cursor += 1;
        }
        let Some(&(k_rate_mean, batters_faced_mean, model_expected_strikeouts)) =
            predictions.get(&(start.pitcher_id, start.game_date))
        else {
            continue;
        };
        let hand = start.pitch_hand.unwrap_or('R');
        fitted.push(Fitted {
            inputs: StartInputs::from_lineup(
                k_rate_mean,
                batters_faced_mean,
                &rates,
                &start.opponent_batters,
                hand,
            ),
            observed_batters_faced: (start.batters_faced as usize).min(MAX_BATTERS_FACED),
            observed_strikeouts: start.strikeouts,
            model_expected_strikeouts,
            bucket: 0,
            seed: 0,
        });
    }
    if fitted.len() < 500 {
        return Err(format!(
            "only {} starts left to fit; need at least 500",
            fitted.len()
        )
        .into());
    }
    for (index, start) in fitted.iter_mut().enumerate() {
        // Common random numbers: a start keeps its seed across every candidate,
        // so two parameter sets are compared on the same simulated season.
        start.seed = args
            .seed
            .wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    }

    let edges = bucket_edges(&fitted);
    for start in &mut fitted {
        start.bucket = bucket_of(start.inputs.batters_faced_mean, &edges);
    }
    let observed = observed_quantiles(&fitted);
    let labels = bucket_labels(&fitted, &edges);
    eprintln!("bucket edges: {edges:?}");

    // Coordinate descent on a stride sample.
    let stride = fitted.len().div_ceil(SEARCH_STARTS).max(1);
    let sample: Vec<&Fitted> = fitted.iter().step_by(stride).collect();
    eprintln!(
        "searching on {} of {} starts at {} draws each, minimising {} loss",
        sample.len(),
        fitted.len(),
        args.draws,
        args.objective.label()
    );
    let mut hazard = SimParams::defaults().hazard;
    let mut best_loss = f64::INFINITY;
    for restart in 0..=args.restarts {
        let mut local = restart_hazard(restart);
        let mut local_loss = search_loss(args.objective, &local, &sample, &observed, args.draws);
        let entry_loss = local_loss;
        for round in 0..ROUNDS {
            for (axis, (_, low, high)) in AXES.iter().enumerate() {
                let span = (high - low) / 2f64.powi(round as i32);
                let center = axis_value(&local, axis);
                for point in 0..GRID_POINTS {
                    let value = (center - span / 2.0
                        + span * point as f64 / (GRID_POINTS - 1) as f64)
                        .clamp(*low, *high);
                    let mut candidate = local;
                    set_axis(&mut candidate, axis, value);
                    let loss =
                        search_loss(args.objective, &candidate, &sample, &observed, args.draws);
                    if loss < local_loss {
                        local_loss = loss;
                        local = candidate;
                    }
                }
            }
        }
        // How many axes actually left where this restart began is the direct
        // read on whether the search is stalling rather than converging.
        let start = restart_hazard(restart);
        let moved = (0..AXES.len())
            .filter(|&axis| axis_value(&local, axis) != axis_value(&start, axis))
            .count();
        let marker = if local_loss < best_loss {
            "  <- best"
        } else {
            ""
        };
        eprintln!(
            "restart {restart}: loss {entry_loss:.4} -> {local_loss:.4}, {moved}/{} axes moved{marker}",
            AXES.len(),
        );
        if local_loss < best_loss {
            best_loss = local_loss;
            hazard = local;
        }
    }
    for (axis, (name, _, _)) in AXES.iter().enumerate() {
        eprintln!("fitted {name}: {:.4}", axis_value(&hazard, axis));
    }

    // Report on every fitted start, not just the search sample.
    let full: Vec<&Fitted> = fitted.iter().collect();
    let (loss, simulated) = evaluate(&hazard, &full, &observed, args.draws);
    let buckets: Vec<_> = (0..BUCKETS)
        .map(|bucket| {
            let members: Vec<_> = fitted
                .iter()
                .filter(|start| start.bucket == bucket)
                .collect();
            BucketFit {
                label: labels[bucket].clone(),
                starts: members.len(),
                mean_predicted_batters_faced: members
                    .iter()
                    .map(|start| start.inputs.batters_faced_mean)
                    .sum::<f64>()
                    / members.len().max(1) as f64,
                observed_p10: observed[bucket][0],
                observed_p50: observed[bucket][2],
                observed_p90: observed[bucket][4],
                simulated_p10: simulated[bucket][0],
                simulated_p50: simulated[bucket][2],
                simulated_p90: simulated[bucket][4],
            }
        })
        .collect();

    let params = SimParams {
        schema_version: SIM_PARAMS_SCHEMA_VERSION,
        times_through_order: SimParams::defaults().times_through_order,
        outcomes: SimParams::defaults().outcomes,
        hazard,
        fit: FitMetadata {
            through: args.through,
            generated_at: chrono::Utc::now().to_rfc3339(),
            model_artifact_id: artifact.artifact_id.clone(),
            starts_fitted: fitted.len(),
            search_starts: sample.len(),
            draws_per_start: args.draws,
            quantile_loss: Some(loss),
            // The search configuration rides along in the free-form caveat
            // rather than a new schema field: two fits of the same data can
            // otherwise differ with nothing on file to say why.
            caveat: format!(
                "{} [objective: {}; search: {} restarts x {ROUNDS} rounds x {GRID_POINTS} points/axis]",
                caveat(&artifact, args.through),
                args.objective.label(),
                args.restarts
            ),
            buckets: buckets.clone(),
        },
    };
    std::fs::write(&args.out, params.to_json_pretty()?)?;

    println!("hazard: {:?}", params.hazard);
    // Both objectives are reported whichever one was minimised: the pair is the
    // whole point, since a hazard can buy one at the other's expense.
    println!("weighted quantile loss (workload): {loss:.4}");
    println!(
        "pooled ladder Brier, in-sample (forecast): {:.5}",
        forecast_brier(&hazard, &full, args.draws)
    );
    println!(
        "\n{:<14} {:>6} {:>9} | {:>13} | {:>13}",
        "bucket", "starts", "pred BF", "observed BF", "simulated BF"
    );
    println!(
        "{:<14} {:>6} {:>9} | {:>4}{:>5}{:>4} | {:>4}{:>5}{:>4}",
        "", "", "", "p10", "p50", "p90", "p10", "p50", "p90"
    );
    for bucket in &buckets {
        println!(
            "{:<14} {:>6} {:>9.2} | {:>4.0}{:>5.0}{:>4.0} | {:>4.0}{:>5.0}{:>4.0}",
            bucket.label,
            bucket.starts,
            bucket.mean_predicted_batters_faced,
            bucket.observed_p10,
            bucket.observed_p50,
            bucket.observed_p90,
            bucket.simulated_p10,
            bucket.simulated_p50,
            bucket.simulated_p90,
        );
    }
    report_overall(&fitted, &params, args.draws);
    report_strikeout_comparison(&fitted, &params);
    println!("\nwrote {}", args.out.display());
    Ok(())
}

/// Honest provenance note, stored beside the numbers.
///
/// The predictions the hazard is fitted against are always in-sample for the
/// artifact that made them — the fit needs history and the artifact was trained
/// on history. What changes with `--model` is whether that history stops before
/// the window someone intends to test: an artifact whose cutoff predates
/// `--through` leaves the test window untouched by both halves of the fit.
fn caveat(artifact: &ModelArtifact, through: Option<NaiveDate>) -> String {
    let (id, cutoff) = (&artifact.artifact_id, artifact.training.data_through);
    match through {
        Some(through) if cutoff < through => format!(
            "hazard fitted on starts before {through} against predictions from {id}, trained \
             through {cutoff}; those predictions are in-sample, but neither the artifact nor the \
             hazard saw any start after {cutoff}, so a test window opening on {through} is clean"
        ),
        _ => format!(
            "hazard fitted against predictions from {id}, which was trained through {cutoff}; \
             predictions on earlier starts are in-sample"
        ),
    }
}

/// Pooled Brier over the survival ladder, on the starts the fit is allowed to
/// see.
///
/// This is deliberately the same statistic `backtest_sim` reports, computed on a
/// different window: the fit only ever receives starts before `--through`, so
/// optimising it here leaves the test window untouched. Using a different metric
/// for fitting than for judging would just add a second proxy gap on top of the
/// one this exists to remove.
fn forecast_brier(hazard: &pns_core::sim::RemovalHazard, starts: &[&Fitted], draws: usize) -> f64 {
    let params = SimParams {
        hazard: *hazard,
        ..SimParams::defaults()
    };
    let mut total = 0.0;
    let mut legs = 0usize;
    for start in starts {
        let pmf = simulate_start(&start.inputs, &params, draws, start.seed);
        for threshold in RUNGS {
            let survival: f64 = pmf
                .iter()
                .skip(threshold as usize)
                .sum::<f64>()
                .clamp(0.0, 1.0);
            let outcome = f64::from(u8::from(start.observed_strikeouts >= threshold));
            // Monte Carlo debias. A survival read off `draws` samples carries
            // variance p(1-p)/draws, and squaring it adds exactly that much to
            // the Brier — an inflation that grows with how *spread out* the
            // simulated distribution is. Left uncorrected the search would
            // prefer overconfident hazards for a purely numerical reason, which
            // is the opposite of what this project needs. Subtracting the
            // sample variance estimate removes it, and makes this loss
            // comparable to the 10,000-draw figure `backtest_sim` reports.
            let variance = if draws > 1 {
                survival * (1.0 - survival) / (draws as f64 - 1.0)
            } else {
                0.0
            };
            total += (survival - outcome).powi(2) - variance;
            legs += 1;
        }
    }
    total / legs.max(1) as f64
}

/// The quantity the search minimises, dispatched on the chosen objective.
fn search_loss(
    objective: Objective,
    hazard: &pns_core::sim::RemovalHazard,
    starts: &[&Fitted],
    observed: &[[f64; QUANTILE_COUNT]],
    draws: usize,
) -> f64 {
    match objective {
        Objective::Workload => evaluate(hazard, starts, observed, draws).0,
        Objective::Forecast => forecast_brier(hazard, starts, draws),
    }
}

/// Weighted per-bucket quantile loss, plus the simulated quantiles behind it.
fn evaluate(
    hazard: &pns_core::sim::RemovalHazard,
    starts: &[&Fitted],
    observed: &[[f64; QUANTILE_COUNT]],
    draws: usize,
) -> (f64, Vec<[f64; QUANTILE_COUNT]>) {
    let params = SimParams {
        hazard: *hazard,
        ..SimParams::defaults()
    };
    let mut counts = vec![[0u64; MAX_BATTERS_FACED + 1]; BUCKETS];
    let mut members = [0u64; BUCKETS];
    for start in starts {
        members[start.bucket] += 1;
        for draw in simulate_draws(&start.inputs, &params, draws, start.seed) {
            counts[start.bucket][draw.batters_faced as usize] += 1;
        }
    }
    let total_starts: u64 = members.iter().sum();
    let mut loss = 0.0;
    let simulated: Vec<_> = (0..BUCKETS)
        .map(|bucket| {
            let total: u64 = counts[bucket].iter().sum();
            let mut quantiles = [0.0; QUANTILE_COUNT];
            for (index, q) in QUANTILES.iter().enumerate() {
                quantiles[index] = histogram_quantile(&counts[bucket], total, *q);
                let error = quantiles[index] - observed[bucket][index];
                loss += error * error * members[bucket] as f64 / total_starts.max(1) as f64;
            }
            quantiles
        })
        .collect();
    (loss, simulated)
}

/// Batters faced is a small integer, so quantiles come off a count histogram
/// rather than a sort — exact, and fast enough to run inside the search loop.
/// Observed and simulated go through the same function so the loss compares like
/// with like.
fn histogram_quantile(counts: &[u64], total: u64, quantile: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let threshold = (total as f64 * quantile).ceil().max(1.0) as u64;
    let mut cumulative = 0u64;
    for (value, count) in counts.iter().enumerate() {
        cumulative += count;
        if cumulative >= threshold {
            return value as f64;
        }
    }
    (counts.len() - 1) as f64
}

/// Quintile cut points of the model's predicted batters faced.
fn bucket_edges(fitted: &[Fitted]) -> Vec<f64> {
    let mut predicted: Vec<_> = fitted
        .iter()
        .map(|start| start.inputs.batters_faced_mean)
        .collect();
    predicted.sort_by(f64::total_cmp);
    (1..BUCKETS)
        .map(|index| predicted[predicted.len() * index / BUCKETS])
        .collect()
}

fn bucket_of(predicted: f64, edges: &[f64]) -> usize {
    edges.iter().filter(|edge| predicted >= **edge).count()
}

fn bucket_labels(fitted: &[Fitted], edges: &[f64]) -> Vec<String> {
    (0..BUCKETS)
        .map(|bucket| {
            let members: Vec<_> = fitted
                .iter()
                .filter(|start| bucket_of(start.inputs.batters_faced_mean, edges) == bucket)
                .map(|start| start.inputs.batters_faced_mean)
                .collect();
            let low = members.iter().copied().fold(f64::INFINITY, f64::min);
            let high = members.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            format!("{low:.1}-{high:.1}")
        })
        .collect()
}

fn observed_quantiles(fitted: &[Fitted]) -> Vec<[f64; QUANTILE_COUNT]> {
    (0..BUCKETS)
        .map(|bucket| {
            let mut counts = vec![0u64; MAX_BATTERS_FACED + 1];
            for start in fitted.iter().filter(|start| start.bucket == bucket) {
                counts[start.observed_batters_faced] += 1;
            }
            let total: u64 = counts.iter().sum();
            let mut quantiles = [0.0; QUANTILE_COUNT];
            for (index, q) in QUANTILES.iter().enumerate() {
                quantiles[index] = histogram_quantile(&counts, total, *q);
            }
            quantiles
        })
        .collect()
}

/// Pooled batters-faced quantiles across every bucket, which is the number a
/// reader checks first.
fn report_overall(fitted: &[Fitted], params: &SimParams, draws: usize) {
    let mut observed = vec![0u64; MAX_BATTERS_FACED + 1];
    let mut simulated = vec![0u64; MAX_BATTERS_FACED + 1];
    let mut observed_sum = 0.0;
    let mut simulated_sum = 0.0;
    for start in fitted {
        observed[start.observed_batters_faced] += 1;
        observed_sum += start.observed_batters_faced as f64;
        for draw in simulate_draws(&start.inputs, params, draws, start.seed) {
            simulated[draw.batters_faced as usize] += 1;
            simulated_sum += draw.batters_faced as f64;
        }
    }
    let observed_total: u64 = observed.iter().sum();
    let simulated_total: u64 = simulated.iter().sum();
    println!(
        "\noverall BF  observed mean {:.2}  p10 {:.0} p50 {:.0} p90 {:.0}",
        observed_sum / observed_total as f64,
        histogram_quantile(&observed, observed_total, 0.10),
        histogram_quantile(&observed, observed_total, 0.50),
        histogram_quantile(&observed, observed_total, 0.90),
    );
    println!(
        "overall BF simulated mean {:.2}  p10 {:.0} p50 {:.0} p90 {:.0}",
        simulated_sum / simulated_total as f64,
        histogram_quantile(&simulated, simulated_total, 0.10),
        histogram_quantile(&simulated, simulated_total, 0.50),
        histogram_quantile(&simulated, simulated_total, 0.90),
    );
}

/// Simulated expected strikeouts against the model's compositional expectation
/// on a sample of real starts. These are two different objects — one integrates
/// a real lineup and a hook, the other multiplies two margins — so they should
/// not agree exactly, but a large gap would mean a bug, not an insight.
///
/// The second line is the check that tells those two apart: re-running every
/// start against a league-average nine strips out the only genuinely new signal
/// in the simulator. Whatever gap survives that is machinery (the hook, the TTO
/// redistribution, the model's own clamps); whatever the lineup adds back is the
/// point of the exercise.
fn report_strikeout_comparison(fitted: &[Fitted], params: &SimParams) {
    let stride = fitted.len().div_ceil(COMPARISON_STARTS).max(1);
    let sample: Vec<_> = fitted.iter().step_by(stride).collect();
    let simulate = |inputs: &StartInputs, seed: u64| {
        expected_strikeouts(&simulate_start(inputs, params, DEFAULT_DRAWS, seed))
    };
    let with_lineup: Vec<_> = sample
        .iter()
        .map(|start| simulate(&start.inputs, start.seed))
        .collect();
    let league_lineup: Vec<_> = sample
        .iter()
        .map(|start| {
            let neutral = StartInputs::league_lineup(
                start.inputs.k_rate_mean,
                start.inputs.batters_faced_mean,
                start.inputs.league_k_rate,
            );
            simulate(&neutral, start.seed)
        })
        .collect();
    let model: Vec<_> = sample
        .iter()
        .map(|start| start.model_expected_strikeouts)
        .collect();
    println!(
        "\nsimulated vs model E[K] on {} starts ({DEFAULT_DRAWS} draws each)",
        sample.len()
    );
    report_pair("real lineups   ", &with_lineup, &model);
    report_pair("league lineups ", &league_lineup, &model);
    let lineup_effect = with_lineup
        .iter()
        .zip(&league_lineup)
        .map(|(real, neutral)| (real - neutral).abs())
        .sum::<f64>()
        / sample.len() as f64;
    println!("  mean |lineup effect| {lineup_effect:.3} K");
}

fn report_pair(label: &str, simulated: &[f64], model: &[f64]) {
    let n = simulated.len() as f64;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / n;
    let (simulated_mean, model_mean) = (mean(simulated), mean(model));
    let mae = simulated
        .iter()
        .zip(model)
        .map(|(sim, model)| (sim - model).abs())
        .sum::<f64>()
        / n;
    let covariance = simulated
        .iter()
        .zip(model)
        .map(|(s, m)| (s - simulated_mean) * (m - model_mean))
        .sum::<f64>()
        / n;
    let deviation = |values: &[f64], center: f64| {
        (values.iter().map(|v| (v - center).powi(2)).sum::<f64>() / n).sqrt()
    };
    let correlation = covariance
        / (deviation(simulated, simulated_mean) * deviation(model, model_mean)).max(1e-12);
    println!(
        "  {label} mean simulated {simulated_mean:.3}  mean model {model_mean:.3}  \
         bias {:+.3}  MAE {mae:.3}  correlation {correlation:.4}",
        simulated_mean - model_mean
    );
}

fn axis_value(hazard: &pns_core::sim::RemovalHazard, axis: usize) -> f64 {
    match axis {
        0 => hazard.intercept,
        1 => hazard.pa_slope,
        2 => hazard.traffic_slope,
        _ => hazard.pivot_scale,
    }
}

fn set_axis(hazard: &mut pns_core::sim::RemovalHazard, axis: usize, value: f64) {
    match axis {
        0 => hazard.intercept = value,
        1 => hazard.pa_slope = value,
        2 => hazard.traffic_slope = value,
        _ => hazard.pivot_scale = value,
    }
}

fn parse_args(
    default_model: PathBuf,
    default_out: PathBuf,
) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        through: None,
        draws: 250,
        seed: 20_260_730,
        model: default_model,
        out: default_out,
        restarts: DEFAULT_RESTARTS,
        objective: Objective::Workload,
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
            "--through" => args.through = Some(NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?),
            "--draws" => args.draws = value()?.parse()?,
            "--seed" => args.seed = value()?.parse()?,
            "--model" => args.model = PathBuf::from(value()?),
            "--out" => args.out = PathBuf::from(value()?),
            "--restarts" => args.restarts = value()?.parse()?,
            "--objective" => args.objective = Objective::parse(&value()?)?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if args.draws == 0 {
        return Err("--draws must be positive".into());
    }
    Ok(args)
}
