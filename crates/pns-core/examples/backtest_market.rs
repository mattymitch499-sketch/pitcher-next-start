//! Grades the model and the simulator against the Kalshi market on the same
//! historical window `backtest_sim` uses.
//!
//! The backtest already answered "is the simulator better than the model?". This
//! answers the question that decides whether any of it is tradeable: **how do
//! both compare to the price the exchange was actually showing?** A forecaster
//! that beats the incumbent but loses to the market is a better model and still
//! not an edge.
//!
//! Method, and the three places it can go wrong:
//!
//! 1. **The join.** Settled `KXMLBKS` markets carry a pitcher name in the
//!    subtitle and a date plus threshold in the ticker. Both are matched with
//!    [`pns_core::paper::normalize_name`] — the same function the live path uses,
//!    so production and backtest cannot disagree about who a market names. Every
//!    matched leg is then cross-checked against the exchange's own settlement:
//!    if Kalshi says YES and the box score says the pitcher missed the number,
//!    the join found the wrong pitcher and the leg is dropped.
//! 2. **The snapshot.** A market's last candle before settlement is worthless —
//!    the book empties and the quote degenerates toward 0/100. The price used
//!    here is the close of the last hourly candle ending at least
//!    [`SNAPSHOT_LEAD_MINUTES`] before first pitch, which is a quote you could
//!    actually have traded against.
//! 3. **The population.** Kalshi lists a *subset* of pitchers, and a ragged
//!    subset of rungs for each. Model and sim are therefore re-scored on exactly
//!    the matched legs; those numbers are NOT comparable to the full-ladder
//!    numbers in `backtest_sim`'s report, and the coverage table is printed so
//!    the subset is visible rather than implied.
//!
//! # Retention
//!
//! Kalshi's settled-market pagination runs out well before the season does. As
//! of 2026-07-30 it reached back to 2026-05-23 and no further — the `/events`
//! feed still lists March events, but their markets come back empty, so older
//! prices are simply gone. [`DEFAULT_FROM`] is therefore later than the sim
//! backtest's window start, and the run reports the earliest date it actually
//! found so a shrinking retention window shows up as a number rather than as a
//! quietly smaller sample.
//!
//! Usage:
//! `cargo run --release -p pns-core --example backtest_market -- [--model PATH] [--sim-params PATH] [--from YYYY-MM-DD] [--through YYYY-MM-DD] [--draws N] [--limit N] [--out PATH]`

use chrono::{DateTime, Duration, NaiveDate, Utc};
use pns_core::backtest::{self, MarketLeg, MarketStartForecast, StartForecast};
use pns_core::data::kalshi::{self, KalshiClient};
use pns_core::model::artifact::ModelArtifact;
use pns_core::sim::{DEFAULT_DRAWS, SimParams, simulate_start};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// How far before first pitch the usable quote must have closed. Five minutes
/// keeps the snapshot on the right side of lineup scratches and first-pitch
/// volatility while still being a live, tradeable price.
const SNAPSHOT_LEAD_MINUTES: i64 = 5;
/// How far back to look for candles. A full day covers the period when these
/// markets carry real two-sided interest.
const SNAPSHOT_LOOKBACK_HOURS: i64 = 24;
/// Quotes wider than this are treated as no market, matching `paper::decide`.
const MAX_SPREAD_CENTS: u16 = 30;
/// Concurrent candlestick fetches. The public API needs no auth; stay polite.
const CHUNK: usize = 6;
/// Below this share of settlement agreement the join is wrong, not merely
/// incomplete, and the run refuses to report numbers.
const MIN_SETTLEMENT_AGREEMENT: f64 = 0.995;
/// First date with retrievable Kalshi prices, measured 2026-07-30. Earlier
/// dates exist in the sim backtest but have no market to compare against.
const DEFAULT_FROM: (i32, u32, u32) = (2026, 5, 24);

struct Args {
    model: PathBuf,
    sim_params: PathBuf,
    dataset: PathBuf,
    lines: PathBuf,
    from: NaiveDate,
    through: NaiveDate,
    draws: usize,
    limit: Option<usize>,
    out: PathBuf,
}

#[derive(Serialize)]
struct Provenance {
    model_artifact_id: String,
    model_trained_through: NaiveDate,
    sim_params_fit_through: Option<NaiveDate>,
    window_from: NaiveDate,
    window_through: NaiveDate,
    draws_per_start: usize,
    snapshot_lead_minutes: i64,
    generated_at: String,
}

/// Where every candidate leg ended up. A benchmark that reports only what it
/// kept is a benchmark you cannot audit.
#[derive(Serialize, Default)]
struct Coverage {
    starts_replayed: usize,
    starts_with_any_market: usize,
    settled_markets_fetched: usize,
    /// Earliest event date among the markets Kalshi actually returned. Later
    /// than `window_from` means retention, not a bug — but it means the sample
    /// is smaller than the window implies.
    market_data_earliest: Option<NaiveDate>,
    candidate_legs: usize,
    matched_legs: usize,
    dropped_no_first_pitch: usize,
    dropped_no_candles: usize,
    dropped_no_valid_quote: usize,
    dropped_settlement_mismatch: usize,
    settlement_checked: usize,
    settlement_agreed: usize,
}

#[derive(Serialize)]
struct Report {
    provenance: Provenance,
    coverage: Coverage,
    evaluation: backtest::MarketEvaluation,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let artifact = ModelArtifact::load(&args.model)?;
    let params = SimParams::load_path(&args.sim_params)?;
    if artifact.training.data_through >= args.from {
        return Err(format!(
            "model artifact is trained through {}, inside the {} window; \
             train a historical artifact first",
            artifact.training.data_through, args.from
        )
        .into());
    }
    let starts = pns_core::data::dataset::read_csv(&args.dataset)?;
    let lines: Vec<_> = pns_core::data::batters::read_batter_lines(&args.lines)?
        .into_iter()
        .map(|row| row.line())
        .collect();
    // ReplayCase carries ids, not names; the market join needs the name.
    let names: HashMap<u32, String> = starts
        .iter()
        .map(|start| (start.pitcher_id, start.pitcher_name.clone()))
        .collect();

    let replay = backtest::replay(&starts, &lines, args.from, args.through)?;
    eprintln!(
        "replaying {} starts from {} to {}",
        replay.cases.len(),
        args.from,
        args.through
    );

    // ---- market side -----------------------------------------------------
    let kalshi = KalshiClient::new()?;
    eprintln!("fetching settled KXMLBKS markets since {}...", args.from);
    let settled = kalshi.settled_strikeout_markets(args.from)?;
    eprintln!("  {} settled markets", settled.len());

    // (normalized pitcher name, event date) -> the rungs Kalshi listed.
    let mut by_pitcher_date: HashMap<(String, NaiveDate), Vec<&kalshi::KalshiMarket>> =
        HashMap::new();
    let mut market_data_earliest: Option<NaiveDate> = None;
    for market in &settled {
        let Some((event_date, _)) = kalshi::parse_ticker(&market.ticker) else {
            continue;
        };
        market_data_earliest = Some(match market_data_earliest {
            Some(earliest) => earliest.min(event_date),
            None => event_date,
        });
        let Some(raw_name) = kalshi::market_pitcher_name(market) else {
            continue;
        };
        by_pitcher_date
            .entry((pns_core::paper::normalize_name(raw_name), event_date))
            .or_default()
            .push(market);
    }
    if let Some(earliest) = market_data_earliest
        && earliest > args.from
    {
        eprintln!(
            "  NOTE: Kalshi returned nothing before {earliest}, though the window opens {}. \
             Older prices are outside its retention, so those starts have no benchmark.",
            args.from
        );
    }

    let mlb = pns_core::data::MlbStatsClient::new()?;
    eprintln!("fetching first-pitch times...");
    let first_pitch: HashMap<u32, DateTime<Utc>> = mlb
        .game_index(args.from, args.through)?
        .into_iter()
        .filter_map(|entry| {
            let raw = entry.first_pitch?;
            let parsed = DateTime::parse_from_rfc3339(&raw).ok()?;
            Some((entry.game_pk, parsed.with_timezone(&Utc)))
        })
        .collect();
    eprintln!("  {} games with a first-pitch timestamp", first_pitch.len());

    // ---- score every start, then attach whatever the market priced --------
    let mut coverage = Coverage {
        starts_replayed: replay.cases.len(),
        settled_markets_fetched: settled.len(),
        market_data_earliest,
        ..Coverage::default()
    };
    let mut forecasts = Vec::with_capacity(replay.cases.len());

    let cases: Vec<_> = match args.limit {
        Some(limit) => replay.cases.iter().take(limit).collect(),
        None => replay.cases.iter().collect(),
    };
    for (index, case) in cases.iter().enumerate() {
        let projection = artifact.score(&case.row)?;
        let sim_pmf = simulate_start(
            &case.inputs(projection.k_rate_mean, projection.batters_faced_mean),
            &params,
            args.draws,
            case.seed(),
        );
        let forecast = StartForecast {
            pitcher_id: case.pitcher_id,
            game_date: case.game_date,
            model_pmf: projection.strikeout_probabilities,
            sim_pmf,
            actual_strikeouts: case.actual_strikeouts,
        };

        let key = names
            .get(&case.pitcher_id)
            .map(|name| (pns_core::paper::normalize_name(name), case.game_date));
        let markets = key
            .as_ref()
            .and_then(|key| by_pitcher_date.get(key))
            .cloned()
            .unwrap_or_default();
        if !markets.is_empty() {
            coverage.starts_with_any_market += 1;
        }
        coverage.candidate_legs += markets.len();

        let market_legs = match first_pitch.get(&case.game_pk) {
            Some(start_time) => {
                let legs = snapshot_legs(&kalshi, &markets, *start_time, &mut coverage);
                // Settlement cross-check: the exchange's own answer must agree
                // with the box score, or the join named the wrong pitcher.
                legs.into_iter()
                    .filter(|(leg, result)| {
                        let Some(result) = result else {
                            return true;
                        };
                        coverage.settlement_checked += 1;
                        let expected = case.actual_strikeouts >= leg.threshold;
                        let agrees = (result == "yes") == expected;
                        if agrees {
                            coverage.settlement_agreed += 1;
                        } else {
                            coverage.dropped_settlement_mismatch += 1;
                        }
                        agrees
                    })
                    .map(|(leg, _)| leg)
                    .collect()
            }
            None => {
                coverage.dropped_no_first_pitch += markets.len();
                Vec::new()
            }
        };
        coverage.matched_legs += market_legs.len();
        forecasts.push(MarketStartForecast {
            forecast,
            market_legs,
        });

        if index % 100 == 99 {
            eprintln!(
                "  {} / {} starts  ({} legs matched)",
                index + 1,
                cases.len(),
                coverage.matched_legs
            );
        }
    }

    if coverage.matched_legs == 0 {
        return Err(format!(
            "no market leg could be matched. Kalshi's earliest retrievable market is {:?} \
             and this run asked for {}..={} — if those do not overlap, the prices are \
             outside Kalshi's retention window",
            coverage.market_data_earliest, args.from, args.through
        )
        .into());
    }
    let agreement = if coverage.settlement_checked == 0 {
        1.0
    } else {
        coverage.settlement_agreed as f64 / coverage.settlement_checked as f64
    };
    if agreement < MIN_SETTLEMENT_AGREEMENT {
        return Err(format!(
            "settlement agreement {:.3}% is below {:.1}% — the pitcher join is \
             mis-attributing markets; refusing to report numbers",
            agreement * 100.0,
            MIN_SETTLEMENT_AGREEMENT * 100.0
        )
        .into());
    }

    let evaluation = backtest::evaluate_against_market(&forecasts);
    let report = Report {
        provenance: Provenance {
            model_artifact_id: artifact.artifact_id.clone(),
            model_trained_through: artifact.training.data_through,
            sim_params_fit_through: params.fit.through,
            window_from: args.from,
            window_through: args.through,
            draws_per_start: args.draws,
            snapshot_lead_minutes: SNAPSHOT_LEAD_MINUTES,
            generated_at: Utc::now().to_rfc3339(),
        },
        coverage,
        evaluation,
    };
    print_report(&report, agreement);
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

/// The pre-first-pitch snapshot for every listed rung of one start, paired with
/// the exchange's settlement so the caller can cross-check the join.
fn snapshot_legs(
    kalshi: &KalshiClient,
    markets: &[&kalshi::KalshiMarket],
    first_pitch: DateTime<Utc>,
    coverage: &mut Coverage,
) -> Vec<(MarketLeg, Option<String>)> {
    let cutoff = (first_pitch - Duration::minutes(SNAPSHOT_LEAD_MINUTES)).timestamp();
    let window_start = (first_pitch - Duration::hours(SNAPSHOT_LOOKBACK_HOURS)).timestamp();
    let mut legs = Vec::new();
    for chunk in markets.chunks(CHUNK) {
        let fetched: Vec<_> = std::thread::scope(|scope| {
            chunk
                .iter()
                .map(|market| {
                    let kalshi = kalshi.clone();
                    let ticker = market.ticker.clone();
                    scope.spawn(move || kalshi.candlesticks(&ticker, window_start, cutoff))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("candlestick worker panicked"))
                .collect()
        });
        for (market, candles) in chunk.iter().zip(fetched) {
            let Some((_, threshold)) = kalshi::parse_ticker(&market.ticker) else {
                continue;
            };
            let Ok(candles) = candles else {
                coverage.dropped_no_candles += 1;
                continue;
            };
            if candles.is_empty() {
                coverage.dropped_no_candles += 1;
                continue;
            }
            // The LAST usable candle before the cutoff, not the last candle:
            // the book thins as settlement nears and the final quotes are noise.
            let quote = candles
                .iter()
                .filter(|candle| candle.end_period_ts <= cutoff)
                .filter_map(|candle| Some((candle.yes_bid.close?, candle.yes_ask.close?)))
                .rfind(|(bid, ask)| {
                    ask >= bid
                        && ask - bid <= MAX_SPREAD_CENTS
                        && (1..=99).contains(bid)
                        && (1..=99).contains(ask)
                });
            let Some((bid, ask)) = quote else {
                coverage.dropped_no_valid_quote += 1;
                continue;
            };
            legs.push((
                MarketLeg {
                    threshold,
                    probability: f64::from(bid + ask) / 200.0,
                },
                market.result.clone().filter(|value| !value.is_empty()),
            ));
        }
    }
    legs
}

fn print_report(report: &Report, agreement: f64) {
    let coverage = &report.coverage;
    let evaluation = &report.evaluation;
    println!(
        "\nmarket benchmark {} .. {}",
        report.provenance.window_from, report.provenance.window_through
    );
    println!(
        "model {} trained through {}; sim params through {:?}",
        report.provenance.model_artifact_id,
        report.provenance.model_trained_through,
        report.provenance.sim_params_fit_through
    );
    println!(
        "quote = mid of the last hourly candle closing >= {} min before first pitch",
        report.provenance.snapshot_lead_minutes
    );

    println!("\ncoverage");
    if let Some(earliest) = coverage.market_data_earliest {
        println!(
            "  earliest retrievable Kalshi market: {earliest} \
             ({} settled markets fetched)",
            coverage.settled_markets_fetched
        );
    }
    println!(
        "  {} starts replayed; {} had at least one listed market",
        coverage.starts_replayed, coverage.starts_with_any_market
    );
    println!(
        "  {} candidate legs -> {} matched",
        coverage.candidate_legs, coverage.matched_legs
    );
    println!(
        "  dropped: {} no first pitch, {} no candles, {} no valid quote, {} settlement mismatch",
        coverage.dropped_no_first_pitch,
        coverage.dropped_no_candles,
        coverage.dropped_no_valid_quote,
        coverage.dropped_settlement_mismatch
    );
    println!(
        "  settlement agreement {:.3}% ({} of {} checked)",
        agreement * 100.0,
        coverage.settlement_agreed,
        coverage.settlement_checked
    );

    println!("\nladder on matched legs only: P(K >= t) and the Brier each paid");
    println!(
        "{:>4} {:>6} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10} {:>10}",
        "t", "legs", "model", "sim", "market", "observed", "mdl Brier", "sim Brier", "mkt Brier"
    );
    for rung in &evaluation.rungs {
        println!(
            "{:>4} {:>6} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>10.4} {:>10.4} {:>10.4}",
            rung.threshold,
            rung.legs,
            rung.model_probability,
            rung.sim_probability,
            rung.market_probability,
            rung.observed_rate,
            rung.model_brier,
            rung.sim_brier,
            rung.market_brier,
        );
    }

    println!(
        "\npooled over {} matched legs on {} starts \
         (NOT comparable to the full-ladder backtest numbers)",
        evaluation.legs, evaluation.starts
    );
    println!(
        "{:<20} {:>11} {:>11} {:>11}",
        "", "model", "sim", "market"
    );
    let line = |label: &str, model: f64, sim: f64, market: f64| {
        println!("{label:<20} {model:>11.5} {sim:>11.5} {market:>11.5}");
    };
    line(
        "pooled Brier",
        evaluation.model.pooled_brier,
        evaluation.sim.pooled_brier,
        evaluation.market.pooled_brier,
    );
    line(
        "rung log loss",
        evaluation.model.rung_log_loss,
        evaluation.sim.rung_log_loss,
        evaluation.market.rung_log_loss,
    );
    line(
        "ECE (10 bins)",
        evaluation.model.ece,
        evaluation.sim.ece,
        evaluation.market.ece,
    );

    let show = |label: &str, difference: &Option<pns_core::paper_stats::Interval>| {
        if let Some(difference) = difference {
            let interval = match (difference.low, difference.high) {
                (Some(low), Some(high)) => format!("[{low:+.5}, {high:+.5}]"),
                _ => "interval suppressed (too few starts)".into(),
            };
            println!("{label}: {:+.5}  95% CI {interval}", difference.point);
        }
    };
    println!();
    show(
        "sim minus market pooled Brier  ",
        &evaluation.sim_minus_market_brier,
    );
    show(
        "model minus market pooled Brier",
        &evaluation.model_minus_market_brier,
    );
    println!("(negative means it beat the market; clustered by pitcher-start)");
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        model: root.join("models/pns-model-hist-2026-04-30.json"),
        sim_params: root.join("models/pns-sim-params-hist-2026-04-30.json"),
        dataset: root.join("data/processed/training_starts_v3.csv"),
        lines: root.join("data/processed/batter_lines.csv"),
        from: NaiveDate::from_ymd_opt(DEFAULT_FROM.0, DEFAULT_FROM.1, DEFAULT_FROM.2)
            .expect("valid default window start"),
        through: NaiveDate::from_ymd_opt(2026, 7, 16).expect("valid default window end"),
        draws: DEFAULT_DRAWS,
        limit: None,
        out: root.join("outputs/backtest_market_2026-04-30.json"),
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
            "--limit" => args.limit = Some(value()?.parse()?),
            "--out" => args.out = PathBuf::from(value()?),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if args.draws == 0 {
        return Err("--draws must be positive".into());
    }
    if args.from > args.through {
        return Err("--from must not be after --through".into());
    }
    Ok(args)
}
