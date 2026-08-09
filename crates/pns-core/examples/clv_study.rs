//! Does the model, or the simulator, lead the Kalshi price?
//!
//! [`backtest_market`] asked whether either forecaster beats the price just
//! before first pitch. Both lost, with confidence. This asks the weaker and more
//! interesting question: when a forecaster disagrees with an *early* price, does
//! the price later move toward it? See [`pns_core::clv`] for why a lead over a
//! well-calibrated close is worth money even when beating that close is not.
//!
//! # This runs entirely off the candlestick cache, and must keep doing so
//!
//! Kalshi's price history has roughly two months of retention: as of 2026-07-30
//! settled-market pagination reached back only to 2026-05-23, and by August the
//! window measured here is gone from the API. The only surviving copy is the
//! local cache (backed up to `pitcher-next-start-backups/`).
//!
//! `data::kalshi::candlesticks` keys its cache on `(ticker, start_ts, end_ts)`,
//! so **this example must request byte-identically the same window
//! `backtest_market` did** — `first_pitch - `[`LOOKBACK_HOURS`]` .. first_pitch -
//! `[`SNAPSHOT_LEAD_MINUTES`]. Change either constant and every request misses
//! the cache, hits the API, and comes back empty. That is why they are copied
//! here as constants with this warning rather than tuned.
//!
//! The saving grace is that `backtest_market` already fetched a full 24 hours
//! and used only the last candle, so the entire pre-game price path is sitting
//! in that cache unused.
//!
//! Usage:
//! `cargo run --release -p pns-core --example clv_study -- [--model PATH] [--sim-params PATH] [--from YYYY-MM-DD] [--through YYYY-MM-DD] [--draws N] [--limit N] [--out PATH]`

use chrono::{DateTime, Duration, NaiveDate, Utc};
use pns_core::backtest;
use pns_core::clv::{AnchorLeg, AnchorResult, AnchorStart, evaluate_anchor};
use pns_core::data::kalshi::{self, KalshiClient};
use pns_core::model::artifact::ModelArtifact;
use pns_core::sim::{DEFAULT_DRAWS, SimParams, simulate_start};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// MUST match `backtest_market::SNAPSHOT_LEAD_MINUTES` — see the cache warning.
const SNAPSHOT_LEAD_MINUTES: i64 = 5;
/// MUST match `backtest_market::SNAPSHOT_LOOKBACK_HOURS` — see the cache warning.
const LOOKBACK_HOURS: i64 = 24;
/// Quotes wider than this are treated as no market, matching `paper::decide`.
const MAX_SPREAD_CENTS: u16 = 30;
const CHUNK: usize = 6;
const MIN_SETTLEMENT_AGREEMENT: f64 = 0.995;
const DEFAULT_FROM: (i32, u32, u32) = (2026, 5, 24);

/// Hours before first pitch to price.
///
/// The earliest is 23h, not 24h: the fetch window *starts* at
/// `first_pitch - `[`LOOKBACK_HOURS`], so no candle can end at or before that
/// instant and a 24h anchor matches nothing at all. Anchors must sit strictly
/// inside the window.
const ANCHORS: [f64; 6] = [23.0, 18.0, 12.0, 6.0, 3.0, 1.0];

/// Anchors at which the simulator is honestly available. It needs a confirmed
/// lineup, and lineups post roughly three hours out; scoring it at 24h would
/// credit it with information nobody had. Lineup posting times are not recorded
/// anywhere, so this is a stated assumption, not a measurement.
const SIM_MAX_HOURS: f64 = 3.0;

/// Primary anchors, fixed before any number existed. Six anchors times two
/// forecasters is twelve chances to find something; everything except these two
/// is descriptive.
const PRIMARY_MODEL_HOURS: f64 = 6.0;
const PRIMARY_SIM_HOURS: f64 = 3.0;

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
    lookback_hours: i64,
    sim_max_hours: f64,
    primary_model_hours: f64,
    primary_sim_hours: f64,
    generated_at: String,
}

/// Where every candidate leg ended up. Coverage is part of the result: early
/// books are thin, and a study that silently drops them would report a slope
/// measured only on the most liquid markets while implying it held everywhere.
#[derive(Serialize, Default)]
struct Coverage {
    /// Starts actually processed. Differs from the replay total when `--limit`
    /// is set, so a smoke run cannot be mistaken for a full one.
    starts_processed: usize,
    starts_replayed: usize,
    starts_with_any_market: usize,
    settled_markets_fetched: usize,
    market_data_earliest: Option<NaiveDate>,
    candidate_legs: usize,
    /// Legs with a usable close quote and an agreeing settlement — the pool the
    /// anchors draw from.
    matched_legs: usize,
    dropped_no_first_pitch: usize,
    dropped_no_candles: usize,
    dropped_no_close_quote: usize,
    dropped_settlement_mismatch: usize,
    settlement_checked: usize,
    settlement_agreed: usize,
    /// Of `matched_legs`, how many had a usable quote at each anchor.
    legs_priced_at_anchor: Vec<(String, usize)>,
}

#[derive(Serialize)]
struct Report {
    provenance: Provenance,
    coverage: Coverage,
    model: Vec<AnchorResult>,
    sim: Vec<AnchorResult>,
}

/// One matched leg with its whole usable price path.
struct PricedLeg {
    threshold: u16,
    model_forecast: f64,
    sim_forecast: f64,
    settled_yes: bool,
    close_mid: f64,
    /// Anchor hours -> (bid, ask) in cents.
    anchor_quotes: HashMap<u64, (u16, u16)>,
}

fn survival(pmf: &[f64], threshold: u16) -> f64 {
    pmf.iter()
        .skip(threshold as usize)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Anchor hours as a hash key. Anchors are fixed constants with one decimal, so
/// scaling by ten is exact and avoids keying a map on a float.
fn anchor_key(hours: f64) -> u64 {
    (hours * 10.0).round() as u64
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

    let kalshi = KalshiClient::new()?;
    eprintln!("fetching settled KXMLBKS markets since {}...", args.from);
    let settled = kalshi.settled_strikeout_markets(args.from)?;
    eprintln!("  {} settled markets", settled.len());

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
            "  NOTE: Kalshi returned nothing before {earliest}, though the window opens {}.",
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

    let mut coverage = Coverage {
        starts_replayed: replay.cases.len(),
        settled_markets_fetched: settled.len(),
        market_data_earliest,
        ..Coverage::default()
    };

    // Anchor hours -> per-start legs, for each forecaster.
    let mut model_by_anchor: HashMap<u64, Vec<AnchorStart>> = HashMap::new();
    let mut sim_by_anchor: HashMap<u64, Vec<AnchorStart>> = HashMap::new();

    let cases: Vec<_> = match args.limit {
        Some(limit) => replay.cases.iter().take(limit).collect(),
        None => replay.cases.iter().collect(),
    };
    coverage.starts_processed = cases.len();
    for (index, case) in cases.iter().enumerate() {
        let projection = artifact.score(&case.row)?;
        let sim_pmf = simulate_start(
            &case.inputs(projection.k_rate_mean, projection.batters_faced_mean),
            &params,
            args.draws,
            case.seed(),
        );
        let model_pmf = &projection.strikeout_probabilities;

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

        let Some(start_time) = first_pitch.get(&case.game_pk) else {
            coverage.dropped_no_first_pitch += markets.len();
            continue;
        };

        let priced = price_legs(
            &kalshi,
            &markets,
            *start_time,
            case.actual_strikeouts,
            model_pmf,
            &sim_pmf,
            &mut coverage,
        );
        coverage.matched_legs += priced.len();

        for hours in ANCHORS {
            let key = anchor_key(hours);
            let model_legs: Vec<AnchorLeg> = priced
                .iter()
                .filter_map(|leg| leg.anchor_leg(key, leg.model_forecast))
                .collect();
            if !model_legs.is_empty() {
                model_by_anchor.entry(key).or_default().push(AnchorStart {
                    pitcher_id: case.pitcher_id,
                    game_date: case.game_date,
                    legs: model_legs,
                });
            }
            if hours <= SIM_MAX_HOURS {
                let sim_legs: Vec<AnchorLeg> = priced
                    .iter()
                    .filter_map(|leg| leg.anchor_leg(key, leg.sim_forecast))
                    .collect();
                if !sim_legs.is_empty() {
                    sim_by_anchor.entry(key).or_default().push(AnchorStart {
                        pitcher_id: case.pitcher_id,
                        game_date: case.game_date,
                        legs: sim_legs,
                    });
                }
            }
        }

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
            "no market leg could be priced. Kalshi's earliest retrievable market is {:?} \
             and this run asked for {}..={}. If the candlestick cache was cleared, this \
             window is unrecoverable — restore from pitcher-next-start-backups/",
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
            "settlement agreement {:.3}% is below the {:.1}% floor — the pitcher join is \
             wrong, not merely incomplete, so no number here is trustworthy",
            agreement * 100.0,
            MIN_SETTLEMENT_AGREEMENT * 100.0
        )
        .into());
    }
    eprintln!(
        "settlement agreement {:.3}% ({}/{})",
        agreement * 100.0,
        coverage.settlement_agreed,
        coverage.settlement_checked
    );

    let mut model: Vec<AnchorResult> = ANCHORS
        .iter()
        .filter_map(|hours| {
            model_by_anchor
                .get(&anchor_key(*hours))
                .map(|starts| evaluate_anchor(*hours, starts))
        })
        .collect();
    let mut sim: Vec<AnchorResult> = ANCHORS
        .iter()
        .filter(|hours| **hours <= SIM_MAX_HOURS)
        .filter_map(|hours| {
            sim_by_anchor
                .get(&anchor_key(*hours))
                .map(|starts| evaluate_anchor(*hours, starts))
        })
        .collect();
    model.sort_by(|a, b| b.hours_before.total_cmp(&a.hours_before));
    sim.sort_by(|a, b| b.hours_before.total_cmp(&a.hours_before));

    coverage.legs_priced_at_anchor = model
        .iter()
        .map(|result| (format!("T-{}h", result.hours_before), result.legs))
        .collect();

    report(&model, &sim, &coverage);

    let report_json = Report {
        provenance: Provenance {
            model_artifact_id: artifact.artifact_id.clone(),
            model_trained_through: artifact.training.data_through,
            sim_params_fit_through: params.fit.through,
            window_from: args.from,
            window_through: args.through,
            draws_per_start: args.draws,
            snapshot_lead_minutes: SNAPSHOT_LEAD_MINUTES,
            lookback_hours: LOOKBACK_HOURS,
            sim_max_hours: SIM_MAX_HOURS,
            primary_model_hours: PRIMARY_MODEL_HOURS,
            primary_sim_hours: PRIMARY_SIM_HOURS,
            generated_at: Utc::now().to_rfc3339(),
        },
        coverage,
        model,
        sim,
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&report_json)?)?;
    println!("\nwrote {}", args.out.display());
    Ok(())
}

impl PricedLeg {
    /// This leg as of one anchor, carrying the supplied forecast. `None` when the
    /// book had no usable quote that far out — the common case early, and the
    /// reason coverage is reported per anchor.
    fn anchor_leg(&self, key: u64, forecast: f64) -> Option<AnchorLeg> {
        let (bid, ask) = *self.anchor_quotes.get(&key)?;
        Some(AnchorLeg {
            threshold: self.threshold,
            forecast,
            anchor_yes_bid: bid,
            anchor_yes_ask: ask,
            close_mid: self.close_mid,
            settled_yes: self.settled_yes,
        })
    }
}

/// Builds the price path for every listed rung of one start.
fn price_legs(
    kalshi: &KalshiClient,
    markets: &[&kalshi::KalshiMarket],
    first_pitch: DateTime<Utc>,
    actual_strikeouts: u16,
    model_pmf: &[f64],
    sim_pmf: &[f64],
    coverage: &mut Coverage,
) -> Vec<PricedLeg> {
    let cutoff = (first_pitch - Duration::minutes(SNAPSHOT_LEAD_MINUTES)).timestamp();
    let window_start = (first_pitch - Duration::hours(LOOKBACK_HOURS)).timestamp();
    let mut out = Vec::new();

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

            // Same rule as backtest_market: the last *usable* quote before the
            // cutoff, not the last quote, because the book empties near settlement.
            let Some((close_bid, close_ask)) = usable_quote(&candles, cutoff) else {
                coverage.dropped_no_close_quote += 1;
                continue;
            };

            // The exchange's own settlement must agree with the box score, or the
            // pitcher join found the wrong person.
            if let Some(result) = market.result.as_ref().filter(|value| !value.is_empty()) {
                coverage.settlement_checked += 1;
                let expected = actual_strikeouts >= threshold;
                if (result == "yes") == expected {
                    coverage.settlement_agreed += 1;
                } else {
                    coverage.dropped_settlement_mismatch += 1;
                    continue;
                }
            }

            let mut anchor_quotes = HashMap::new();
            for hours in ANCHORS {
                let anchor_ts =
                    (first_pitch - Duration::minutes((hours * 60.0) as i64)).timestamp();
                if let Some(quote) = usable_quote(&candles, anchor_ts) {
                    anchor_quotes.insert(anchor_key(hours), quote);
                }
            }

            out.push(PricedLeg {
                threshold,
                model_forecast: survival(model_pmf, threshold),
                sim_forecast: survival(sim_pmf, threshold),
                settled_yes: actual_strikeouts >= threshold,
                close_mid: f64::from(close_bid + close_ask) / 200.0,
                anchor_quotes,
            });
        }
    }
    out
}

/// The last two-sided, sane quote at or before `at`. Mirrors the filter in
/// `backtest_market::snapshot_legs` so the two studies cannot disagree about
/// what counts as a tradeable price.
fn usable_quote(candles: &[kalshi::KalshiCandle], at: i64) -> Option<(u16, u16)> {
    candles
        .iter()
        .filter(|candle| candle.end_period_ts <= at)
        .filter_map(|candle| Some((candle.yes_bid.close?, candle.yes_ask.close?)))
        .rfind(|(bid, ask)| {
            ask >= bid
                && ask - bid <= MAX_SPREAD_CENTS
                && (1..=99).contains(bid)
                && (1..=99).contains(ask)
        })
}

fn interval_text(interval: &Option<pns_core::paper_stats::Interval>, scale: f64) -> String {
    match interval {
        None => "        --                        ".into(),
        Some(interval) => match (interval.low, interval.high) {
            (Some(low), Some(high)) => format!(
                "{:>+9.4}  [{:>+8.4}, {:>+8.4}]",
                interval.point * scale,
                low * scale,
                high * scale
            ),
            _ => format!("{:>+9.4}  (interval suppressed)  ", interval.point * scale),
        },
    }
}

fn report(model: &[AnchorResult], sim: &[AnchorResult], coverage: &Coverage) {
    println!("\ncoverage");
    println!(
        "  starts processed          {} of {} replayed",
        coverage.starts_processed, coverage.starts_replayed
    );
    println!(
        "  starts with any market    {}",
        coverage.starts_with_any_market
    );
    println!("  candidate legs            {}", coverage.candidate_legs);
    println!("  matched legs (usable)     {}", coverage.matched_legs);
    println!(
        "  dropped: no candles {}, no close quote {}, no first pitch {}, settlement mismatch {}",
        coverage.dropped_no_candles,
        coverage.dropped_no_close_quote,
        coverage.dropped_no_first_pitch,
        coverage.dropped_settlement_mismatch
    );

    for (label, results, primary) in [
        ("MODEL", model, PRIMARY_MODEL_HOURS),
        ("SIM (lineup-aware)", sim, PRIMARY_SIM_HOURS),
    ] {
        println!("\n{label}: does the price move toward this forecast?");
        println!(
            "{:>6} {:>7} {:>7} {:>8} {:>8}   {:^33}   {:^33}",
            "anchor",
            "legs",
            "starts",
            "|signal|",
            "|drift|",
            "drift slope",
            "P&L per contract (c)"
        );
        for result in results {
            let star = if (result.hours_before - primary).abs() < 1e-9 {
                " *"
            } else {
                "  "
            };
            println!(
                "{:>4}h{star}{:>7} {:>7} {:>8.4} {:>8.4}   {}   {}",
                result.hours_before,
                result.legs,
                result.starts,
                result.mean_abs_signal,
                result.mean_abs_drift,
                interval_text(&result.drift_slope, 1.0),
                interval_text(&result.pnl_per_contract, 1.0),
            );
        }
        if let Some(result) = results
            .iter()
            .find(|result| (result.hours_before - primary).abs() < 1e-9)
        {
            let hit = result
                .hit_rate
                .as_ref()
                .map(|interval| format!("{:.1}%", interval.point * 100.0))
                .unwrap_or_else(|| "--".into());
            println!(
                "  * primary anchor (fixed in advance): hit rate {hit} over {} disagreeing legs",
                result.signal_legs
            );
        }
    }
    println!(
        "\nslope > 0 means the price moved toward the forecast; 1.0 would mean it landed on it."
    );
    println!("only the starred anchors were pre-registered — the rest are descriptive.");
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
        out: root.join("outputs/clv_study_2026-04-30.json"),
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
    Ok(args)
}
