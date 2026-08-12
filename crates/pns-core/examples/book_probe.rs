//! Do sportsbook strikeout props differ from Kalshi enough for the answer to be
//! different?
//!
//! ## Why this exists
//!
//! Directional betting against Kalshi is closed: over 8,284 legs the market beat
//! both the model and the simulator with confidence, and the optimal blend
//! weight on the model was 0.014. But a sportsbook line is a *different price*,
//! and Stage A was planned in July 2026 and never run.
//!
//! The strong prior is that the two markets agree, and book costs are **worse**
//! (about 3.2pp per side at -115 against Kalshi's 1.67pp half-spread). So this
//! probe answers the question for **$0** on The Odds API free tier before
//! anything gets bought: if book lines hug Kalshi, we already know the answer.
//!
//! ## The decision rule, fixed before any data exists
//!
//! Primary statistic: RMS gap between the **devigged book consensus** and the
//! Kalshi mid, against the model's known ~8.7pp gap to that same mid.
//!
//! * RMS <= ~2pp — books carry the same information as Kalshi. Thread closes,
//!   nothing gets bought.
//! * Materially wider — re-measure using the **best available number across
//!   books** rather than the consensus, since that is what you would actually
//!   bet, and only then consider the paid historical month.
//!
//! Naming this before the numbers exist is deliberate: picking it afterwards is
//! the failure mode that produced the quarter-Kelly and 12+pp reversals.
//!
//! ## Cost control
//!
//! The free tier is 500 credits a month, and player props bill per event. Every
//! response carries `x-requests-remaining`; this run prints it, and **stops
//! before the floor** rather than discovering the cap by exhausting it — a spent
//! key returns empty slates that look exactly like a day with no lines.
//!
//! Needs `ODDS_API_KEY` in the environment. Get a free key at the-odds-api.com;
//! it is never read from a file or written to one.
//!
//! Usage:
//! `cargo run --release -p pns-core --example book_probe -- [--date YYYY-MM-DD] [--out PATH] [--min-credits N] [--max-events N]`

use chrono::NaiveDate;
use pns_core::card::DecisionCard;
use pns_core::data::kalshi::{self, KalshiClient};
use pns_core::paper::{normalize_name, pmf_over_probability};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

const ODDS_BASE: &str = "https://api.the-odds-api.com/v4";
const SPORT: &str = "baseball_mlb";
const PROP_MARKET: &str = "pitcher_strikeouts";

struct Args {
    date: NaiveDate,
    out: PathBuf,
    /// Stop before the remaining credit balance falls below this.
    min_credits: u32,
    max_events: usize,
}

#[derive(Deserialize)]
struct OddsEvent {
    id: String,
    commence_time: String,
    home_team: String,
    away_team: String,
}

#[derive(Deserialize)]
struct EventOdds {
    #[serde(default)]
    bookmakers: Vec<Bookmaker>,
}
#[derive(Deserialize)]
struct Bookmaker {
    key: String,
    #[serde(default)]
    markets: Vec<Market>,
}
#[derive(Deserialize)]
struct Market {
    key: String,
    #[serde(default)]
    outcomes: Vec<Outcome>,
}
#[derive(Deserialize)]
struct Outcome {
    /// `"Over"` or `"Under"`.
    name: String,
    /// The player the prop is on.
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    price: Option<f64>,
    #[serde(default)]
    point: Option<f64>,
}

/// One book's two-sided quote on one pitcher's strikeout line.
#[derive(Serialize)]
struct ProbeRow {
    date: String,
    pitcher_name: String,
    /// Kalshi's integer rung. `Over 5.5` is exactly `6+`, so the join needs no
    /// interpolation.
    threshold: u16,
    book_line: f64,
    bookmaker: String,
    over_price_american: f64,
    under_price_american: f64,
    /// Vig removed proportionally across the two sides.
    book_devigged_over: f64,
    book_overround: f64,
    kalshi_yes_bid: Option<u16>,
    kalshi_yes_ask: Option<u16>,
    kalshi_mid: Option<f64>,
    model_over: Option<f64>,
    sim_over: Option<f64>,
    minutes_to_first_pitch: Option<i64>,
    lineup_posted: bool,
}

/// American odds to implied probability, before any vig removal.
fn implied_probability(american: f64) -> Option<f64> {
    if american >= 100.0 {
        Some(100.0 / (american + 100.0))
    } else if american <= -100.0 {
        Some(-american / (-american + 100.0))
    } else {
        // Between -100 and +100 is not a valid American price.
        None
    }
}

/// Removes the book's margin proportionally across a two-sided market.
///
/// A raw -115 is **not** a 53.5% probability: both sides sum to more than one,
/// and that surplus is the book's take. Comparing a raw implied price to a
/// Kalshi mid would manufacture an edge on every single leg, always in the same
/// direction, so this is not a refinement — it is the difference between a
/// measurement and a bug.
fn devig(over_american: f64, under_american: f64) -> Option<(f64, f64)> {
    let over = implied_probability(over_american)?;
    let under = implied_probability(under_american)?;
    let overround = over + under;
    if overround <= 0.0 {
        return None;
    }
    Some((over / overround, overround))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let args = parse_args(&root)?;

    let api_key = std::env::var("ODDS_API_KEY").map_err(|_| {
        "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com (500 credits a \
         month, no card), then set it for this shell:\n  $env:ODDS_API_KEY = \"<key>\""
    })?;

    let http = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .user_agent("pitcher-next-start/0.1")
        .build()?;

    // Model and simulator forecasts for the slate, from the shipped path so the
    // probe never re-implements scoring.
    let slate = pns_core::score_date_detailed(&args.date.to_string())?;
    let by_name: HashMap<String, &DecisionCard> = slate
        .cards
        .iter()
        .map(|card| (normalize_name(&card.pitcher_name), card))
        .collect();
    eprintln!("{} cards scored for {}", slate.cards.len(), args.date);

    // Kalshi's live board for the same slate.
    let kalshi_client = KalshiClient::new()?;
    let markets = kalshi_client.strikeout_markets()?;
    let mut kalshi_by_key: HashMap<(String, u16), &kalshi::KalshiMarket> = HashMap::new();
    for market in &markets {
        let Some((event_date, threshold)) = kalshi::parse_ticker(&market.ticker) else {
            continue;
        };
        if event_date != args.date {
            continue;
        }
        let Some(name) = kalshi::market_pitcher_name(market) else {
            continue;
        };
        kalshi_by_key.insert((normalize_name(name), threshold), market);
    }
    eprintln!("{} Kalshi rungs listed for {}", kalshi_by_key.len(), args.date);

    // The events list is one credit and bounds everything that follows.
    let (events, mut remaining) = {
        let response = http
            .get(format!("{ODDS_BASE}/sports/{SPORT}/events"))
            .query(&[("apiKey", api_key.as_str())])
            .send()?;
        let remaining = response
            .headers()
            .get("x-requests-remaining")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok())
            .map(|value| value as u32);
        let status = response.status();
        let body = response.text()?;
        if !status.is_success() {
            return Err(format!("odds events request failed ({status}): {body}").into());
        }
        (serde_json::from_str::<Vec<OddsEvent>>(&body)?, remaining)
    };
    eprintln!(
        "{} MLB events listed; credits remaining {}",
        events.len(),
        remaining.map_or("unknown".into(), |value| value.to_string())
    );

    let today = args.date.to_string();
    let dated: Vec<&OddsEvent> = events
        .iter()
        .filter(|event| event.commence_time.starts_with(&today))
        .collect();
    // **Once a game starts the market is answering a different question.** It can
    // see how many strikeouts the pitcher already has; the model's forecast is
    // still the pregame one. Comparing the two inflates the model's apparent
    // error badly — measured at 15.64pp with in-play rows against 9.28pp
    // without — so those events never enter the sample.
    let mut underway = 0usize;
    let mut untimed = 0usize;
    let todays_events: Vec<&OddsEvent> = dated
        .into_iter()
        .filter(|event| match minutes_until(&event.commence_time) {
            Some(minutes) if minutes >= 0 => true,
            Some(_) => {
                underway += 1;
                false
            }
            None => {
                untimed += 1;
                false
            }
        })
        .take(args.max_events)
        .collect();
    eprintln!(
        "{} events on {} still pregame ({underway} already underway, {untimed} with an \
         unreadable start time — both skipped)",
        todays_events.len(),
        args.date
    );

    let mut rows: Vec<ProbeRow> = Vec::new();
    for event in &todays_events {
        if let Some(left) = remaining
            && left <= args.min_credits
        {
            eprintln!(
                "stopping: {left} credits remain, at or below the {} floor",
                args.min_credits
            );
            break;
        }
        let response = http
            .get(format!("{ODDS_BASE}/sports/{SPORT}/events/{}/odds", event.id))
            .query(&[
                ("apiKey", api_key.as_str()),
                ("regions", "us"),
                ("markets", PROP_MARKET),
                ("oddsFormat", "american"),
            ])
            .send()?;
        if let Some(value) = response
            .headers()
            .get("x-requests-remaining")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<f64>().ok())
        {
            remaining = Some(value as u32);
        }
        let status = response.status();
        let body = response.text()?;
        if !status.is_success() {
            // A single event without props is normal; a hard failure is not.
            eprintln!(
                "  {} @ {}: request failed ({status})",
                event.away_team, event.home_team
            );
            continue;
        }
        let odds: EventOdds = serde_json::from_str(&body)?;
        let mut added = 0usize;
        for bookmaker in &odds.bookmakers {
            for market in &bookmaker.markets {
                if market.key != PROP_MARKET {
                    continue;
                }
                // Pair Over and Under by (player, line) before devigging: a
                // one-sided quote carries no removable margin.
                let mut sides: HashMap<(String, String), (Option<f64>, Option<f64>)> =
                    HashMap::new();
                for outcome in &market.outcomes {
                    let (Some(player), Some(price), Some(point)) = (
                        outcome.description.as_deref(),
                        outcome.price,
                        outcome.point,
                    ) else {
                        continue;
                    };
                    let key = (player.to_string(), format!("{point:.1}"));
                    let entry = sides.entry(key).or_insert((None, None));
                    if outcome.name.eq_ignore_ascii_case("over") {
                        entry.0 = Some(price);
                    } else if outcome.name.eq_ignore_ascii_case("under") {
                        entry.1 = Some(price);
                    }
                }
                for ((player, point_text), (over, under)) in sides {
                    let (Some(over_price), Some(under_price)) = (over, under) else {
                        continue;
                    };
                    let Ok(point) = point_text.parse::<f64>() else {
                        continue;
                    };
                    // Only half-point lines map to a Kalshi rung without a push.
                    if (point.fract() - 0.5).abs() > 1e-9 {
                        continue;
                    }
                    let threshold = point.ceil() as u16;
                    let Some((devigged, overround)) = devig(over_price, under_price) else {
                        continue;
                    };
                    let normalized = normalize_name(&player);
                    let card = by_name.get(&normalized);
                    let market_quote = kalshi_by_key.get(&(normalized.clone(), threshold));
                    let mid = market_quote.and_then(|quote| {
                        quote
                            .yes_bid
                            .zip(quote.yes_ask)
                            .map(|(bid, ask)| (f64::from(bid) + f64::from(ask)) / 200.0)
                    });
                    rows.push(ProbeRow {
                        date: today.clone(),
                        pitcher_name: player.clone(),
                        threshold,
                        book_line: point,
                        bookmaker: bookmaker.key.clone(),
                        over_price_american: over_price,
                        under_price_american: under_price,
                        book_devigged_over: devigged,
                        book_overround: overround,
                        kalshi_yes_bid: market_quote.and_then(|quote| quote.yes_bid),
                        kalshi_yes_ask: market_quote.and_then(|quote| quote.yes_ask),
                        kalshi_mid: mid,
                        model_over: card.and_then(|card| {
                            pmf_over_probability(&card.strikeout_probabilities, threshold)
                        }),
                        sim_over: card.and_then(|card| {
                            card.sim_strikeout_probabilities
                                .as_deref()
                                .and_then(|pmf| pmf_over_probability(pmf, threshold))
                        }),
                        minutes_to_first_pitch: minutes_until(&event.commence_time),
                        lineup_posted: card
                            .is_some_and(|card| card.sim_strikeout_probabilities.is_some()),
                    });
                    added += 1;
                }
            }
        }
        eprintln!(
            "  {} @ {}: {added} quotes ({} credits left)",
            event.away_team,
            event.home_team,
            remaining.map_or("?".into(), |value| value.to_string())
        );
    }

    if rows.is_empty() {
        return Err("no book quotes collected; nothing to measure".into());
    }
    summarise(&rows);

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Append-only: the probe is meant to accumulate across slates, and a run
    // that overwrote yesterday would destroy the sample it is building.
    let exists = args.out.exists();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(!exists)
        .from_writer(file);
    for row in &rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    eprintln!("appended {} rows to {}", rows.len(), args.out.display());
    Ok(())
}

/// Every book's quote on one rung, collapsed toward a consensus.
///
/// The forecasts and the Kalshi mid are properties of the rung rather than of
/// any one book, so they are carried from whichever quote arrived first instead
/// of being averaged.
struct Consensus {
    devigged_sum: f64,
    quotes: usize,
    kalshi_mid: Option<f64>,
    model_over: Option<f64>,
    sim_over: Option<f64>,
}

/// The measurement the whole probe exists for.
fn summarise(rows: &[ProbeRow]) {
    let rms = |values: &[f64]| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
    };

    // Filtered again here, not only at collection: the CSV appends across runs,
    // and rows written before this guard existed would otherwise poison every
    // later summary.
    let pregame: Vec<&ProbeRow> = rows
        .iter()
        .filter(|row| is_pregame(row.minutes_to_first_pitch))
        .collect();
    let dropped = rows.len() - pregame.len();
    if dropped > 0 {
        println!("excluded {dropped} in-play quotes; a started game prices a different question");
    }
    if pregame.is_empty() {
        println!("no pregame quotes to measure");
        return;
    }

    // Consensus across books for each (pitcher, threshold), then the gap to the
    // Kalshi mid on the same rung.
    let mut consensus: HashMap<(String, u16), Consensus> = HashMap::new();
    for row in &pregame {
        let entry = consensus
            .entry((row.pitcher_name.clone(), row.threshold))
            .or_insert_with(|| Consensus {
                devigged_sum: 0.0,
                quotes: 0,
                kalshi_mid: row.kalshi_mid,
                model_over: row.model_over,
                sim_over: row.sim_over,
            });
        entry.devigged_sum += row.book_devigged_over;
        entry.quotes += 1;
    }

    let mut book_gaps = Vec::new();
    let mut model_gaps = Vec::new();
    let mut sim_gaps = Vec::new();
    for entry in consensus.values() {
        let Some(mid) = entry.kalshi_mid else { continue };
        let book = entry.devigged_sum / entry.quotes.max(1) as f64;
        book_gaps.push((book - mid) * 100.0);
        if let Some(model) = entry.model_over {
            model_gaps.push((model - mid) * 100.0);
        }
        if let Some(sim) = entry.sim_over {
            sim_gaps.push((sim - mid) * 100.0);
        }
    }

    let books: std::collections::BTreeSet<&str> =
        pregame.iter().map(|row| row.bookmaker.as_str()).collect();
    let mean_overround =
        pregame.iter().map(|row| row.book_overround).sum::<f64>() / pregame.len() as f64;

    println!("\n=== book vs Kalshi, {} pregame quotes ===", pregame.len());
    println!(
        "books: {} | mean overround {:.4} (~{:.2}pp cost per side)",
        books.into_iter().collect::<Vec<_>>().join(", "),
        mean_overround,
        (mean_overround - 1.0) * 100.0 / 2.0
    );
    println!("rungs matched to a Kalshi mid: {}", book_gaps.len());
    if book_gaps.is_empty() {
        println!("no overlapping rungs — cannot measure divergence from this slate");
        return;
    }
    println!("\n{:<28} {:>10}", "RMS gap to Kalshi mid", "pp");
    println!("{:<28} {:>10.2}", "devigged book consensus", rms(&book_gaps));
    if !model_gaps.is_empty() {
        println!("{:<28} {:>10.2}", "model", rms(&model_gaps));
    }
    if !sim_gaps.is_empty() {
        println!("{:<28} {:>10.2}", "simulator", rms(&sim_gaps));
    }

    let book_rms = rms(&book_gaps);
    println!("\n--- the pre-registered rule ---");
    if book_rms <= 2.0 {
        println!(
            "books track Kalshi ({book_rms:.2}pp <= 2.00pp). They carry the same information, \
             the model loses to both, and NOTHING SHOULD BE BOUGHT."
        );
    } else {
        println!(
            "books diverge from Kalshi ({book_rms:.2}pp > 2.00pp). Re-measure on the BEST \
             number across books rather than the consensus before considering the paid month."
        );
    }
    println!(
        "\nNote this is one slate. Accumulate several before acting — the CSV appends for \
         exactly that reason."
    );
}

/// Whether a quote is usable: its game had not started when it was captured.
///
/// An unreadable start time is treated as **not** usable. The cost of dropping a
/// good quote is one row; the cost of keeping an in-play one is a corrupted
/// measurement, and the first slate run showed that is not hypothetical.
fn is_pregame(minutes_to_first_pitch: Option<i64>) -> bool {
    minutes_to_first_pitch.is_some_and(|minutes| minutes >= 0)
}

fn minutes_until(commence_time: &str) -> Option<i64> {
    let start = chrono::DateTime::parse_from_rfc3339(commence_time).ok()?;
    Some((start.with_timezone(&chrono::Utc) - chrono::Utc::now()).num_minutes())
}

fn parse_args(root: &std::path::Path) -> Result<Args, Box<dyn std::error::Error>> {
    let mut args = Args {
        date: chrono::Local::now().date_naive(),
        out: root.join("data/processed/book_probe.csv"),
        min_credits: 50,
        max_events: 20,
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
            "--date" => args.date = NaiveDate::parse_from_str(&value()?, "%Y-%m-%d")?,
            "--out" => args.out = PathBuf::from(value()?),
            "--min-credits" => args.min_credits = value()?.parse()?,
            "--max-events" => args.max_events = value()?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    /// A raw -115/-115 pair is 53.49% on each side and sums to 1.0698. Devigging
    /// must return exactly 0.5, not 0.5349 — comparing the raw number to a
    /// Kalshi mid would invent a 3.5pp edge on every leg.
    #[test]
    fn devig_removes_the_margin_from_a_balanced_pair() {
        let (over, overround) = super::devig(-115.0, -115.0).unwrap();
        assert!((over - 0.5).abs() < 1e-12, "got {over}");
        assert!((overround - 1.0697674418604652).abs() < 1e-12, "got {overround}");
    }

    /// A favourite/underdog pair keeps its lean after the margin comes out, and
    /// the two devigged sides must sum to exactly one.
    #[test]
    fn devig_preserves_the_lean_and_sums_to_one() {
        let (over, _) = super::devig(-150.0, 130.0).unwrap();
        let (under, _) = super::devig(130.0, -150.0).unwrap();
        assert!(over > 0.5, "the favourite must stay favoured, got {over}");
        assert!((over + under - 1.0).abs() < 1e-12);
    }

    /// A game already underway must never enter the sample. On the first real
    /// slate, 8 of 96 quotes came from started games and dragged the model's
    /// measured error from 9.28pp to 15.64pp — the market could see strikeouts
    /// that had already happened while the forecast could not.
    #[test]
    fn in_play_quotes_are_excluded_and_unknown_times_are_treated_as_in_play() {
        assert!(super::is_pregame(Some(180)), "a game three hours out is usable");
        assert!(super::is_pregame(Some(0)), "first pitch exactly now is still pregame");
        assert!(!super::is_pregame(Some(-1)), "one minute in is already in play");
        assert!(!super::is_pregame(Some(-54)), "the Jack Perkins case");
        assert!(
            !super::is_pregame(None),
            "an unreadable start time must not be assumed pregame"
        );
    }

    #[test]
    fn american_prices_convert_and_invalid_ones_are_refused() {
        assert!((super::implied_probability(100.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((super::implied_probability(-100.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((super::implied_probability(-200.0).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        // Nothing legitimately prices between -100 and +100.
        assert!(super::implied_probability(50.0).is_none());
        assert!(super::implied_probability(0.0).is_none());
    }
}
