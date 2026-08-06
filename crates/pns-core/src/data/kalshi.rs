use crate::{PnsError, Result};
use chrono::NaiveDate;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration as StdDuration;

const BASE_URL: &str = "https://api.elections.kalshi.com/trade-api/v2";
/// Kalshi's per-pitcher strikeout series: one market per pitcher per
/// threshold, titled "Name: N+ strikeouts?". YES settles when the starter
/// records at least N strikeouts.
pub const STRIKEOUT_SERIES: &str = "KXMLBKS";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiMarket {
    pub ticker: String,
    #[serde(default)]
    pub yes_sub_title: String,
    /// Kalshi quotes prices as dollar strings (e.g. `"0.0900"` = 9 cents), not
    /// the whole-cent integers the rest of this app works with.
    #[serde(default, rename = "yes_bid_dollars", deserialize_with = "cents_from_dollar_string")]
    pub yes_bid: Option<u16>,
    #[serde(default, rename = "yes_ask_dollars", deserialize_with = "cents_from_dollar_string")]
    pub yes_ask: Option<u16>,
    #[serde(default)]
    pub status: String,
    /// `"yes"` / `"no"` once settled, absent or empty before. The market
    /// benchmark uses this to cross-validate its pitcher join: the exchange's
    /// own settlement must agree with the box score's strikeout total.
    #[serde(default)]
    pub result: Option<String>,
    /// RFC-3339 settlement-side close, e.g. `"2026-06-01T02:10:37Z"`. Note this
    /// lands AFTER the game ends — window filtering should key off the ticker's
    /// event date; this field only bounds pagination.
    #[serde(default)]
    pub close_time: Option<String>,
}

/// Statuses that mean a market is no longer actually tradeable. We already
/// ask Kalshi to only send `status=open` markets, but this is a client-side
/// backstop in case that server-side filter and a market's real status ever
/// disagree (e.g. it closed in the moment between the query and this read).
const CLOSED_STATUSES: &[&str] = &["closed", "settled", "finalized", "determined"];

pub fn is_closed_status(status: &str) -> bool {
    CLOSED_STATUSES.contains(&status.to_lowercase().as_str())
}

/// Kalshi sends prices as dollar STRINGS (`"0.4000"` = 40 cents). Our own cache
/// round-trip re-serializes the parsed field as a cents NUMBER. The type tells
/// the two apart, so both decode correctly.
#[derive(Deserialize)]
#[serde(untagged)]
enum DollarTextOrCents {
    Text(String),
    Cents(f64),
}

fn cents_from_dollar_string<'de, D>(deserializer: D) -> std::result::Result<Option<u16>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Option<DollarTextOrCents> = Option::deserialize(deserializer)?;
    Ok(raw.and_then(|value| match value {
        DollarTextOrCents::Text(text) => text
            .parse::<f64>()
            .ok()
            .map(|dollars| (dollars * 100.0).round() as u16),
        DollarTextOrCents::Cents(cents) => Some(cents.round() as u16),
    }))
}

#[derive(Deserialize)]
struct MarketsResponse {
    #[serde(default)]
    markets: Vec<KalshiMarket>,
    #[serde(default)]
    cursor: Option<String>,
}

#[derive(Clone)]
pub struct KalshiClient {
    http: Client,
    base_url: String,
}

impl KalshiClient {
    pub fn new() -> Result<Self> {
        let http = Client::builder()
            .timeout(StdDuration::from_secs(20))
            .user_agent("pitcher-next-start/0.1")
            .build()?;
        Ok(Self {
            http,
            base_url: BASE_URL.into(),
        })
    }

    /// All open strikeout markets. Market data is public; no key is required.
    pub fn strikeout_markets(&self) -> Result<Vec<KalshiMarket>> {
        let mut markets = Vec::new();
        let mut cursor: Option<String> = None;
        for _page in 0..20 {
            let mut request = self.http.get(format!("{}/markets", self.base_url)).query(&[
                ("series_ticker", STRIKEOUT_SERIES),
                ("status", "open"),
                ("limit", "1000"),
            ]);
            if let Some(value) = &cursor {
                request = request.query(&[("cursor", value.as_str())]);
            }
            let text = request_with_retry(request)?;
            let page = parse_markets(&text)?;
            let page_empty = page.0.is_empty();
            markets.extend(page.0);
            match page.1 {
                Some(next) if !next.is_empty() && !page_empty => cursor = Some(next),
                _ => break,
            }
        }
        Ok(markets)
    }

    /// Every settled strikeout market whose ticker event date is on or after
    /// `earliest_event_date`. Kalshi serves settled markets newest first, so
    /// pagination walks back in time and stops one day past the window (the
    /// settlement-side `close_time` lands after the game, hence the slack).
    ///
    /// The assembled result is cached for the calendar day: settled history is
    /// immutable, but the newest page shifts as tonight's games settle.
    pub fn settled_strikeout_markets(
        &self,
        earliest_event_date: NaiveDate,
    ) -> Result<Vec<KalshiMarket>> {
        let today = chrono::Utc::now().date_naive();
        let cache_key = format!("kalshi-settled-{STRIKEOUT_SERIES}-{earliest_event_date}-{today}");
        let cache_file = super::cache_path(&cache_key);
        if super::cache_is_fresh(&cache_file, 2_592_000)
            && let Ok(text) = std::fs::read_to_string(&cache_file)
            && let Ok(markets) = serde_json::from_str::<Vec<KalshiMarket>>(&text)
        {
            return Ok(markets);
        }

        let stop_before = earliest_event_date - chrono::Duration::days(1);
        let mut markets = Vec::new();
        let mut cursor: Option<String> = None;
        for _page in 0..60 {
            let mut request = self.http.get(format!("{}/markets", self.base_url)).query(&[
                ("series_ticker", STRIKEOUT_SERIES),
                ("status", "settled"),
                ("limit", "1000"),
            ]);
            if let Some(value) = &cursor {
                request = request.query(&[("cursor", value.as_str())]);
            }
            let text = request_with_retry(request)?;
            let (page, next) = parse_markets(&text)?;
            let page_empty = page.is_empty();
            let oldest_event = page
                .iter()
                .filter_map(|market| parse_ticker(&market.ticker))
                .map(|(date, _)| date)
                .min();
            markets.extend(
                page.into_iter().filter(|market| {
                    parse_ticker(&market.ticker)
                        .is_some_and(|(date, _)| date >= earliest_event_date)
                }),
            );
            if oldest_event.is_some_and(|date| date < stop_before) {
                break;
            }
            match next {
                Some(next) if !next.is_empty() && !page_empty => cursor = Some(next),
                _ => break,
            }
        }

        if let Some(parent) = cache_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Ok(text) = serde_json::to_string(&markets) {
            std::fs::write(&cache_file, text).ok();
        }
        Ok(markets)
    }

    /// Hourly candles for one market between two unix timestamps. Settled
    /// history never changes, so responses cache for 30 days.
    pub fn candlesticks(
        &self,
        market_ticker: &str,
        start_ts: i64,
        end_ts: i64,
    ) -> Result<Vec<KalshiCandle>> {
        let cache_key = format!("kalshi-candles-{market_ticker}-{start_ts}-{end_ts}");
        let cache_file = super::cache_path(&cache_key);
        if super::cache_is_fresh(&cache_file, 2_592_000)
            && let Ok(text) = std::fs::read_to_string(&cache_file)
        {
            return parse_candlesticks(&text);
        }
        let text = request_with_retry(
            self.http
                .get(format!(
                    "{}/series/{STRIKEOUT_SERIES}/markets/{market_ticker}/candlesticks",
                    self.base_url
                ))
                .query(&[
                    ("start_ts", start_ts.to_string().as_str()),
                    ("end_ts", end_ts.to_string().as_str()),
                    ("period_interval", "60"),
                ]),
        )?;
        let candles = parse_candlesticks(&text)?;
        if let Some(parent) = cache_file.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&cache_file, &text).ok();
        Ok(candles)
    }
}

pub(crate) fn parse_markets(json: &str) -> Result<(Vec<KalshiMarket>, Option<String>)> {
    let response: MarketsResponse = serde_json::from_str(json)?;
    Ok((response.markets, response.cursor))
}

/// One hourly (or minute) candle of a market's book. Only the bid/ask closes
/// matter to the benchmark: the close of the last candle before first pitch is
/// the "what could you have traded at" snapshot.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct KalshiCandle {
    pub end_period_ts: i64,
    #[serde(default)]
    pub yes_bid: CandleQuote,
    #[serde(default)]
    pub yes_ask: CandleQuote,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CandleQuote {
    #[serde(
        default,
        rename = "close_dollars",
        deserialize_with = "cents_from_dollar_string"
    )]
    pub close: Option<u16>,
}

#[derive(Deserialize)]
struct CandlesticksResponse {
    #[serde(default)]
    candlesticks: Vec<KalshiCandle>,
}

pub(crate) fn parse_candlesticks(json: &str) -> Result<Vec<KalshiCandle>> {
    let response: CandlesticksResponse = serde_json::from_str(json)?;
    Ok(response.candlesticks)
}

/// `KXMLBKS-26JUL201910LADPHI-PHICSANCHEZ61-9` → (2026-07-20, 9). The event
/// segment starts with a two-digit year, three-letter month, and two-digit
/// day; the final segment is the strikeout threshold.
pub fn parse_ticker(ticker: &str) -> Option<(NaiveDate, u16)> {
    let mut parts = ticker.split('-');
    if parts.next()? != STRIKEOUT_SERIES {
        return None;
    }
    let event = parts.next()?;
    parts.next()?;
    let threshold: u16 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || threshold == 0 || threshold > 30 {
        return None;
    }
    let year = 2000 + event.get(0..2)?.parse::<i32>().ok()?;
    let month = match event.get(2..5)? {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    };
    let day = event.get(5..7)?.parse::<u32>().ok()?;
    Some((NaiveDate::from_ymd_opt(year, month, day)?, threshold))
}

/// Pitcher name from "Cristopher Sánchez: 9+" style subtitles.
pub fn market_pitcher_name(market: &KalshiMarket) -> Option<&str> {
    let name = market.yes_sub_title.split(':').next()?.trim();
    (!name.is_empty()).then_some(name)
}

fn request_with_retry(request: reqwest::blocking::RequestBuilder) -> Result<String> {
    let mut last_error = String::new();
    for attempt in 0..3 {
        let candidate = request
            .try_clone()
            .ok_or_else(|| PnsError::Data("Kalshi request could not be cloned for retry".into()))?;
        match candidate
            .send()
            .and_then(|response| response.error_for_status())
            .and_then(|response| response.text())
        {
            Ok(text) => return Ok(text),
            Err(error) => {
                last_error = error.to_string();
                if attempt < 2 {
                    std::thread::sleep(StdDuration::from_millis(250 * (attempt + 1) as u64));
                }
            }
        }
    }
    Err(PnsError::Data(format!(
        "Kalshi request failed after 3 attempts: {last_error}"
    )))
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_markets_with_accents_and_missing_quotes() {
        let json = r#"{"markets":[
            {"ticker":"KXMLBKS-26JUL201910LADPHI-PHICSANCHEZ61-6","yes_sub_title":"Cristopher Sánchez: 6+","yes_bid_dollars":"0.4000","yes_ask_dollars":"0.4600","status":"active"},
            {"ticker":"KXMLBKS-26JUL201910LADPHI-LADESHEEHAN80-7","yes_sub_title":"Emmet Sheehan: 7+","status":"active"}
        ],"cursor":"abc"}"#;
        let (markets, cursor) = super::parse_markets(json).unwrap();
        assert_eq!(markets.len(), 2);
        assert_eq!(cursor.as_deref(), Some("abc"));
        assert_eq!(markets[0].yes_bid, Some(40));
        assert_eq!(markets[1].yes_bid, None);
        assert_eq!(
            super::market_pitcher_name(&markets[0]),
            Some("Cristopher Sánchez")
        );
    }

    #[test]
    fn parses_settled_fields_and_survives_a_cache_round_trip() {
        let json = r#"{"markets":[
            {"ticker":"KXMLBKS-26MAY311920CHCSTL-CHCJWICKS36-5","yes_sub_title":"Jordan Wicks: 5+",
             "yes_bid_dollars":"0.2600","yes_ask_dollars":"0.2900","status":"settled",
             "result":"no","close_time":"2026-06-01T02:10:37Z"}
        ]}"#;
        let (markets, _) = super::parse_markets(json).unwrap();
        assert_eq!(markets[0].result.as_deref(), Some("no"));
        assert_eq!(
            markets[0].close_time.as_deref(),
            Some("2026-06-01T02:10:37Z")
        );
        assert_eq!(markets[0].yes_bid, Some(26));

        // The settled-markets cache stores the PARSED struct, whose prices
        // re-serialize as cents numbers rather than dollar strings.
        let cached = serde_json::to_string(&markets).unwrap();
        let restored: Vec<super::KalshiMarket> = serde_json::from_str(&cached).unwrap();
        assert_eq!(restored[0].yes_bid, Some(26));
        assert_eq!(restored[0].yes_ask, Some(29));
        assert_eq!(restored[0].result.as_deref(), Some("no"));
    }

    #[test]
    fn parses_candlesticks_from_the_real_payload_shape() {
        // Trimmed from a live response for the market above (period_interval=60).
        let json = r#"{"ticker":"KXMLBKS-26MAY311920CHCSTL-CHCJWICKS36-5","candlesticks":[
            {"end_period_ts":1780257600,
             "yes_bid":{"open_dollars":"0.2600","high_dollars":"0.2600","low_dollars":"0.2600","close_dollars":"0.2600"},
             "yes_ask":{"open_dollars":"0.2900","high_dollars":"0.2900","low_dollars":"0.2900","close_dollars":"0.2900"},
             "price":{"previous_dollars":"0.2600"},"volume_fp":"0.00","open_interest_fp":"1.32"},
            {"end_period_ts":1780261200,"yes_bid":{},"yes_ask":{}}
        ]}"#;
        let candles = super::parse_candlesticks(json).unwrap();
        assert_eq!(candles.len(), 2);
        assert_eq!(candles[0].end_period_ts, 1_780_257_600);
        assert_eq!(candles[0].yes_bid.close, Some(26));
        assert_eq!(candles[0].yes_ask.close, Some(29));
        assert_eq!(candles[1].yes_bid.close, None, "an empty book is None, not 0");
    }

    #[test]
    fn parses_ticker_dates_and_thresholds() {
        let (date, threshold) =
            super::parse_ticker("KXMLBKS-26JUL201910LADPHI-PHICSANCHEZ61-9").unwrap();
        assert_eq!(date, chrono::NaiveDate::from_ymd_opt(2026, 7, 20).unwrap());
        assert_eq!(threshold, 9);
        assert!(super::parse_ticker("KXMLBGAME-26JUL201910LADPHI-LAD").is_none());
        assert!(super::parse_ticker("KXMLBKS-26XXX201910LADPHI-P-6").is_none());
    }

    #[test]
    fn only_flags_genuinely_closed_statuses() {
        assert!(super::is_closed_status("closed"));
        assert!(super::is_closed_status("Settled"));
        assert!(!super::is_closed_status("active"));
        assert!(!super::is_closed_status("open"));
        assert!(!super::is_closed_status(""));
    }
}
