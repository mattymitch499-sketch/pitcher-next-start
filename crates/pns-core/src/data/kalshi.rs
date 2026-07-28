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
}

/// Statuses that mean a market is no longer actually tradeable. We already
/// ask Kalshi to only send `status=open` markets, but this is a client-side
/// backstop in case that server-side filter and a market's real status ever
/// disagree (e.g. it closed in the moment between the query and this read).
const CLOSED_STATUSES: &[&str] = &["closed", "settled", "finalized", "determined"];

pub fn is_closed_status(status: &str) -> bool {
    CLOSED_STATUSES.contains(&status.to_lowercase().as_str())
}

fn cents_from_dollar_string<'de, D>(deserializer: D) -> std::result::Result<Option<u16>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Option<String> = Option::deserialize(deserializer)?;
    Ok(raw
        .and_then(|value| value.parse::<f64>().ok())
        .map(|dollars| (dollars * 100.0).round() as u16))
}

#[derive(Deserialize)]
struct MarketsResponse {
    #[serde(default)]
    markets: Vec<KalshiMarket>,
    #[serde(default)]
    cursor: Option<String>,
}

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
}

pub(crate) fn parse_markets(json: &str) -> Result<(Vec<KalshiMarket>, Option<String>)> {
    let response: MarketsResponse = serde_json::from_str(json)?;
    Ok((response.markets, response.cursor))
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
