# baseballr → Rust data contract

Phase 1 uses MLBAM identifiers and public MLB Stats API JSON. Raw responses are
normalized before any feature or model code can consume them.

| baseballr surface | Purpose | Upstream endpoint | Required normalized columns | Rust module | Status |
|---|---|---|---|---|---|
| `mlb_schedule()` / schedule helpers | Games and probable starters for a date | `statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher` | `game_pk`, `game_date`, `home_team_id`, `away_team_id`, probable pitcher ids/names | `data::schedule` | implemented live + fixture tests |
| `mlb_probables()` / `get_probables_mlb()` | Starting-pitcher slate | Same schedule endpoint with `hydrate=probablePitcher` | pitcher id/name, opponent id, home/away, game date | `data::schedule` | implemented live + fixture tests |
| `mlb_stats()` game-log query (baseballr equivalent of pitcher game logs) | Prior MLB starts | `statsapi.mlb.com/api/v1/people/{id}/stats?stats=gameLog&group=pitching&startDate={date}&endDate={date}` | pitcher id/name, game pk/date, teams, IP, pitch count, K, BB, HR, ER, H, BF | `data::pitcher_logs` | implemented live + fixture tests |
| `fg_park()` intent | Park run environment | MLB Stats API all-team home/away hitting splits | home team id, home/away runs per game, relative factor | `data::context` | implemented live via MLB equivalent |
| FanGraphs team batting-split intent | Opponent offense vs pitcher hand | MLB Stats API all-team `statSplits` (`vl`/`vr`) | team id, season, split OPS, league-relative factor | `data::context` | implemented live via MLB equivalent |
| Player handedness lookup | Choose opponent split | MLB Stats API batch people endpoint | pitcher id, pitch-hand code | `data::context` | implemented live |
| `statcast_search_pitchers()` velocity subset | Four-seam/sinker velocity by start date | Baseball Savant Statcast CSV search | pitcher id, game date, pitch type, release speed | `data::enrich` | implemented live + processed enrichment |
| `mlb_batting_orders()` intent, slate side | Today's posted lineup cards | `statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher,lineups` | `game_pk`, `game_date`, team ids, ordered starting batter ids (absent until posted), probable pitcher ids | `data::lineups` | implemented live + fixture tests |
| `mlb_batting_orders()` / box-score player stats, history side | Completed-game batting lines and starters | `statsapi.mlb.com/api/v1/game/{game_pk}/boxscore?hydrate=person` | team id, starting nine in slot order, batter K and PA, starting pitcher id and `pitchHand` code | `data::lineups` | implemented live + fixture tests |
| no baseballr analogue | Game index over a date range: which games exist, when they start, whether they are final | `statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={a}&endDate={b}` | `game_pk`, official `game_date`, `gameDate` first-pitch timestamp, final/not | `data::schedule::normalize_game_index` | implemented live + fixture test |
| no baseballr analogue | Settled Kalshi strikeout markets and their historical order books | `api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMLBKS&status=settled` and `/series/KXMLBKS/markets/{ticker}/candlesticks` | ticker, `yes_sub_title`, settlement `result`, `close_time`; per-candle `yes_bid`/`yes_ask` closes | `data::kalshi` | implemented live + fixture tests |

## Runtime and normalization rules

- Tests require no network or R runtime and run from committed MLB-shaped fixtures.
- Live clients may cache raw JSON under `data/raw/`, but model code accepts only normalized Rust types.
- Source-shape changes must fail explicitly rather than silently producing partial rows.
- Offline fixture tests begin context at neutral (`1.0`); processed history is enriched chronologically from prior dates only.
- Batter strikeout rates (`data::batters`) are accumulated from box-score lines strictly before the date being served, and shrunk toward the league rate against the same pitching hand.
- Kalshi quotes prices as dollar strings (`"0.4000"`); `data::kalshi` converts to whole cents on the way in and decodes both shapes so its own cache round-trip cannot corrupt a price.
- **Kalshi price history has a retention limit.** Measured 2026-07-30, settled-market pagination reached back only to 2026-05-23. The `/events` feed still lists March events, but their markets return empty, so older order books are unrecoverable. Anything joining historical prices must report the earliest date it actually found rather than assuming the requested window was covered.

## Processed datasets

| File | Built by | Contents |
|---|---|---|
| `data/processed/training_starts.csv` | `examples/build_dataset.rs`, then `examples/enrich_dataset.rs` | One row per start; the frozen input of the shipped v2 model. No lineup column. |
| `data/processed/training_starts_v3.csv` | `examples/build_lineup_dataset.rs` | The same starts plus `opponent_batters`: the opposing starting nine in slot order, `\|`-separated in one column. An empty field means the lineup is unknown, never "no batters". |
| `data/processed/batter_lines.csv` | `examples/build_lineup_dataset.rs` | One row per starting batter per game — `game_date`, `game_pk`, `batter_id`, `strikeouts`, `plate_appearances`, `opposing_hand` — deduped on (`game_pk`, `batter_id`). Read/written by `data::batters`. |

Readers of `training_starts.csv` accept both shapes: the lineup column defaults to
empty when absent, so `data::asof` keeps seeding from the frozen v2 file.
