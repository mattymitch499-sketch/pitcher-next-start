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

## Runtime and normalization rules

- Tests require no network or R runtime and run from committed MLB-shaped fixtures.
- Live clients may cache raw JSON under `data/raw/`, but model code accepts only normalized Rust types.
- Source-shape changes must fail explicitly rather than silently producing partial rows.
- Offline fixture tests begin context at neutral (`1.0`); processed history is enriched chronologically from prior dates only.
