# Moved

The authoritative data contract is now [`references/baseballr_map.md`](../references/baseballr_map.md), as required by the project architecture. This compatibility pointer remains so older links do not silently break.

<!-- Historical draft below. -->

# baseballr → Rust function map (historical draft)

Phase 1 ports only. Update this file when a `baseballr_rs` public function is added.

| Rust | baseballr reference | Source | Phase | Status |
|---|---|---|---|---|
| `mlb::probables(date)` | `mlb_probables` / `get_probables_mlb` | MLB Stats API | 1 | not started |
| `mlb::schedule(date)` | MLB schedule helpers | MLB Stats API | 1 | not started |
| `fangraphs::pitcher_game_logs(id, season)` | `fg_pitcher_game_logs` | FanGraphs | 1 | not started |
| `fangraphs::park(season)` | `fg_park` | FanGraphs | 1 | not started |
| `fangraphs::team_offense_vs_hand(season, hand)` | FG team splits / leaders | FanGraphs | 1 | not started |
| `savant::recent_avg_fb_velo(id)` | Savant/statcast helpers | Baseball Savant | 1 optional | not started |

## Out of scope for Phase 1

NCAA, Retrosheet bulk loads, Spotrac, ESPN twins, Fox/Bifrost odds, draft, awards, full Statcast pitch-by-pitch warehouse.

## Notes for porters

- Prefer MLBAM ids at the product boundary; record FanGraphs ids when logs require them.
- Cache raw responses under `data/raw/...` with stable keys.
- On HTML/JSON shape drift, raise `ScrapeShapeChanged` and keep fixtures under `data/fixtures/`.
