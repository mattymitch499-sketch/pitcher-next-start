# AGENTS.md — Pitcher Next-Start

## Mission
Build a predictive MLB next-start pitcher model.  
Data contract comes from `baseballr`. Implementation is **Rust core + Tauri v2 + Svelte UI**.  
Ship a working projection card in a desktop app, not architecture theater.

## Canonical docs
- `README.md` — product plan
- `techSpec.md` — Phase 1 build contract
- `references/baseballr_map.md` — data source-of-truth map

Follow those over improvisation.

## Stack (locked)
- **Rust** — data, features, model, card generation (library crate first)
- **Tauri v2** — desktop shell, commands/IPC, packaging
- **Svelte** — frontend UI

Use Adam’s global `AGENTS.md` / `CLAUDE.md` for default Tauri + Svelte project conventions (v2 layout, Svelte patterns, tooling). Do not invent a parallel stack.

## Non-negotiables
1. **Predictive first.** No matchup-board demos as the main deliverable.
2. **baseballr → Rust.** Every data pull maps to a known baseballr surface, then is reimplemented in Rust.
3. **No R runtime dependency in the demo path.** R may be used once to inspect schemas.
4. **One projection, two uses:** player-prop lean and team-model input.
5. **Bayesian from day one.** Phase 1 model is Bayesian linear regression (not frequentist ridge first). Candidate engine: rustmc. No hierarchical / multi-level models in Phase 1.
6. **No leakage.** Time-aware splits only.
7. **Normalized tables only into the model.** No raw JSON shortcuts.
8. **Core before chrome.** Library pipeline first; UI is a thin shell. Ugly UI is fine in Phase 1.
9. **Learn is required.** Phase 1 includes an in-app Learn page teaching the Bayesian model math (mlb-PE pattern). Source content: `docs/learn.md`. Do not ship cards without a path to explain the math.

## Architecture rule
```text
pns-core (Rust lib)  ->  Tauri commands  ->  Svelte UI
```
- Modeling logic lives in the Rust core crate.
- Tauri commands are thin wrappers around core.
- Svelte renders results; it does not own modeling logic.
- Optional thin CLI can wrap the same core later. Not Phase 1 primary path.

## Build order
1. `references/baseballr_map.md`
2. Rust core: data clients + fixtures
3. Feature builder
4. Bayesian linear model train/score (posterior mean + intervals)
5. Card + CSV export from core
6. Tauri commands that call core
7. Minimal Svelte UI: date → run → cards → export + **Learn** route
8. Port `docs/learn.md` into the Learn page (formulas, worked example, limits)
9. Only then polish

## Output contract
Each scored pitcher must include:
- `projected_fip`
- `expected_runs_low` / `expected_runs_base` / `expected_runs_high`
- `confidence` = high | medium | low
- `judgment` = one sentence
- optional `prop_lean`
- optional `team_model_input`

Write:
- `outputs/predictions.csv`
- `outputs/cards/`

## Phase 1 UI surface (minimal)
- date picker
- run / score action
- list of projection cards (≥2 starters)
- export CSV action
- basic error display
- **Learn** page (nav item + math walkthrough from `docs/learn.md`)

Not Phase 1 UI:
- polished design system
- odds shopping
- bankroll tools
- multi-day slate automation dashboard
- charts for their own sake
- Playground (optional later; Learn is enough for v1 explainability)

## Do not
- switch primary stack to Python notebooks
- make a CLI the primary product path (thin CLI later is fine)
- put model logic in Svelte / JS
- build a full betting platform
- hide uncertainty behind a vague “AI score”
- expand into full team simulation before next-start projections work
- skip the baseballr map and freestyle the data layer
- overbuild UI before core scores pitchers
- ship projection cards without a Learn path that explains posterior mean, credible bands, and confidence

## Definition of done
A task is done only when:
- code compiles (`cargo` / app build as relevant)
- tests cover the changed path
- docs/map updated if data surfaces changed
- sample projection produced, or a clear blocker is written down
