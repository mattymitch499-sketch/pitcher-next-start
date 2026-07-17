# Pitcher Next-Start — Phase 1 Technical Specification

## How to use this document
This is the build contract for Phase 1.  
Section order follows the project-planner skeleton.  
Mark decisions with `[DECISION]` so they can be challenged deliberately later.

## 1. Scope & Success Criteria

### 1.1 In scope (Phase 1)
- baseballr → Rust data map for required pulls
- Rust clients for schedule + pitcher game logs (minimum)
- normalization into stable tables
- rolling next-start feature builder
- Bayesian linear regression model for next-start projection (from day one; not frequentist-first)
- Rust core library that produces projections + cards + CSV
- Tauri v2 shell with thin commands over core
- minimal Svelte UI:
  - date picker
  - run / score action
  - projection cards for ≥2 starters
  - export CSV
  - **Learn** page (in-app math walkthrough; content from `docs/learn.md`)
- outputs:
  - projected FIP
  - expected runs band
  - confidence
  - judgment
  - optional prop lean / team-model input fields
- fixtures + unit/smoke tests for data and features

### 1.2 Explicitly NOT in Phase 1
- full team win-probability model
- live odds ingestion
- bankroll management
- multi-day slate automation UI
- deep learning
- hierarchical / multi-level Bayesian models
- frequentist ridge as a temporary stand-in
- paid data vendors
- R runtime dependency in demo path
- polished design system / marketing dashboard
- CLI as the primary product path (optional thin CLI later)

### 1.3 Success criteria (the gate)
1. `references/baseballr_map.md` exists and lists every required baseballr surface → Rust module → columns
2. Rust core can fetch/normalize recent starts for at least 2 pitchers without calling R
3. Feature builder produces leakage-safe rolling features
4. Model scores those pitchers and writes `outputs/predictions.csv`
5. Each scored pitcher has:
   - `projected_fip` (posterior mean)
   - `expected_runs_low/base/high` (credible interval + central estimate)
   - `confidence` (normalized epistemic uncertainty, gated by a disjoint later-date accuracy audit)
   - `judgment`
6. Desktop app path reproduces the demo:
   - open app
   - pick date
   - score ≥2 starters
   - show cards
   - export CSV
   - open **Learn** and walk the Bayesian model math
7. `docs/learn.md` exists and is ported into a Svelte Learn route (mlb-PE Learn pattern: TOC + formulas + worked example + limits)
8. Tests cover:
   - normalization of fixture responses
   - rolling feature windows
   - model smoke score

## 2. Architecture Overview

```text
baseballr map (contract)
        |
        v
Rust data clients  -> raw fixtures
        |
        v
normalize tables   -> data/processed
        |
        v
feature builder    -> feature matrix
        |
        v
model train/score  -> projections
        |
        v
card + export      -> outputs/
        |
        v
Tauri commands (thin) -> Svelte UI
```

### Authority rule
- `baseballr_map.md` is authoritative for data intent
- normalized tables are authoritative for model inputs
- generated cards/CSV are artifacts, not source of truth
- Svelte is a view layer only

### Golden invariant
No model code reads raw API JSON directly.  
All modeling reads normalized tables only.  
No modeling logic in frontend JS/Svelte.

### [DECISION] Training and scoring use a versioned artifact boundary
Training and walk-forward validation are explicit offline operations. They write a checked, versioned JSON artifact containing the fitted posteriors, ordered feature contract, standardization, interval calibration, confidence audit, and training provenance. Daily Tauri scoring loads the frozen artifact and never silently retrains or falls back from an invalid override.

## 3. Repository & Module Layout

```text
pitcher-next-start/
  README.md
  techSpec.md
  PROJECT.md
  AGENTS.md
  crates/
    pns-core/
      Cargo.toml
      src/
        lib.rs
        data/
          mod.rs
          schedule.rs
          pitcher_logs.rs
          normalize.rs
          types.rs
        features/
          mod.rs
          rolling.rs
          next_start.rs
        model/
          mod.rs
          train.rs
          score.rs
          calibrate.rs
        card/
          mod.rs
          render.rs
        error.rs
  src-tauri/
    Cargo.toml
    src/
      lib.rs                 # Tauri command surface
      main.rs
  src/                       # Svelte UI (or app/ per global convention)
  data/
    raw/
    processed/
    fixtures/
  outputs/
  references/
    baseballr_map.md
  tests/
    data_normalize_test.rs
    features_rolling_test.rs
    model_smoke_test.rs
```

Exact frontend folder names may follow Adam’s global Tauri + Svelte conventions. Keep the **core vs shell** split.

### [DECISION] Rust core library first, app shell second
**Rationale:** modeling and data correctness matter more than UI chrome. Tauri/Svelte wrap a working core.

### [DECISION] Stack = Rust + Tauri v2 + Svelte
**Rationale:** Adam’s normal desktop product stack. Matches Plume/Launchpad-class work and is better for live product demos than CLI-only.

### [DECISION] No Python-primary path
**Rationale:** project requirement is convert baseballr functions to Rust; long-term path should not be notebook-first.

### [DECISION] Optional CLI later, not Phase 1 primary
**Rationale:** headless batch is useful later. Phase 1 demo path is the desktop app.

## 4. Core Data Model

### Canonical entities
- `PitcherId`
- `GameId`
- `PitcherStartLog` (one row per start)
- `NextStartContext`
- `FeatureRow`
- `Projection`
- `DecisionCard`

### Suggested fields

#### PitcherStartLog
- pitcher_id
- game_pk
- game_date
- team_id
- opponent_id
- home_away
- ip
- k
- bb
- hr
- er
- h
- bf (if available)
- avg_fb_velo (if available)
- gb_pct (if available)

#### NextStartContext
- pitcher_id
- game_date
- home_away
- park_id / park_factor
- opponent_id
- opponent_k_rate
- opponent_woba_or_ops_vs_hand
- rest_days

#### Projection
- pitcher_id
- target_game_date
- projected_fip (posterior mean)
- expected_runs_low / expected_runs_base / expected_runs_high (credible interval + central)
- confidence (from posterior width / sample strength)
- judgment
- prop_lean (optional)
- team_model_input (expected runs allowed)

### [DECISION] Primary labels in v1
Track both:
- next-start FIP
- next-start expected runs allowed

Use FIP as core skill projection and expected runs as bettable/team-model translation. Both come from the Bayesian posterior, not point-estimate hacks.

## 5. On-Disk Format

### Source-of-truth files
- `references/baseballr_map.md`
- fixtures under `data/fixtures/`
- processed parquet/csv under `data/processed/`

### Artifact files
- `outputs/predictions.csv`
- `outputs/cards/*`

### Round-trip stability
- normalized column names are stable
- unknown source fields are preserved in a raw side map or ignored explicitly
- fixtures must be enough to run offline tests

## 6. Data Subsystem (baseballr → Rust)

### Responsibilities
- map baseballr functions to public endpoints
- fetch schedule / pitcher logs
- normalize into `PitcherStartLog`
- write fixtures for offline use

### Phase 0 map columns
For each function:
- baseballr function name
- purpose
- upstream endpoint
- required output columns
- Rust module
- status: planned | implemented | blocked

### Likely first surfaces
Exact names locked in Phase 0 spike, but intent is:
1. schedule for date/team
2. pitcher game logs / starter history
3. optional pbp only if game logs are insufficient
4. opponent/team context joins

### [DECISION] No R at runtime
R may be used offline to inspect package outputs. Runtime demo path is pure Rust.

### Acceptance criteria
- map file complete for v1 functions
- at least one fixture per client
- normalize test green

## 7. Feature Subsystem

### Responsibilities
- compute rolling form over last N starts (default N=5, fallback to available)
- attach next-start context
- emit `FeatureRow`s with as-of dates to prevent leakage

### Core features (v1)
From recent starts:
- k_rate
- bb_rate
- hr_per_9
- whip
- er_per_9 / fip components
- avg_fb_velo + trend if available
- gb_pct if available

Context:
- home/away
- park factor
- opponent offense vs hand
- rest days

### [DECISION] Last-5 default window
**Rationale:** short enough for form, long enough for stability in a demo model.

### Acceptance criteria
- rolling features only use prior starts
- missing fields degrade without crashing
- feature table reproducible from fixtures

## 8. Model Subsystem

### Responsibilities
- fit Bayesian linear regression on historical feature rows
- score next-start projections from the posterior
- produce credible intervals + confidence from posterior uncertainty
- basic calibration / posterior predictive checks

### v1 model (locked intent)
- **Bayesian linear regression from day one**
- weakly informative priors (league-average shrinkage is fine)
- targets: next-start FIP and/or expected runs allowed
- no deep learning
- no hierarchical multi-level structure in Phase 1
- time-aware split (train past → test future)

### Candidate engine
- **[rustmc](https://tbosier.github.io/rustmc/)** — Rust-native Bayesian inference (NUTS / batch path, regression fast path)
- Integration detail may evolve; model family does not: Bayesian linear from the start
- Do not ship a frequentist ridge stand-in and “upgrade later”

### Outputs
- `projected_fip` = posterior mean
- expected runs band = credible interval (+ central estimate)
- `confidence` high/medium/low from training-size-normalized epistemic uncertainty; candidate thresholds are learned on complete five-start histories and must pass a disjoint later-date runs-MAE audit
- if the audit gate fails, withhold high: complete five-start histories within the candidate-medium uncertainty boundary fall back to medium; unusually uncertain or shorter histories remain low
- coefficients remain inspectable for explainability on camera

### [DECISION] Bayesian from the beginning
**Rationale:** the product needs real uncertainty (runs band + confidence). Thin recent-form samples make priors/shrinkage natural. Adam wants Bayesian modeling from day one, not frequentist first.

### [DECISION] Simple Bayesian linear only in Phase 1
**Rationale:** keep the live build shippable. Hierarchical pitcher/team/season models are later.

### Acceptance criteria
- smoke fit/score runs on fixtures
- posterior summaries produce FIP + runs band
- MAE or equivalent reported on held-out starts (using posterior means)
- projections written to CSV

## 9. Decision / Card Subsystem

### Responsibilities
- convert projection into human-readable card
- optional prop lean and team-model input packaging
- provide structured objects the UI can render directly

### Card fields
- pitcher name/id
- projected FIP
- expected runs band
- confidence
- judgment sentence
- optional market lean

### Acceptance criteria
- structured card available from core for each scored pitcher
- markdown/text card can still be written under `outputs/cards/`
- comparison of two pitchers possible in one app run

## 10. App Shell (Tauri + Svelte)

### Responsibilities
- expose thin Tauri commands over core APIs
- render a minimal scoring UI
- trigger export to `outputs/`

### Phase 1 UI surface
- date picker
- run / score action
- list of projection cards
- export CSV action
- basic error / empty-state text
- primary nav includes **Learn**
- Learn page renders the model math walkthrough (see §10.1 / `docs/learn.md`)

### Suggested Tauri commands (names flexible)
- `score_date(date) -> Vec<DecisionCard>`
- `export_predictions(path?) -> path`
- `list_recent_runs()` optional later

Commands should:
- accept/return serializable types
- call `pns-core`
- avoid embedding model logic in the command layer

### [DECISION] Ugly UI is acceptable in Phase 1
**Rationale:** live-build risk is overdesigning the shell before the core scores pitchers.

### Acceptance criteria
- app can score ≥2 starters for a chosen date
- cards visible in UI
- CSV export works
- failures surface as readable errors
- Learn is reachable from primary nav and covers the locked model math

### 10.1 Learn section (required)

**Pattern:** same product role as mlb-PE `/learn` — sticky TOC, formula blocks, worked example, explicit limits. Static content is fine in Phase 1; no playground required.

**Source of truth for copy:** `docs/learn.md`

**Required sections (must ship):**
1. The idea — next-start projection, not matchup recap
2. Why Bayesian — thin samples need real uncertainty
3. The model — Bayesian linear regression form + targets
4. Features — rolling form + next-start context
5. Priors / shrinkage — weakly informative, league-average pull
6. Posterior → card fields — FIP mean, runs credible band, confidence
7. Leakage rule — prior starts only, time-aware validation
8. Worked example — one pitcher → card numbers
9. What it doesn’t do — Phase 1 non-goals in plain language
10. How to use the card — prop lean vs team-model input

**[DECISION] Learn in Phase 1, not polish**
**Rationale:** mlb-PE’s Learn tab is part of product quality. Pitcher Next-Start claims transparent posteriors; Learn is how that claim is cashed. Content page is low risk and camera-friendly.

## 11. Error Model

Typed errors, no silent failures at boundaries:
- `DataFetchError`
- `NormalizeError`
- `FeatureError`
- `ModelError`
- `IoError`

Surfaces:
- core returns typed errors
- Tauri commands map them to user-readable messages
- UI shows actionable text:
  - missing fixture
  - empty start history
  - blocked baseballr map item

## 12. Security Considerations
- public MLB data only in v1
- no secrets required for core path
- do not commit large raw dumps unnecessarily
- no scraping of paid odds sites
- local-first desktop app; no account system in Phase 1

## 13. Testing Strategy

### Layers
1. unit tests for normalize + rolling windows
2. fixture-based integration for fetch/normalize offline
3. model smoke test
4. core “score date” dry-run on fixtures
5. optional later: Tauri command smoke / UI manual checklist

### Definition of done for any feature
- code compiles
- tests green
- map/docs updated if data surface changed
- sample output produced or blocker documented

## 14. Build, Tooling & CI
- Rust stable toolchain
- Tauri v2 toolchain
- Svelte frontend tooling per Adam’s global conventions
- `cargo test` for core
- app build/dev via normal Tauri flow
- optional later: github actions for test on push

Pinned approach:
- keep dependency set small
- HTTP client + serde + csv/parquet as needed
- frontend stays thin

## 15. Implementation Sequence

### M0 — Contract
- write baseballr map
- choose 2 demo pitchers
- **Check:** map reviewed, endpoints identified

### M1 — Data clients (core)
- schedule + pitcher logs
- fixtures
- **Check:** normalize tests green

### M2 — Features (core)
- rolling + next-start context
- **Check:** leakage-safe feature test green

### M3 — Model (core)
- Bayesian linear regression fit/score
- posterior mean + credible intervals
- **Check:** smoke Bayesian projections exist

### M4 — Card + export (core)
- structured cards + `outputs/predictions.csv`
- **Check:** core can score ≥2 pitchers without UI

### M5 — Tauri + Svelte shell
- thin commands
- minimal UI: date → run → cards → export
- Learn route + nav item
- port `docs/learn.md` into Learn page (formulas + example + limits)
- **Check:** Phase 1 success criteria all green, including Learn

Do not advance until the current check is green.  
Do not polish UI before M4 is green.  
Do not treat Learn as optional polish — it is part of M5.

## 16. Extensibility Seams
- data client trait so new baseballr surfaces can be added
- feature builder trait for new contexts
- projection struct already includes team-model input field
- later team model consumes projections without rewriting data layer
- optional thin CLI can call the same core crate later
- richer UI can replace Phase 1 Svelte views without touching model code

## 17. Deferred Decisions
- exact baseballr function names after inventory
- parquet vs csv as primary processed format
- whether K projection is in first public video
- rustmc integration surface details (API wiring, packaging) — model family is not deferred
- exact prior hyperparameters (document defaults in Learn when locked)
- exact Svelte file layout if global conventions already define one
- whether a headless CLI ships in Phase 2+
- hierarchical Bayesian extensions after Phase 1
- Playground-style interactive prior/feature editor (Learn is required; Playground is later)

None of these block starting M0/M1. Learn content draft already exists in `docs/learn.md`.

## 18. Pressure Test Summary

### Why this is good enough to build
- real prediction problem
- clear reusable wedge
- constrained scope
- aligns with public video story
- durable Rust data/model layer under a real desktop product shell
- matches Adam’s current build stack

### Main risks
1. baseballr endpoint mess stalls conversion  
   → mitigated by Phase 0 map + fixtures first
2. thin samples make weak projections  
   → mitigated by simple model + confidence tags
3. scope creep into full team model  
   → mitigated by explicit non-goals
4. UI overbuild stalls the live build  
   → mitigated by core-first sequence and “ugly UI is fine”
5. Model looks like a black box on camera  
   → mitigated by required Learn section (mlb-PE pattern) in Phase 1

### Build recommendation
Proceed.  
Start with `references/baseballr_map.md`, not model code or UI chrome.
