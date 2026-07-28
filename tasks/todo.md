# Paper trading against Kalshi strikeout markets

## Goal

Grade the v2 strikeout distribution against live market prices without wagering: log every matched Kalshi pitcher-strikeout market with the model's probability, open simulated one-contract positions when the edge is at least 5 points, and settle against completed box scores.

## Implementation

- [x] `data/kalshi.rs`: public no-auth client for the `KXMLBKS` series with cursor pagination and bounded retries; ticker date/threshold parser; subtitle name extraction.
- [x] `paper.rs`: name normalization with diacritic folding, market↔card matching by pitcher and slate date, edge-based side selection (YES at ask / NO at 100−bid, spread- and quote-sanity gated), first-record-wins CSV log under the app-data dir, grading via completed MLB game logs, and a running summary (P&L, ROI, record, calibration of logged probabilities).
- [x] Tauri commands `paper_trading_update` (score slate → fetch markets → record → grade) and `paper_trading_status` (read-only).
- [x] Svelte "Paper trading" page: summary cards plus a daily log table using the existing validation styling; additive TypeScript types.

## Verification

- [x] `cargo fmt` / strict clippy / `cargo test --workspace` (40 tests, 7 new), `pnpm check` (0 errors), `pnpm test -- --run`, `pnpm build`.
- [x] Live Kalshi API verified: per-pitcher markets exist at thresholds 4+ through 11+, matching the model's discrete distribution exactly.

## Review

- Markets without two-sided sane quotes are logged for calibration but never bet; thin books are expected on Kalshi props.
- Scratched starters stay ungraded rather than guessing settlement; a future pass could void entries older than a few days.
- The 5-point edge threshold and 30-cent max spread are inspectable constants in `paper.rs`.

# Model v2: learned noise and a discrete strikeout distribution

## Goal

Make the model price-capable for strikeout props: learn observation noise from data instead of fixing it, and produce a full discrete P(K = k) distribution so P(over line) is a model output rather than a Gaussian approximation.

## Implementation

- [x] Replaced fixed-noise conjugate regression with Normal-Inverse-Gamma updates for all targets; predictive variance is now `(b_n/a_n) * (1 + x'V_n x)` with the old constants demoted to prior means.
- [x] Added per-batter strikeout-rate and batters-faced regressions to the shared feature pipeline (targets already existed in the normalized data).
- [x] Built the compositional strikeout distribution: beta-binomial in the K rate, mixed over 19 empirical batters-faced residual quantiles; hand-rolled Lanczos ln-gamma, no new dependencies.
- [x] Calibrated the beta-binomial concentration by predictive log-likelihood on the earlier 70% of held-out dates; audited the later 30% with log score vs the old Gaussian treatment, both point MAEs, and over-line reliability at 3.5–7.5.
- [x] Bumped the artifact to schema v2 (`models/pns-model-v2.json`, id prefix `pns-blr-v2`), added invariant checks for the new posterior and distribution fields.
- [x] Surfaced `expected_strikeouts_compositional`, the pmf, and P(K over 3.5–8.5) through DecisionCard, the CSV export, and TypeScript types (UI rendering unchanged).

## Verification

- [x] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test --workspace` (33 tests), `pnpm check`, `pnpm test -- --run`, `pnpm build`.
- [x] Two independent training runs produced byte-identical artifacts (SHA-256 `1c3df2b1ddf16f1e9a34f952c391afe8582389cda3f95f3b699d6e5bdd06f550`).
- [x] Regenerated `outputs/` from the frozen v2 artifact.

## Review

- Raw 90% runs interval coverage moved from 0.805 (fixed noise, too narrow) to 0.923; the empirical rescale now nudges rather than rescues (recommended scale 0.87 vs 1.14).
- Point MAEs unchanged to four decimals — learned noise affects widths, not means, exactly as expected at n = 16,477.
- Strikeout distribution audit (754 later-date rows with no say in calibration): log score −2.2225 vs −2.2508 for the Gaussian baseline; selected concentration 75. Over-line reliability at 6.5: predicted 27.1% vs observed 26.9%; at 7.5: 16.74% vs 16.71%. Lines 3.5/4.5 run ~2–3 points optimistic — left-tail (early hook) mass is still slightly light; candidate follow-up is a hook-aware batters-faced mixture.
- Compositional MAE 1.8584 vs direct 1.8594 on the same audit rows; the card keeps the direct estimate for continuity, with the compositional mean exported alongside.

# Versioned model artifacts

## Smoke-test follow-up: explanation and 2026 freshness

- [x] Rework the player page so the favorable case, risk case, and overall model read are explicit above the technical details.
- [x] Replace the dashboard's artifact-ID-first label with a human-readable trained-through/freshness message; keep the ID as secondary provenance.
- [x] Update the canonical normalized dataset through the latest completed 2026 games using the existing baseballr-compatible Rust path.
- [x] Retrain and validate a new artifact, then regenerate outputs and update displayed provenance.
- [x] Run the full review and verification suite before committing.

### Smoke-test follow-up review

- Live probable starters and recent pitcher histories were already current; only the trained artifact was stale.
- The canonical dataset now contains 16,477 starts through 2026-07-16 with 98.44% fastball-velocity coverage. Missing velocity uses the tested neutral imputation.
- The promoted artifact is `pns-blr-v1-2026-07-16-d8861f625b8d`; a second independent training run produced identical bytes.
- Refresh tooling now supports an explicit cutoff and staged output, refuses request failures, enforces a coverage floor, reuses prior velocity enrichment, and reports batch progress.
- The first player-page explanation treatment conflicted with the editorial theme. It was replaced with a compact light full-width model case, and the four-card grid now includes a Model track record panel.

## Goal

Separate leakage-safe model training and validation from daily slate scoring. Training produces one deterministic, versioned artifact containing the four fitted Bayesian models, feature standardization, interval calibration, confidence calibration/audit, and training provenance. Scoring loads that frozen artifact and only builds live next-start features.

## Contract

- Artifact format: human-inspectable JSON at `models/pns-model-v1.json`.
- Schema version: explicit integer checked before any model data is used.
- Feature contract: persist the ordered feature names and reject a mismatch in count, order, or name.
- Training provenance: training row count, first/last target date, normalized-data SHA-256, and validation configuration. Volatile creation timestamps are deliberately excluded so identical training input produces identical bytes.
- Model state: posterior mean/covariance/noise variance for FIP, runs, strikeouts, and innings; feature means/scales; all interval multipliers.
- Validation state: complete walk-forward metrics, confidence thresholds, and later-date audit used by cards and the Validation page.
- Runtime behavior: live scoring fails clearly if the artifact is missing, malformed, unsupported, or incompatible. It must not silently retrain or substitute unvalidated values.
- Reproducibility: the same artifact and normalized scoring inputs must produce the same numeric projections and confidence labels regardless of unrelated slate rows.
- Architecture: serialization and invariant validation live in `pns-core`; Tauri remains a thin wrapper and Svelte only displays artifact provenance.

## Implementation plan

- [x] Add serializable/deserializable model and validation state with invariant checks for finite values, matrix dimensions, standardizer widths, interval scales, confidence thresholds, and audit configuration.
- [x] Add a `ModelArtifactV1` envelope with schema version, feature contract, training provenance, model state, and validation metrics.
- [x] Add deterministic JSON save/load APIs using safe same-directory replacement semantics for writes.
- [x] Add a training entry point and example that reads normalized history, builds leakage-safe rows, validates, fits, and writes `models/pns-model-v1.json`.
- [x] Refactor normalized scoring so it accepts a loaded artifact and does not run validation or fit a model.
- [x] Make live scoring locate the artifact via `PNS_MODEL_ARTIFACT` first, then repository-relative paths, with an embedded release artifact for packaged builds.
- [x] Keep an explicit train-and-score helper only for artifact training/tests; do not use it in the production Tauri command.
- [x] Add artifact provenance to `ScoredSlate`, CSV export, prediction history, TypeScript types, and a compact UI status/readout.
- [x] Add tests for numeric round trips, deterministic generation, unsupported schema, malformed dimensions/values, feature-contract mismatch, missing artifacts, cutoff leakage, and slate-invariant scoring.
- [x] Generate the v1 artifact from the canonical processed dataset.
- [x] Update README, tech spec, and Learn documentation to distinguish training/calibration from daily scoring.
- [x] Regenerate sample predictions and validation output from the frozen artifact.

## Verification

- [x] `cargo fmt --all --check`
- [x] `cargo test --workspace`
- [x] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [x] `pnpm check`
- [x] `pnpm test -- --run`
- [x] `pnpm build`
- [x] Train the final artifact twice and prove byte-identical output; both files hashed to `37501765fbede5e1134b2cb9b070b99d9e40957125d78e32426a5a87e3e702ab`.
- [x] Score the fixture slate from the artifact and compare results across an unrelated-slate perturbation.
- [x] Inspect generated artifact, validation JSON, predictions CSV, and UI provenance manually.

## Pre-commit code review

- [x] Review serialization boundaries and compatibility errors.
- [x] Review leakage and training-cutoff provenance.
- [x] Review production path for accidental retraining or fallback.
- [x] Review Rust/Tauri/Svelte contract changes.
- [x] Refactor any unnecessary duplication or overly broad changes.
- [x] Record findings and fixes below before committing.

## Review

The production path now loads a checked artifact before network work, rejects incompatible overrides, and refuses to score on or before the artifact cutoff. The v1 artifact contains four Bayesian posteriors, the exact 16-feature contract, standardization, calibrated intervals, confidence audit, validation metrics, and normalized-data fingerprint. Artifact IDs combine schema, cutoff, and that fingerprint.

Review findings resolved:

- Added the training-cutoff guard after review found that retrospective dates could otherwise leak later outcomes.
- Added a normalized-data SHA-256 to disambiguate corrected datasets with the same cutoff.
- Added cross-field row-count, validation-range, covariance-shape/symmetry, diagonal, and finite-value checks.
- Made replacement files unique and added a platform-safe replace/rollback path.
- Persisted model artifact ID/cutoff in prediction history and CSV exports.
- Corrected missing fastball velocity imputation after regenerated fixtures exposed that the old zero sentinel became an extreme standardized value.
- Cleared stale card artifacts during regeneration so `outputs/cards/` matches `outputs/predictions.csv`.

Verification passed with 28 Rust tests, strict Clippy, Svelte diagnostics, 4 frontend tests, the Vite production build, deterministic artifact generation, fixture slate invariance, and regenerated two-pitcher sample outputs.
