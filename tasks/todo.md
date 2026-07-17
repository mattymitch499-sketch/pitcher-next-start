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
