# Calibrate confidence labels

- [x] Define a leakage-safe confidence score from normalized epistemic run uncertainty and recent-start sample strength.
- [x] Learn candidate high/medium cutoffs from earlier walk-forward predictions instead of hard-coded run widths.
- [x] Audit candidate tiers on later dates and withhold them when accuracy does not separate monotonically.
- [x] Carry calibration and audit metadata through the production scoring pipeline.
- [x] Add focused Rust tests for cutoff calculation, zero-boundary behavior, training-size normalization, validation gating, and card fallback.
- [x] Update Learn and validation UI language with the exact method and current withheld status.
- [x] Regenerate validation and sample projection outputs.
- [x] Run formatting, Rust tests, frontend checks/tests, and production build.
- [x] Review the final diff for correctness, leakage, simplicity, and unnecessary changes; refactor findings.

## Review

The original fixed thresholds were removed. Total predictive variance remains the source of the displayed run interval, while confidence candidates use epistemic uncertainty normalized for model training size and penalized for histories shorter than five starts. The first 70% of walk-forward dates trains candidate percentile boundaries; the final 30% audits them.

Current audit separates 820 later-date five-start rows from 20 short-history rows. Candidate high/medium/low run MAE is 1.666/1.507/1.624, so the candidate tiers do not validate. Short-history run MAE is 1.830. The gate therefore withholds high confidence; complete five-start histories inside the candidate-medium uncertainty boundary are medium, while unusually uncertain or shorter histories are low.

Verification passed: 21 Rust tests, strict Clippy, Svelte diagnostics, 4 frontend tests, Vite production build, Rust formatting, and diff whitespace checks. The historical 2025-07-12 slate and validation artifacts were regenerated successfully with 18 medium and 10 low cards.

## Pre-commit code review

- [x] Review Rust calibration, gating, and fallback logic for correctness and edge cases.
- [x] Review Rust/Svelte serialization and UI states.
- [x] Review generated artifacts, documentation, and task completeness.
- [x] Run static analysis and inspect the complete diff.
- [x] Record findings and recommended fixes before committing.

### Resolved findings

- Conservative fallback now sends extreme/non-finite uncertainty to low instead of making every five-start row medium.
- High and medium now require exactly five starts, matching the calibrated population.
- Five-start candidate tiers and short histories are audited separately.
- Learn-page metrics match the regenerated validation report.
- Non-finite scores and invalid/nonnegative threshold ordering are rejected defensively.
- Pitcher Detail handles validated, withheld, and unavailable states; frontend tests cover withheld/null states.
- README and tech spec describe the normalized epistemic/audit design.
- Strict Clippy is clean across the workspace.

Deferred follow-up: persist confidence calibration as a versioned model artifact instead of recomputing it with each slate's assembled corpus. This is an architectural improvement, not a correctness blocker for the current pipeline.

### Review fixes

- [x] Make fallback classification conservative for extreme/non-finite uncertainty.
- [x] Enforce minimum history eligibility for high and medium confidence.
- [x] Audit candidate five-start tiers against a matching five-start population and report short-history performance separately.
- [x] Add calibration invariant checks and Rust regression tests.
- [x] Correct UI states, stale Learn metrics, and canonical documentation.
- [x] Add frontend coverage for withheld, unavailable, and null-audit states.
- [x] Clear strict Clippy findings.
- [x] Regenerate outputs and rerun the complete verification suite.
- [x] Perform final post-fix review.
