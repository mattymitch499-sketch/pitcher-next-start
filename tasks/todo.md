# Swinging-strike rate shipped: feature schema 2, artifact schema 3

## Goal

The whiff study cleared its bar on full history, so integrate the feature: new
dataset columns, a feature-contract bump, an artifact schema bump, retrained
artifacts, and a live path that can supply the feature daily.

## Parameterisation, decided by measurement rather than assumption

SwStr% and CSW% overlap by construction (CSW = called + swinging), and on the
primary split they visibly traded credit. Three candidates on the same rows:

| features | gain | 95% CI | coefficients |
|---|---|---|---|
| SwStr alone | +0.00235 | [+0.00111, +0.00360] | z 4.64 |
| **SwStr + called** | **+0.00324** | **[+0.00187, +0.00466]** | z 4.96 / 1.96 |
| SwStr + CSW | +0.00324 | [+0.00187, +0.00466] | z 3.67 / 1.96 |

SwStr+called and SwStr+CSW are **numerically identical** — same gain, same
interval to five decimals — because they span the same space. So the choice is
purely conditioning, and the non-overlapping pair wins: cleaner coefficient on
the primary feature, and two quantities that do not contain each other. Two
features also genuinely beat one, so called-strike rate earns its slot.

## Implementation

- [x] `PitcherStartLog` gains `pitches` / `swinging_strikes` / `called_strikes`
      as `#[serde(default)]` columns, so every pre-v4 CSV still reads.
- [x] `examples/build_dataset_v4.rs` joins `pitch_outcomes.csv` onto v3 and
      writes **v4**, never rewriting its input. Refuses below 90% join coverage;
      measured **100.00%** on all 16,477 starts.
- [x] `FEATURE_NAMES` 16 -> 18, `FEATURE_SCHEMA_VERSION` 1 -> 2,
      `MODEL_ARTIFACT_SCHEMA_VERSION` 2 -> 3.
- [x] `features::build` sums counts across the prior window and divides once.
      Below 200 pitches it emits the 0.0 sentinel.
- [x] **Generalised the standardizer's missing-data sentinel.** It hardcoded
      `feature != 7` for fastball velocity; that is now
      `SENTINEL_FEATURE_INDICES`, and a test pins those indices to their feature
      *names* — reordering `FEATURE_NAMES` would otherwise leave the numbers
      valid but pointing at the wrong columns, silently computing means over
      sentinels with nothing failing.
- [x] Artifact IDs now carry the schema version (`pns-blr-v3-...`), since
      `model_artifact_id` is stamped on every paper-trade row and a schema-3
      artifact labelled v2 would make the log's own history unreadable.
- [x] Live path fetches `savant_pitch_outcomes` beside the velocity call, with
      the same `if let Ok(..)` degradation — an outage costs the feature, not the
      slate.

## Verification

- [x] Both artifacts retrained; live artifact **byte-identical across two runs**.
- [x] 121 tests pass, strict clippy clean of new issues, `pnpm check` 0 errors.
- [x] New tests: sentinel indices pinned to names; whiff rates sum counts rather
      than averaging per-start rates (a 200-pitch and a 20-pitch start must not
      weigh equally).

## Review — it works, and it is small

`backtest_sim` is the valid paired comparison: fixed 1,694 starts, fixed 11-rung
ladder, same code both times.

| | v2 / 16 features | v3 / 18 features |
|---|---|---|
| model pooled Brier | 0.11463 | **0.11435** |
| model rung log loss | 0.36145 | **0.36039** |
| model ECE | 0.00904 | **0.00817** |
| model exact log score | -2.23071 | **-2.22848** |
| sim pooled Brier | 0.11333 | **0.11316** |

All four model measures improve; **bar cleared**. Model mean K moved from 5.110
toward the actual 5.007, so the over-prediction shrank rather than shifted. The
gain is 0.00028 Brier — about 5% of the gap to the market, consistent with the
study's predicted magnitude. Real, directionally right, not a transformation.
The sim's edge over the model narrowed (-0.00130 -> -0.00119) because the feature
helps the model directly and the simulator inherits only part of it.

## The market benchmark is not comparable across dates

Ran once as planned, and the run itself produced a methodological finding:
**matched legs moved 6,112 -> 8,284 on the same 1,115 starts**. Kalshi's settled
market list shifts with retention, so two runs on different days score different
leg sets and their absolute Brier values are not comparable. Treat the benchmark
as meaningful only *within* a single run.

Within today's run the ordering is unchanged and decisive: market **0.15577** <
sim **0.16193** < model **0.16379**. The market still beats both.

## Note on promotion

The plan called promotion a separate decision. It is not one: bumping the
artifact schema makes the old v2 artifact unloadable, so shipping schema 3
necessarily ships the new artifact. `lib.rs` loads `models/pns-model-v3.json`.

# Does swinging-strike rate add signal the model is missing?

## Goal

The model predicts strikeouts from *past strikeouts*. Swinging-strike rate
stabilises faster and is the better-known leading indicator, and the model has
never seen it. Decide whether it earns a feature-contract bump, a schema version
and a retrain — **before** paying for any of that.

## What the existing Savant call could not give us

`savant_fastball_velocity` (`data/mod.rs:218`) passes `hfPT=FF|SI|`, so it only
ever sees fastballs and sinkers, and `normalize_savant_velocity` keeps three
columns. A swinging-strike *rate* needs every pitch in the denominator, so this
needed a widened request, not just extra parsing. (An earlier note in project
memory claimed the data was already being downloaded and discarded — it was not.)

## Implementation

- [x] `savant_pitch_outcomes` (`data/mod.rs`): same endpoint, no `hfPT` filter,
      cached 30 days since completed pitch history never changes.
- [x] `PitchOutcomes` + `normalize_savant_pitch_outcomes` (`data/enrich.rs`):
      per-date counts of pitches, swinging strikes and called strikes. Stored as
      **counts, not rates**, so a rolling window sums whole starts and divides
      once — averaging per-start rates would weigh a 10-pitch start like a
      100-pitch one. `foul_tip`, `bunt_foul_tip` and `missed_bunt` count as
      whiffs (the batter swung and missed); ordinary `foul` does not. That is the
      contestable line, so it is written down.
- [x] `examples/fetch_pitch_outcomes.rs`: writes the compact
      `data/processed/pitch_outcomes.csv`, refuses to write a partial download,
      and never touches `training_starts*.csv`.
- [x] `model/whiff.rs`: fits the **real** k-rate estimator twice on identical
      rows differing only in feature width. Lives as a child of `model` so it can
      reuse `fit_target_nig`, the real standardizer and `predict` without
      widening any visibility — a study that re-implemented the regression would
      be measuring its own arithmetic.
- [x] Scoring uses the **exact Student-t predictive density** (the NIG predictive
      is t with `2 * noise_shape` df, and `predict` returns its scale), not a
      Gaussian approximation that would flatter whichever arm is more confident.
- [x] Held-out gain carries a bootstrap interval **clustered by pitcher**, since
      a pitcher's starts share whatever the features get wrong about him.

## Verification

- [x] Parser test pins every `description` value to its bucket and the
      denominator to all pitches.
- [x] Positive control: a genuinely predictive synthetic feature is detected
      (z > 2, held-out likelihood up). Negative control: pure noise is not.
      Without both, a null result elsewhere would mean nothing.
- [x] Mismatched arms are refused rather than silently compared.
- [x] League SwStr came out **0.1183** — the real MLB figure, an independent
      check that the buckets are right.

## Review — first pass: right direction, underpowered

Fetched 2025-03 onward for the 191 pitchers in the evaluation window, which
dropped **9,873 of 15,187 feature rows** for thin history, leaving 3,606 fit and
1,708 eval rows.

| | mean log-lik | MAE | RMSE |
|---|---|---|---|
| baseline | 0.89667 | 0.078514 | 0.098687 |
| + whiff | 0.89897 | 0.078617 | 0.098465 |
| difference | **+0.00230** | +0.000102 | |

95% CI on the gain, clustered by pitcher: **[-0.00178, +0.00616]** — includes
zero.

Both coefficients came out separated from zero, and notably **CSW% carries more
signal per standard deviation (0.0109, z = 3.78) than the model's own
`recent_k_rate` feature (0.0084)**; SwStr% is 0.0086, z = 2.94.

So: the pre-registered bar (likelihood up, a coefficient separated) is cleared,
but the interval — added afterwards, and only ever making the bar harder —
includes zero. **The direction is right; the size is not established.** This is
not a negative result, it is a test too small to answer the question, and the
reason is known: only 1.4 seasons of pitch data for 191 of 430 pitchers.

## Second pass on full history — the bar is cleared

Fetched 2023-03 onward for all 430 dataset pitchers: **27,028 start-rows,
1,707,399 pitches, league SwStr 0.1181** against 0.1183 on the first sample —
unchanged across 2.3x the pitchers, which is the parser-regression check.
Dropped rows fell from **9,873 to 132**.

Two splits, with the high-power one named primary *before* either result existed,
because more data mostly grows the training set while the evaluation window is
fixed:

| split | fit | eval | gain | 95% CI | verdict |
|---|---|---|---|---|---|
| **primary** | <= 2024 | 2025-2026, 6,960 rows | **+0.00324** | **[+0.00187, +0.00466]** | **clears** |
| continuity | < 2026-05-01 | May-Jul 2026, 1,726 rows | +0.00267 | [-0.00017, +0.00545] | grazes zero |

On the primary split held-out log-likelihood, MAE **and** RMSE all improve, and
`recent_swinging_strike_rate` has z = 3.67. The two splits agree on the effect
(+0.0032 vs +0.0027) and disagree only on precision — 6,960 eval rows over ~400
pitchers resolve it, 1,726 over ~191 cannot. That is the power argument playing
out exactly as predicted rather than a conflict between results.

**Collinearity to fix before shipping.** SwStr% and CSW% overlap by construction
(CSW = called + swinging, so it contains SwStr), and they visibly trade credit:
primary gives SwStr z = 3.67 and CSW 1.96, continuity splits it 4.98 / 4.10.
Shipping both would put two features in the contract fighting over one piece of
variance. Test SwStr% + *called-strike* rate (non-overlapping) against SwStr%
alone before fixing the feature contract, since that choice is permanent.

# The model leads the Kalshi price, by about a tenth of what it would need

## Goal

Every earlier comparison asked whether a forecaster beats the price just before
first pitch. Model and simulator both lost, with confidence. Ask the weaker
question instead: when a forecast disagrees with an *early* price, does the price
later move toward it? That is closing-line value, and a lead over a
well-calibrated close converts to expected value at settlement — so it is worth
money even when beating that close is not.

## Why it was possible at all

`backtest_market` requests candles over `SNAPSHOT_LOOKBACK_HOURS = 24` and uses
only the last one before first pitch, so the **entire pre-game price path was
already sitting in the cache unused** — median 16 hourly candles over 18 hours.
Kalshi's ~2-month retention has since erased that window from the API, so this
study was only possible because the cache had been backed up first, and it
cannot be redone.

`candlesticks` keys its cache on `(ticker, start_ts, end_ts)`, so `clv_study`
must request byte-identically the window `backtest_market` did. Both constants
are duplicated in the example with that warning attached.

## Implementation

- [x] `src/clv.rs`: drift slope (OLS of `close - anchor` on `forecast - anchor`,
      no intercept), directional hit rate over legs disagreeing by >= 5pp, and
      fee-aware settlement P&L that **crosses the spread** — ask for YES,
      `100 - bid` for NO — reusing `paper::kalshi_fee_cents`. One bootstrap
      serves all three, clustered by pitcher-start.
- [x] `examples/clv_study.rs`: prices six anchors from the cached path, reusing
      `backtest::replay`, the settled-market join, the quote-sanity filter and
      the settlement-agreement gate from `backtest_market`.
- [x] Primary anchors fixed in advance — T-6h for the model, T-3h for the sim.
      Six anchors times two forecasters is twelve chances to find something.
- [x] The sim is scored only at T-3h and later, since it needs a confirmed
      lineup. Lineup posting times are recorded nowhere, so 3h is a stated
      assumption, not a measurement.

## Verification

- [x] 8,284 legs across 1,164 starts, **100.000% settlement agreement**.
- [x] Coverage reported per anchor; drops attributed by reason (7 no candles,
      13 no close quote, 0 settlement mismatches).
- [x] Unit tests pin the slope: 1.0 when the close lands on the forecast, 0.0
      when the price never moves, negative when it moves away.
- [x] `cargo test --workspace` (119), strict clippy, `pnpm check`.

## Review — a real lead, an order of magnitude too small

| anchor | legs | drift slope (95% CI) | hit rate | P&L/contract |
|---|---|---|---|---|
| T-23h | 443 | +0.075 [-0.067, +0.180] | 55.6% | -2.81c |
| T-18h | 4,007 | **+0.076 [+0.048, +0.104]** | 57.9% | -3.11c |
| T-12h | 7,219 | **+0.059 [+0.040, +0.078]** | 56.3% | -3.13c |
| **T-6h \*** | 7,982 | **+0.021 [+0.007, +0.034]** | 53.2% | **-3.38c [-5.69, -1.12]** |
| T-3h | 8,165 | +0.009 [-0.001, +0.020] | 52.7% | -2.77c |
| T-1h | 8,261 | +0.003 [-0.001, +0.007] | 50.5% | -3.34c |

The lead is real: at the pre-registered anchor the slope's interval excludes
zero. More convincing than any single number, **the slope decays monotonically
toward first pitch** and the hit rate decays with it to 50.5% in the final hour.
That is information being absorbed, not a shape noise produces.

It is also hopeless economically. Slope 0.021 against a mean disagreement of
6.4pp is ~0.13c of expected movement, against 1.5-3c of spread plus a 2c fee.
Even at 18h, where the lead is strongest, it is ~0.48c against 3-4c of cost.
Realised P&L is negative at every anchor and significantly so at the primary one.
**Pre-registered bar required slope above zero AND positive P&L; the first half
passes, the second fails.**

**The simulator leads about 3x harder than the model** at the same anchor
(+0.028 [+0.017, +0.038] vs +0.009, hit rate 57.6% vs 53.2%). That is the
clearest evidence yet that lineup information is genuinely underpriced in these
thin books — consistent with the sim's only previously measured advantage. Its
P&L interval straddles zero rather than sitting below it, which is *not*
permission to reach for maker fills: the bar said positive, and it is not.

Caveats worth keeping: markets quoted 24h out are the liquid ones, so early
anchors are a selected subset (443 legs at 23h against 8,261 at 1h); and the
close is the best available estimate of truth, not truth.

# The removal hazard is not where the simulator's remaining error lives

## Goal

Close some of the simulator's 0.00550 pooled-Brier gap to the Kalshi market by
fixing the removal-hazard fit, which the build notes flagged as having stalled.

## What was actually wrong

The stall was real and worse than recorded. The *shipped* live fit
(`pns-sim-params-v1.json`) moved all four axes, but the **historical** fit the
backtest depends on (`pns-sim-params-hist-2026-04-30.json`) left `intercept`,
`traffic_slope` and `pivot_scale` at **exactly** their defaults, moving only
`pa_slope`. Coordinate descent accepts a move only when one axis improves the
loss alone, so it halts on any diagonal ridge.

## Implementation

- [x] Deterministic multi-start added to `examples/fit_sim_params.rs`: restart
      positions come from a Halton sequence over the existing axis bounds
      (coprime bases 2/3/5/7), so restarts spread without an RNG and reproduce
      exactly. `--restarts N`.
- [x] Second objective added, `--objective forecast`: pooled survival-ladder
      Brier over `backtest::RUNGS`, scored on the fit window only. The original
      quantile objective stays as `--objective workload` (default).
- [x] Monte Carlo debias in the forecast objective. A survival read off `draws`
      samples carries variance `p(1-p)/draws`, and squaring inflates Brier by
      exactly that — an inflation that grows with how dispersed the simulated
      distribution is, so an uncorrected search would prefer overconfident
      hazards for a purely numerical reason. Subtracting the sample-variance
      estimate removes it.
- [x] Both losses are reported whichever was minimised, since a hazard can buy
      one at the other's expense.
- [x] `--restarts` defaults to **0**, reproducing the shipped artifact. See
      Review for why the better search is opt-in.

## Verification

- [x] Baseline backtest reproduced the 2026-07-30 run exactly (pooled Brier
      0.11333, advantage -0.00130, CI [-0.00223, -0.00034]), which validates
      every comparison below.
- [x] Two runs of the same fit produce byte-identical artifacts.
- [x] `cargo test --workspace`, strict clippy, `pnpm check`, `pnpm test`,
      `pnpm build`.

## Review — negative result, deliberately not shipped

The search improvement works. On the workload objective multi-start reached a
quantile loss of **0.80** where the single start stalled at **1.80**, confirmed
on all 13,441 fitted starts rather than the 4,481-start search sample, with no
parameter pinned to a boundary.

It did not help, and that is the finding:

| hazard | workload loss | backtest Brier | ECE | advantage vs model |
|---|---|---|---|---|
| shipped | 1.80 | 0.11333 | 0.00987 | -0.00130 |
| workload-optimised | **0.80** | 0.11338 | 0.01025 | -0.00124 |
| forecast-optimised | 5.00 | **0.11325** | **0.00813** | **-0.00138** |

Fitting the workload proxy 2.25x better made out-of-sample forecasting slightly
*worse* on all four metrics. Fitting forecast quality directly improved Brier by
0.00008 — around 1.5% of the gap to the market, against a pre-registered bar of
"meaningfully below +0.00550". **Bar not cleared; shipped params unchanged.**

The two objectives are in direct conflict: they move `traffic_slope` in opposite
directions (0.45 -> 0.2375 workload, 0.45 -> 0.9350 forecast), and the forecast
optimum reproduces batters-faced **six times worse** than the workload optimum.
Lowering `traffic_slope` keeps struggling pitchers in longer, which flatters the
workload fit but manufactures strikeouts — the non-strikeout outcome alphabet is
league-average for everyone, so it has no notion of a pitcher being hit hard.

One sub-result is worth keeping: the forecast fit essentially removes the sim's
known high-rung under-prediction (rung 9 error -0.0197 -> +0.0008, the exact
defect the market benchmark flagged) and improves ECE 18%, at the cost of new
over-prediction at rungs 10-12. It was still not shipped, because the bar was set
on Brier in advance and rewriting it afterwards is the failure mode that produced
the quarter-Kelly and 12+pp reversals.

**Conclusion for the next pass: two opposite hazard fits move Brier by less than
0.0002 while workload calibration swings by 6x. The hazard is not the binding
constraint — the remaining error almost certainly sits in the per-batter rate
model, which none of this touches.** The rung 10-12 over-prediction suggests the
forecast fit is partly compensating for a rate-side bias, which is a reason to
fix the rate side before revisiting the hook.

Caveat on method: the held-out window was scored three times (baseline, workload,
forecast). All three are reported above rather than only the winner, but the
independence of that window is partly spent and further hazard tuning against it
should be avoided.

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
