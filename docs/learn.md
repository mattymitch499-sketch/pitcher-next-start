# Learn — Pitcher Next-Start Models

> Source draft for the in-app **Learn** page (mlb-PE Learn pattern).
> Port this into a Svelte route with sticky TOC + formula blocks.
> Update this file when model decisions lock (priors, exact feature set, confidence thresholds).

## On this page (TOC)

1. The idea
2. Why Bayesian
3. The models
4. Features
5. Priors and shrinkage
6. Posterior → card fields
7. Leakage rule
8. Worked example
9. What it doesn’t do
10. How to use the card
11. Later extensions

---

## 1 · The idea

Most “pitching analysis” is descriptive: last-5 recaps, matchup boards, vibes about an ace vs a bad lineup.

This app does something narrower and harder:

**Predict a starting pitcher’s next start** from public form + context, then turn that projection into:

- a **player-prop lean** (K / ER / outs-adjacent thinking), and/or
- a **team-model input** (expected runs allowed by that starter’s outing)

The product is a projection card, not a highlight package.

---

## 2 · Why Bayesian

A starter’s recent sample is thin. Five starts is not “big data.” Point estimates lie with confidence they didn’t earn.

We want three things a plain OLS fit does not give cleanly:

1. **A central projection** (what we expect next)
2. **A real uncertainty band** (how wide that expectation is)
3. **Shrinkage** when the sample is weak (pull toward league-average / prior belief)

That’s why Phase 1 is **Bayesian linear regression from day one** — not frequentist ridge first with a Bayesian upgrade later.

Intuition:

- lots of stable history → posterior tightens → higher confidence
- short/noisy history → posterior stays wide → lower confidence, stronger prior pull

---

## 3 · The models

### Family (locked)

Six independent Bayesian linear regressions sharing one feature pipeline: next-start FIP, runs allowed, innings pitched, and strikeouts (the four card targets), plus per-batter strikeout rate and batters faced (the two components that build the discrete strikeout distribution below). Each has separate coefficients, priors, observation noise, and posteriors.

Observation noise is **learned, not assumed**. Each regression is a conjugate Normal-Inverse-Gamma model: the noise variance has an Inverse-Gamma prior whose mean is a documented default, and the posterior noise is estimated from the training residuals. With thousands of starts the data dominate, so predictive intervals are honest at the source instead of relying entirely on after-the-fact rescaling.

Training and scoring are deliberately separate. Walk-forward validation and final fitting produce a versioned model artifact with the four posteriors, standardization values, interval calibration, confidence audit, feature contract, and training cutoff. A daily slate loads that frozen artifact and builds only the probable starters' live feature rows, so unrelated slate composition cannot change the fitted model or confidence boundaries.

### Observation model

For each historical next-start row \(i\):

\[
y_i = \beta_0 + x_i^{\top}\beta + \varepsilon_i
\]

\[
\varepsilon_i \sim \mathcal{N}(0, \sigma^2)
\]

Where:

- \(y_i\) = target for that start. Independent Bayesian regressions predict next-start FIP, runs allowed, innings pitched, and strikeouts.
- \(x_i\) = features known **before** the start
- \(\beta\) = coefficients (effects of form/context)
- \(\sigma\) = residual noise

### Scoring a future start

For a new feature vector \(x^*\):

\[
p(y^* \mid x^*, \text{data})
\]

That **posterior predictive** distribution is the object we summarize on the card.

### Candidate engine

[rustmc](https://tbosier.github.io/rustmc/) (Rust-native Bayesian inference).  
Engine wiring may evolve; the **model family does not**.

### Explicit non-choices in Phase 1

- no deep learning
- no hierarchical pitcher/team/season multi-level model
- no frequentist stand-in shipped as the product model

---

## 4 · Features

Features come from two buckets.

### A. Rolling form (default last 5 starts, fallback to available)

Examples:

- K rate
- BB rate
- HR/9
- WHIP
- FIP components / ER per 9
- average FB velo (+ trend if available)
- GB% if available

### B. Next-start context

- home / away
- park factor
- opponent offense vs hand
- rest days

In the live Phase 1 implementation, opponent offense is team OPS versus the starter's handedness relative to the league split average. The park factor is the home club's home-versus-away runs-per-game ratio, bounded to reduce early-season noise.

### Rule

Only information available **as of** the projection moment enters \(x\).  
Future box-score results never leak into past feature rows.

---

## 5 · Priors and shrinkage

Phase 1 uses **weakly informative priors**.

Intent, not dogma:

- coefficients start near “small / league-plausible”
- with little data, predictions don’t go insane
- with more data, likelihood dominates

Practical product effect:

- a pitcher with 2 weird starts does **not** get a wild card
- a pitcher with a longer, coherent form signal can move farther from the prior

The conjugate implementation uses independent Gaussian coefficient priors with marginal variance **4.0** at the prior noise level. Intercept priors center at **4.20 FIP**, **3.0 runs**, **5.3 innings**, **5.2 strikeouts**, a **0.22** per-batter strikeout rate, and **24 batters faced**. Noise-variance priors center at the previous fixed constants (2.25 for FIP and runs, 6.25 for strikeouts, 1.0 for innings) but are updated by the residuals, so a misjudged constant cannot silently distort the bands. Each target has its own posterior; innings and strikeouts are not derived from expected runs. These are inspectable defaults, not universal baseball constants.

---

## 6 · Posterior → card fields

Each model produces its own posterior. The card is a translation layer across all four outputs.

| Card field | Meaning | Source |
|---|---|---|
| `projected_fip` | Best single number for next-start FIP | Posterior mean of target (or posterior predictive mean) |
| `expected_runs_low / base / high` | Runs allowed band for the outing | Credible interval + central estimate from posterior predictive (or calibrated mapping from FIP) |
| `expected_innings` | Expected starter workload | Posterior mean from the innings model |
| `expected_strikeouts` | Expected strikeout total | Posterior mean from the strikeout model |
| `expected_strikeouts_compositional` | Mean of the discrete strikeout distribution | Beta-binomial mixture (below) |
| `strikeout_over_probabilities` | P(strikeouts > line) for lines 3.5–8.5 | Discrete strikeout distribution |
| `confidence` | high / medium / low | Walk-forward-calibrated run uncertainty + recent-start sample strength |
| `judgment` | One sentence a human can act on | Rule/template over projection + confidence + context |
| `prop_lean` (optional) | Betting-facing lean | Derived from projection vs market/context later |
| `team_model_input` | Expected runs allowed input | Central runs estimate (and optionally band) |

### The discrete strikeout distribution

Strikeouts are counts, and betting lines ask for tail probabilities such as P(K ≥ 6.5) — a Gaussian mean-and-band cannot price that honestly. The app therefore builds a full probability for every strikeout total:

1. The **per-batter strikeout rate** model predicts \(\hat{p}\), the chance this pitcher strikes out any given batter in this context.
2. The **batters faced** model predicts workload \(\hat{b}\), and its walk-forward residual quantiles form an empirical workload distribution — early hooks and deep starts shape the tails instead of a symmetric bell curve.
3. For each workload draw, strikeouts follow a **beta-binomial**: a binomial in \(\hat{p}\) whose game-to-game variability is governed by a concentration parameter.

\[
P(K = k) = \sum_{b} P(\text{BF} = b)\; \text{BetaBin}(k;\, b,\, \hat{p}\nu,\, (1-\hat{p})\nu)
\]

The concentration \(\nu\) is chosen by maximizing predictive log-likelihood on the earlier 70% of held-out walk-forward dates. The later 30% — which had no say in that choice — audits the distribution: mean log score against the old Gaussian treatment, point MAE for both estimators, and predicted-versus-observed frequency at each common prop line. Those numbers live in `outputs/validation.json` under `k_distribution`.

### Confidence calibration

Confidence is a reliability tier, not an opinion about whether the pitcher is good. The model separates reducible parameter uncertainty from the irreducible outing noise used in the run band, normalizes that epistemic uncertainty for the model's training-set size, then increases the score when fewer than five recent starts are available. Five-start predictions from the earlier 70% of held-out dates set the high/medium boundaries at the 33rd and 67th percentiles. The later 30% audit all tiers against results that did not choose those boundaries.

The three learned tiers activate only when every audit tier has at least 100 starts and run MAE improves by at least 5% from low to medium and again from medium to high. If that validation gate fails, the app withholds **high**. Complete five-start histories inside the candidate-medium uncertainty boundary are **medium**; unusually uncertain or shorter histories are **low**. It does not force variety unsupported by later results.

- **High:** lowest calibrated uncertainty score
- **Medium:** middle calibrated uncertainty score
- **Low:** highest calibrated uncertainty score — trust the band more than the point

The displayed interval still uses total predictive uncertainty and clamps impossible negative runs to zero. Confidence does not use that clipped width, preventing low projected-run totals from looking artificially precise.

Uncertainty is a feature. Hiding it would make the app dumber.

The run interval is calibrated from walk-forward residuals. The model first measures raw 90% coverage on future-only held-out dates, then scales the production band by the empirical 90th percentile of standardized run errors. Both raw and calibrated coverage remain visible in `outputs/validation.json`.

### Current measured baseline check

The processed dataset contains 16,477 normalized starts from 2023 through completed games on July 16, 2026. Walk-forward validation scores FIP, runs, innings, and strikeouts only on future dates. The app reports separate MAE values for the innings and strikeout models alongside the existing FIP and runs validation. This remains model evidence rather than proof of a betting edge, so the app does not emit automatic prop recommendations.

---

## 7 · Leakage rule

If the model can see the future, the demo is fake.

Hard rules:

1. Rolling features for start \(t\) use only starts \(< t\)
2. Train/validate with **time-aware** splits (train past → test future)
3. Normalized tables are the only model inputs
4. Raw API JSON never feeds the model directly

If a feature can’t be known before first pitch, it doesn’t ship.

---

## 8 · Worked example (illustrative)

> Numbers below are pedagogical. Replace with a fixture-backed example once M3/M4 exist.

**Pitcher:** Example RHP, 5 prior starts  
**Next start:** road, neutral-ish park, average offense, 5 days rest

### Suppose rolling features (last 5)

- K rate: elevated
- BB rate: slightly elevated
- HR/9: near average
- WHIP: slightly better than average
- FB velo: stable

### Context

- road
- park factor ≈ 1.00
- opponent vs R ≈ league average
- rest = 5

### Model output (illustrative posterior summary)

- `projected_fip` = **3.85** (posterior mean)
- expected runs band = **2.4 / 3.1 / 4.0**
- `confidence` = **medium** (only 5 starts; band still meaningful)

### Judgment (example)

> “Solid mid-3s FIP projection with medium confidence; fine as a team-model input, not a hammer prop without a market gap.”

That’s the product: a number, a band, a confidence tag, and a sentence.

---

## 9 · What it doesn’t do

Phase 1 does **not**:

- predict full game win probability by itself
- ingest live sportsbook odds
- manage bankroll
- model every pitch / Stuff+ warehouse
- run hierarchical multi-season Bayesian structures
- replace human judgment on whether a lean is bettable after vig

It also does not claim certainty. A tight card can still lose. The point is calibrated process, not prophecy.

---

## 10 · How to use the card

### Player props

Use the projection as a prior/lean generator:

- higher K-rate form + favorable context → K prop interest
- elevated expected runs band → ER / outs caution

Always compare to market. Projection alone is not a ticket.

### Team model input

Feed `team_model_input` (expected runs allowed) into a larger team/game model as the starter’s contribution piece.

This app is a **module**, not the whole stack.

---

## 11 · Later extensions

After Phase 1 works on camera:

- hierarchical pitcher/team/season effects
- richer pitch-quality / Statcast features
- richer pitch-level targets such as whiffs and CSW, if added to the normalized data contract
- calibration dashboards
- optional Playground (tweak features/priors live) — **not required for v1**
- thin CLI for batch scoring the same core

None of those replace Learn. They add sections to it.

---

## Implementation notes (for agents)

- UI pattern reference: `WalrusQuant/mlb-pe` → `src/routes/learn/+page.svelte` + `Formula.svelte`
- Keep Learn static in Phase 1 (no backend required)
- Nav label: **Learn**
- If model math changes, update this file first, then the Svelte page
- Do not invent a different model story in UI copy than `techSpec.md` §8

## One-line promise for Learn

**Show the Bayesian next-start math in plain language so the projection card is explainable, not mystical.**
