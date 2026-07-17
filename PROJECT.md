# Pitcher Next-Start — Project Index

> Seed project for a predictive MLB pitcher next-start model (props + team-model input), built as an AI-agent demo. **Rust + Tauri + Svelte**, data via a **baseballr-parity** public surface.

This folder follows the **project-planner** template (same quality bar as `projects/ballast_template/`).

## Canonical docs

| File | Role |
|---|---|
| **[README.md](./README.md)** | Product doc — positioning, problem/solution, architecture, phases, competitive reality |
| **[techSpec.md](./techSpec.md)** | Phase 1 technical specification — scope gate, types, subsystems, app shell, M0–M5 sequence |
| **[AGENTS.md](./AGENTS.md)** | Rules for Grok Build / Claude / Cursor agents implementing the repo |
| **[docs/learn.md](./docs/learn.md)** | Learn section source — Bayesian model math walkthrough (mlb-PE style) |

## Stack (locked)

- **Rust** — data clients, features, model, projection/card generation (core library)
- **Tauri v2** — desktop app shell / IPC / native packaging
- **Svelte** — UI (minimal in Phase 1; not a polished dashboard)

Adam’s normal desktop stack. Coding agents should treat global `AGENTS.md` / `CLAUDE.md` stack conventions as authoritative for Tauri/Svelte project layout and patterns.

## One-line promise

**Turn recent public pitcher form into a next-start projection and one clear betting judgment — in a Rust-core Tauri app, agent-buildable, exportable to props and team models.**

## Hard constraints

1. Predictive first (not matchup boards as the product).
2. All data from baseballr-class public sources; **port needed pulls to Rust** (not Python-primary).
3. Outputs: projected FIP, runs band, optional Ks; CSV for props + team-model handoff.
4. Bayesian linear regression from day one (not frequentist first). Transparent posteriors; time-aware validation only. No hierarchical models in Phase 1.
5. UI is a thin shell over the core. Core pipeline works without a pretty interface.
6. **Learn section is required.** In-app math walkthrough of the model (mlb-PE Learn pattern). Draft content lives in `docs/learn.md`; Phase 1 UI includes a Learn route/page.

## Phase 1 gate (summary)

Open the app → pick a date → score ≥2 starters → show cards with FIP + runs band + judgment → export `outputs/predictions.csv` → open **Learn** and walk the Bayesian math. No leakage. Explainable on camera.

See techSpec §1.3 and §15 for the full gate and milestone order. Learn content source of truth: `docs/learn.md`.

## Status

Live Phase 1 vertical slice implemented. Date-specific probable starters and pitcher histories come from MLB Stats API and are normalized before modeling. Core tests, Svelte checks/build, and Tauri workspace compile are green; production calibration and richer context data remain.
