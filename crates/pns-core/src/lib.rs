pub mod backtest;
pub mod card;
pub mod clv;
pub mod data;
pub mod error;
pub mod features;
pub mod model;
pub mod paper;
pub mod paper_stats;
pub mod sim;
pub mod tracking;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub use card::{Confidence, DecisionCard, Projection};
pub use error::{PnsError, Result};
pub use model::artifact::ModelInfo;

#[derive(Debug, serde::Serialize)]
pub struct ScoredSlate {
    pub cards: Vec<DecisionCard>,
    pub validation: model::ValidationMetrics,
    pub model: ModelInfo,
    pub generated_at: String,
}

#[derive(serde::Serialize)]
struct PredictionCsv<'a> {
    model_artifact_id: Option<&'a str>,
    model_trained_through: Option<String>,
    pitcher_id: u32,
    pitcher_name: &'a str,
    pitcher_team: &'a str,
    opponent_team: &'a str,
    target_game_date: &'a str,
    projected_fip: f64,
    expected_runs_low: f64,
    expected_runs_p25: f64,
    expected_runs_base: f64,
    expected_runs_p75: f64,
    expected_runs_high: f64,
    expected_strikeouts: f64,
    expected_innings: f64,
    expected_strikeouts_low: f64,
    expected_strikeouts_high: f64,
    expected_innings_low: f64,
    expected_innings_high: f64,
    expected_strikeouts_compositional: f64,
    p_strikeouts_over_3_5: f64,
    p_strikeouts_over_4_5: f64,
    p_strikeouts_over_5_5: f64,
    p_strikeouts_over_6_5: f64,
    p_strikeouts_over_7_5: f64,
    p_strikeouts_over_8_5: f64,
    confidence: Confidence,
    recent_k_rate: f64,
    recent_bb_rate: f64,
    recent_hr_per_9: f64,
    recent_whip: f64,
    recent_er_per_9: f64,
    recent_ground_ball_pct: f64,
    recent_fastball_velocity: Option<f64>,
    park_factor: f64,
    opponent_offense_factor: f64,
    judgment: &'a str,
    team_model_input: Option<f64>,
}

fn over_probability(card: &DecisionCard, line: f64) -> f64 {
    card.strikeout_over_probabilities
        .iter()
        .find(|entry| entry.line == line)
        .map(|entry| entry.probability)
        .unwrap_or(0.0)
}

impl<'a> PredictionCsv<'a> {
    fn new(card: &'a DecisionCard, model: Option<&'a ModelInfo>) -> Self {
        Self {
            model_artifact_id: model.map(|info| info.artifact_id.as_str()),
            model_trained_through: model.map(|info| info.trained_through.to_string()),
            pitcher_id: card.pitcher_id,
            pitcher_name: &card.pitcher_name,
            pitcher_team: &card.pitcher_team_name,
            opponent_team: &card.opponent_team_name,
            target_game_date: &card.target_game_date,
            projected_fip: card.projected_fip,
            expected_runs_low: card.expected_runs_low,
            expected_runs_p25: card.expected_runs_p25,
            expected_runs_base: card.expected_runs_base,
            expected_runs_p75: card.expected_runs_p75,
            expected_runs_high: card.expected_runs_high,
            expected_strikeouts: card.expected_strikeouts,
            expected_innings: card.expected_innings,
            expected_strikeouts_low: card.expected_strikeouts_low,
            expected_strikeouts_high: card.expected_strikeouts_high,
            expected_innings_low: card.expected_innings_low,
            expected_innings_high: card.expected_innings_high,
            expected_strikeouts_compositional: card.expected_strikeouts_compositional,
            p_strikeouts_over_3_5: over_probability(card, 3.5),
            p_strikeouts_over_4_5: over_probability(card, 4.5),
            p_strikeouts_over_5_5: over_probability(card, 5.5),
            p_strikeouts_over_6_5: over_probability(card, 6.5),
            p_strikeouts_over_7_5: over_probability(card, 7.5),
            p_strikeouts_over_8_5: over_probability(card, 8.5),
            confidence: card.confidence,
            recent_k_rate: card.recent_k_rate,
            recent_bb_rate: card.recent_bb_rate,
            recent_hr_per_9: card.recent_hr_per_9,
            recent_whip: card.recent_whip,
            recent_er_per_9: card.recent_er_per_9,
            recent_ground_ball_pct: card.recent_ground_ball_pct,
            recent_fastball_velocity: card.recent_fastball_velocity,
            park_factor: card.park_factor,
            opponent_offense_factor: card.opponent_offense_factor,
            judgment: &card.judgment,
            team_model_input: card.team_model_input,
        }
    }
}

/// Fetch and score the probable starters MLB lists for the requested date.
/// Network responses are normalized before reaching feature/model code.
pub fn score_date(date: &str) -> Result<Vec<DecisionCard>> {
    Ok(score_date_detailed(date)?.cards)
}

pub fn score_date_detailed(date: &str) -> Result<ScoredSlate> {
    let target_date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| PnsError::Feature(format!("invalid date {date}: {e}")))?;
    let artifact = load_model_artifact()?;
    if target_date <= artifact.training.data_through {
        return Err(PnsError::Model(format!(
            "model artifact {} was trained through {}; choose a later slate or train a leakage-safe historical artifact",
            artifact.artifact_id, artifact.training.data_through
        )));
    }
    let client = data::MlbStatsClient::new()?;
    let mut contexts = client.probable_starters(target_date)?;
    let pitcher_ids: Vec<_> = contexts.iter().map(|context| context.pitcher_id).collect();
    let hands = client.pitcher_hands(&pitcher_ids)?;
    // Serve the same opponent/park statistic the model was trained on, not the
    // live team-splits ratio it used to be scored against.
    let as_of =
        data::asof::AsOfContext::build(&client, target_date, artifact.training.data_through)?;
    for context in &mut contexts {
        context.pitcher_hand = hands.get(&context.pitcher_id).copied();
        context.park_factor = as_of.park_factor(context.home_team_id);
        context.opponent_offense = as_of.opponent_offense_factor(
            context.opponent_id,
            hands.get(&context.pitcher_id).copied().unwrap_or('R'),
        );
    }
    let histories: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = contexts
            .iter()
            .map(|context| {
                let client = client.clone();
                scope.spawn(move || {
                    let mut rows = client.pitcher_starts(context.pitcher_id, target_date)?;
                    let from = target_date - chrono::Duration::days(400);
                    let to = target_date - chrono::Duration::days(1);
                    if let Ok(velocities) =
                        client.savant_fastball_velocity(context.pitcher_id, from, to)
                    {
                        data::enrich::join_velocity(&mut rows, context.pitcher_id, &velocities);
                    }
                    // Both Savant calls degrade the same way: a failure costs the
                    // feature, not the slate. The feature builder falls back to
                    // its missing-data sentinel, which standardizes to the mean.
                    if let Ok(outcomes) = client.savant_pitch_outcomes(context.pitcher_id, from, to)
                    {
                        data::enrich::join_pitch_outcomes(&mut rows, context.pitcher_id, &outcomes);
                    }
                    Ok::<_, PnsError>(rows)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("pitcher history worker panicked"))
            .collect()
    });
    let mut starts = Vec::new();
    let mut failures = Vec::new();
    for (context, history) in contexts.iter().zip(histories) {
        match history {
            Ok(mut rows) => starts.append(&mut rows),
            Err(error) => failures.push(format!("{}: {error}", context.pitcher_name)),
        }
    }
    if starts.is_empty() {
        return Err(PnsError::Data(format!(
            "no usable pitcher histories were returned: {}",
            failures.join("; ")
        )));
    }
    let deduplicated: BTreeMap<_, _> = starts
        .into_iter()
        .map(|row| ((row.pitcher_id, row.game_pk), row))
        .collect();
    starts = deduplicated.into_values().collect();
    for row in &mut starts {
        if let Some(hand) = hands.get(&row.pitcher_id) {
            row.pitch_hand = Some(*hand);
        }
    }
    let all_hands: std::collections::HashMap<_, _> = starts
        .iter()
        .filter_map(|row| row.pitch_hand.map(|hand| (row.pitcher_id, hand)))
        .collect();
    data::enrich::enrich_historical_context(&mut starts, &all_hands);
    let mut slate = score_normalized_detailed(&starts, &contexts, &artifact)?;
    attach_sim_forecasts(&client, target_date, &mut slate.cards);
    Ok(slate)
}

/// Prices each card with the per-PA simulator when the opposing lineup is
/// posted, storing the PMF on the card for side-by-side logging. Log-only:
/// nothing here influences positions, and every failure downgrades to a card
/// data warning so ordinary scoring never breaks on the sim's account.
fn attach_sim_forecasts(
    client: &data::MlbStatsClient,
    date: chrono::NaiveDate,
    cards: &mut [DecisionCard],
) {
    let warn_all = |cards: &mut [DecisionCard], message: String| {
        for card in cards.iter_mut() {
            card.data_warnings.push(message.clone());
        }
    };
    let params = match sim::SimParams::load() {
        Ok(params) => params,
        Err(error) => {
            return warn_all(cards, format!("Simulator unavailable: {error}"));
        }
    };
    let lineups = match client.slate_lineups(date) {
        Ok(lineups) => lineups,
        Err(error) => {
            return warn_all(cards, format!("Simulator skipped, no lineup feed: {error}"));
        }
    };
    let rates = match data::batters::live_batter_rates(client, date) {
        Ok(rates) => rates,
        Err(error) => {
            return warn_all(cards, format!("Simulator skipped, no batter data: {error}"));
        }
    };
    for card in cards.iter_mut() {
        let Some(hand) = card.pitcher_hand else {
            continue;
        };
        // The lineup the pitcher faces is the OTHER side of his game.
        let lineup = lineups.iter().find_map(|game| {
            if game.away.probable_pitcher_id == Some(card.pitcher_id) {
                game.home.batters.as_deref()
            } else if game.home.probable_pitcher_id == Some(card.pitcher_id) {
                game.away.batters.as_deref()
            } else {
                None
            }
        });
        let Some(batters) = lineup else {
            card.data_warnings
                .push("Opposing lineup not posted yet; simulator projection pending.".into());
            continue;
        };
        let inputs = sim::StartInputs::from_lineup(
            card.k_rate_mean,
            card.batters_faced_mean,
            &rates,
            batters,
            hand,
        );
        card.sim_strikeout_probabilities = Some(sim::simulate_start(
            &inputs,
            &params,
            sim::DEFAULT_DRAWS,
            sim_seed(card.pitcher_id, &card.target_game_date),
        ));
    }
}

/// Deterministic per-start seed so a projection never jitters between clicks of
/// the daily button.
///
/// Hand-rolled FNV-1a rather than `DefaultHasher`: the standard hasher's output
/// is explicitly not stable across Rust releases, so a toolchain upgrade would
/// silently reseed every start and move published probabilities. Same reasoning
/// as `backtest::ReplayCase::seed`.
fn sim_seed(pitcher_id: u32, target_game_date: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut absorb = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    };
    for byte in pitcher_id.to_le_bytes() {
        absorb(byte);
    }
    for byte in target_game_date.as_bytes() {
        absorb(*byte);
    }
    hash
}

fn find_model_artifact() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("PNS_MODEL_ARTIFACT") {
        let path = PathBuf::from(path);
        return Some(path);
    }
    let current = std::env::current_dir().ok()?;
    [
        current.join("models/pns-model-v3.json"),
        current.join("../models/pns-model-v3.json"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn load_model_artifact() -> Result<model::artifact::ModelArtifact> {
    if let Some(path) = find_model_artifact() {
        return model::artifact::ModelArtifact::load(path);
    }
    model::artifact::ModelArtifact::from_json(include_str!("../../../models/pns-model-v3.json"))
}

#[cfg(test)]
fn score_normalized(
    starts: &[data::types::PitcherStartLog],
    contexts: &[data::types::NextStartContext],
    artifact: &model::artifact::ModelArtifact,
) -> Result<Vec<DecisionCard>> {
    Ok(score_normalized_detailed(starts, contexts, artifact)?.cards)
}

fn score_normalized_detailed(
    starts: &[data::types::PitcherStartLog],
    contexts: &[data::types::NextStartContext],
    artifact: &model::artifact::ModelArtifact,
) -> Result<ScoredSlate> {
    artifact.validate()?;
    let cards: Vec<_> = contexts
        .iter()
        .filter_map(|context| {
            let row = match features::next_start_row(starts, context, 5) {
                Ok(row) => row,
                Err(_) => return None,
            };
            Some(artifact.score(&row).map(|projection| {
                card::build_card(
                    context,
                    projection,
                    &row,
                    starts,
                    artifact.validation.confidence_calibration,
                )
            }))
        })
        .collect::<Result<_>>()?;
    if cards.is_empty() {
        return Err(PnsError::Data(
            "probable starters were listed, but none had enough prior MLB starts to score".into(),
        ));
    }
    Ok(ScoredSlate {
        cards,
        validation: artifact.validation.clone(),
        model: artifact.info(),
        generated_at: chrono::Utc::now().to_rfc3339(),
    })
}

pub fn score_date_fixture(date: &str) -> Result<Vec<DecisionCard>> {
    Ok(score_date_fixture_detailed(date)?.cards)
}

pub fn score_date_fixture_detailed(date: &str) -> Result<ScoredSlate> {
    let target_date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map_err(|e| PnsError::Feature(format!("invalid date {date}: {e}")))?;
    let starts = data::pitcher_logs::fixture_starts()?;
    let contexts = vec![
        data::types::NextStartContext {
            pitcher_id: 660271,
            pitcher_name: "Fixture Ace".into(),
            game_date: target_date,
            opponent_id: 147,
            home_team_id: 120,
            home: true,
            park_factor: 1.0,
            opponent_offense: 1.0,
            pitcher_team_name: "Fixture Club".into(),
            opponent_team_name: "Fixture Opponent".into(),
            venue_name: "Fixture Park".into(),
            game_time: None,
            pitcher_hand: Some('R'),
        },
        data::types::NextStartContext {
            pitcher_id: 621141,
            pitcher_name: "Fixture Southpaw".into(),
            game_date: target_date,
            opponent_id: 119,
            home_team_id: 119,
            home: false,
            park_factor: 1.0,
            opponent_offense: 1.0,
            pitcher_team_name: "Fixture Club".into(),
            opponent_team_name: "Fixture Opponent".into(),
            venue_name: "Fixture Park".into(),
            game_time: None,
            pitcher_hand: Some('L'),
        },
    ];
    let artifact = load_model_artifact()?;
    score_normalized_detailed(&starts, &contexts, &artifact)
}

pub fn export_predictions(cards: &[DecisionCard], root: impl AsRef<Path>) -> Result<PathBuf> {
    export_predictions_with_model(cards, None, root)
}

pub fn export_predictions_with_model(
    cards: &[DecisionCard],
    model: Option<&ModelInfo>,
    root: impl AsRef<Path>,
) -> Result<PathBuf> {
    let output_dir = root.as_ref().join("outputs");
    let cards_dir = output_dir.join("cards");
    std::fs::create_dir_all(&cards_dir)?;
    for entry in std::fs::read_dir(&cards_dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "md") {
            std::fs::remove_file(path)?;
        }
    }
    let csv_path = output_dir.join("predictions.csv");
    let mut writer = csv::Writer::from_path(&csv_path)?;
    for card in cards {
        writer.serialize(PredictionCsv::new(card, model))?;
        let slug = card.pitcher_name.to_lowercase().replace([' ', '.'], "-");
        std::fs::write(
            cards_dir.join(format!("{slug}.md")),
            card::render_markdown(card),
        )?;
    }
    writer.flush()?;
    Ok(csv_path)
}

#[cfg(test)]
mod tests {
    /// The seed must depend on both inputs and must not drift. The pinned
    /// constant is the point of the test: if a refactor changes it, every
    /// published simulator probability moves, and this fails instead of the
    /// change landing silently.
    #[test]
    fn sim_seed_is_stable_and_input_dependent() {
        assert_eq!(
            super::sim_seed(660271, "2026-07-30"),
            16_343_921_367_459_293_529
        );
        assert_ne!(
            super::sim_seed(660271, "2026-07-30"),
            super::sim_seed(660271, "2026-07-31"),
            "a different date must reseed"
        );
        assert_ne!(
            super::sim_seed(660271, "2026-07-30"),
            super::sim_seed(660272, "2026-07-30"),
            "a different pitcher must reseed"
        );
    }

    #[test]
    fn scores_and_exports_two_fixture_pitchers() {
        let cards = super::score_date_fixture("2026-07-17").unwrap();
        assert_eq!(cards.len(), 2);
        assert!(cards.iter().all(|card| card.starts_used == 5));
        assert!(cards.iter().all(|card| card.recent_k_rate > 0.0));
        assert!(cards.iter().all(|card| card.park_factor > 0.0));
        assert!(cards.iter().all(|card| !card.drivers.is_empty()));
        assert!(cards.iter().all(|card| card.season_form.starts >= 5));
        assert!(cards.iter().all(|card| card.recent_starts.len() == 5));
        assert!(cards.iter().all(|card| !card.outlook_summary.is_empty()));
        assert!(cards.iter().all(|card| card.starter_status == "Probable"));
        assert!(cards.iter().all(|card| card.data_through != "Unavailable"));
        assert!(cards.iter().all(|card| card.history_starts >= 5));
        assert!(cards.iter().all(|card| card.expected_innings > 0.0));
        assert!(cards.iter().all(|card| card.expected_strikeouts >= 0.0));
        assert!(
            cards
                .iter()
                .all(|card| card.strikeout_over_probabilities.len() == 6)
        );
        assert!(cards.iter().all(|card| {
            card.strikeout_over_probabilities
                .windows(2)
                .all(|pair| pair[0].probability >= pair[1].probability)
        }));
        assert!(
            cards
                .iter()
                .all(|card| (card.strikeout_probabilities.iter().sum::<f64>() - 1.0).abs() < 0.01)
        );
        assert!(
            cards
                .iter()
                .all(|card| card.expected_strikeouts_compositional > 0.0)
        );
        assert!(cards.iter().all(|card| {
            card.recent_starts
                .iter()
                .all(|start| start.batters_faced > 0)
        }));
        assert!(
            cards
                .iter()
                .all(|card| card.expected_runs_low <= card.expected_runs_p25
                    && card.expected_runs_p25 <= card.expected_runs_base
                    && card.expected_runs_base <= card.expected_runs_p75
                    && card.expected_runs_p75 <= card.expected_runs_high)
        );
        assert!(
            cards
                .iter()
                .all(|c| c.expected_runs_low < c.expected_runs_high)
        );
        let root = std::env::temp_dir().join("pns-core-export-test");
        let path = super::export_predictions(&cards, &root).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn frozen_artifact_scores_are_independent_of_unrelated_history() {
        let starts = crate::data::pitcher_logs::fixture_starts().unwrap();
        let artifact = crate::model::artifact::ModelArtifact::train(&starts).unwrap();
        let target_date = chrono::NaiveDate::from_ymd_opt(2026, 7, 16).unwrap();
        let context = crate::data::types::NextStartContext {
            pitcher_id: 660271,
            pitcher_name: "Fixture Ace".into(),
            game_date: target_date,
            opponent_id: 147,
            home_team_id: 120,
            home: true,
            park_factor: 1.0,
            opponent_offense: 1.0,
            pitcher_team_name: "Fixture Club".into(),
            opponent_team_name: "Fixture Opponent".into(),
            venue_name: "Fixture Park".into(),
            game_time: None,
            pitcher_hand: Some('R'),
        };
        let first = super::score_normalized(&starts, std::slice::from_ref(&context), &artifact)
            .unwrap()
            .remove(0);
        let mut expanded = starts.clone();
        let mut unrelated = starts[0].clone();
        unrelated.pitcher_id = 999_999;
        unrelated.game_pk = 999_999;
        expanded.push(unrelated);
        let second = super::score_normalized(&expanded, &[context], &artifact)
            .unwrap()
            .remove(0);
        assert_eq!(first.projected_fip, second.projected_fip);
        assert_eq!(first.expected_runs_base, second.expected_runs_base);
        assert_eq!(first.confidence, second.confidence);
    }
}
