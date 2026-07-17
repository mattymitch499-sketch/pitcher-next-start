use pns_core::{DecisionCard, ScoredSlate};
use tauri::Manager;

fn writable_root(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    app.path().app_data_dir().map_err(|error| error.to_string())
}

#[tauri::command]
fn score_date(app: tauri::AppHandle, date: String) -> Result<ScoredSlate, String> {
    let slate = pns_core::score_date_detailed(&date).map_err(|error| error.to_string())?;
    let root = writable_root(&app)?;
    pns_core::tracking::record_predictions(&slate.cards, &slate.model, root)
        .map_err(|error| error.to_string())?;
    Ok(slate)
}

#[tauri::command]
fn prediction_history(
    app: tauri::AppHandle,
) -> Result<Vec<pns_core::tracking::TrackedPrediction>, String> {
    let root = writable_root(&app)?;
    pns_core::tracking::read_history(root).map_err(|error| error.to_string())
}

#[tauri::command]
fn reconcile_prediction_history(
    app: tauri::AppHandle,
) -> Result<Vec<pns_core::tracking::TrackedPrediction>, String> {
    let root = writable_root(&app)?;
    pns_core::tracking::reconcile_history(root).map_err(|error| error.to_string())
}

#[tauri::command]
fn export_predictions(
    app: tauri::AppHandle,
    cards: Vec<DecisionCard>,
    model: Option<pns_core::ModelInfo>,
) -> Result<String, String> {
    let root = writable_root(&app)?;
    pns_core::export_predictions_with_model(&cards, model.as_ref(), root)
        .map(|path| path.display().to_string())
        .map_err(|error| error.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            score_date,
            export_predictions,
            prediction_history,
            reconcile_prediction_history
        ])
        .run(tauri::generate_context!())
        .expect("error while running Pitcher Next-Start");
}
