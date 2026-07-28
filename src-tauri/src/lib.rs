use pns_core::{DecisionCard, ScoredSlate};
use std::sync::Mutex;
use tauri::{Manager, State};

/// Guards paper_trades.csv's read-modify-write cycle. Without this, two
/// overlapping calls to `paper_trading_update` (e.g. a stray double-click,
/// or a page reload while one is still in flight) can each read the file,
/// fetch results independently, and write back — whichever finishes last
/// silently erases the other's correctly-graded results.
type PaperTradingLock = Mutex<()>;

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
fn paper_trading_update(
    app: tauri::AppHandle,
    lock: State<PaperTradingLock>,
    date: String,
) -> Result<pns_core::paper::PaperLog, String> {
    let _guard = lock.lock().map_err(|error| error.to_string())?;
    let root = writable_root(&app)?;
    let slate = pns_core::score_date_detailed(&date).map_err(|error| error.to_string())?;
    pns_core::paper::update(&slate.cards, &slate.model, root).map_err(|error| error.to_string())
}

#[tauri::command]
fn paper_trading_status(app: tauri::AppHandle) -> Result<pns_core::paper::PaperLog, String> {
    let root = writable_root(&app)?;
    pns_core::paper::status(root).map_err(|error| error.to_string())
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
        .manage(PaperTradingLock::default())
        .invoke_handler(tauri::generate_handler![
            score_date,
            export_predictions,
            prediction_history,
            reconcile_prediction_history,
            paper_trading_update,
            paper_trading_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running Pitcher Next-Start");
}
