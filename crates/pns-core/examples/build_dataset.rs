fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seasons: Vec<i32> = std::env::args()
        .skip(1)
        .map(|value| value.parse())
        .collect::<Result<_, _>>()?;
    if seasons.len() < 2 {
        return Err("provide at least two seasons, e.g. 2023 2024 2025".into());
    }
    let client = pns_core::data::MlbStatsClient::new()?;
    let rows = pns_core::data::dataset::build_seasons(&client, &seasons, 5)?;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/processed/training_starts.csv");
    pns_core::data::dataset::write_csv(&rows, &path)?;
    println!(
        "wrote {} normalized starts across {} seasons to {}",
        rows.len(),
        seasons.len(),
        path.display()
    );
    Ok(())
}
