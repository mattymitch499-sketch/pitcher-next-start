fn main() -> Result<(), Box<dyn std::error::Error>> {
    let date = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "2026-07-16".into());
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cards = pns_core::score_date_fixture(&date)?;
    let path = pns_core::export_predictions(&cards, root)?;
    println!("wrote {} cards to {}", cards.len(), path.display());
    for card in cards {
        println!(
            "{}: FIP {:.2}, runs {:.2}/{:.2}/{:.2}",
            card.pitcher_name,
            card.projected_fip,
            card.expected_runs_low,
            card.expected_runs_base,
            card.expected_runs_high
        );
    }
    Ok(())
}
