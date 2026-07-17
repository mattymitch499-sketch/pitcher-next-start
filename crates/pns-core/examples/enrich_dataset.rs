use std::collections::{BTreeSet, HashMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("data/processed/training_starts.csv");
    let mut rows = pns_core::data::dataset::read_csv(&path)?;
    let all_pitcher_ids: Vec<u32> = rows
        .iter()
        .map(|row| row.pitcher_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let client = pns_core::data::MlbStatsClient::new()?;

    let mut hands = HashMap::new();
    for chunk in all_pitcher_ids.chunks(100) {
        hands.extend(client.pitcher_hands(chunk)?);
    }
    pns_core::data::enrich::enrich_historical_context(&mut rows, &hands);

    let start = rows
        .iter()
        .map(|row| row.game_date)
        .min()
        .ok_or("dataset is empty")?;
    let end = rows
        .iter()
        .map(|row| row.game_date)
        .max()
        .ok_or("dataset is empty")?;
    let mut failed = 0usize;
    let requested_ids: Vec<u32> = std::env::args()
        .skip(1)
        .map(|value| value.parse())
        .collect::<Result<_, _>>()?;
    let pitcher_ids = if requested_ids.is_empty() {
        all_pitcher_ids
    } else {
        requested_ids
    };
    for chunk in pitcher_ids.chunks(8) {
        let results: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = chunk
                .iter()
                .map(|&pitcher_id| {
                    let client = client.clone();
                    scope.spawn(move || {
                        (
                            pitcher_id,
                            client.savant_fastball_velocity(pitcher_id, start, end),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("Savant worker panicked"))
                .collect()
        });
        for (pitcher_id, result) in results {
            match result {
                Ok(velocities) => {
                    pns_core::data::enrich::join_velocity(&mut rows, pitcher_id, &velocities)
                }
                Err(error) => {
                    failed += 1;
                    eprintln!("velocity enrichment failed for {pitcher_id}: {error}");
                }
            }
        }
    }
    pns_core::data::dataset::write_csv(&rows, &path)?;
    let velocity_rows = rows.iter().filter(|row| row.avg_fb_velo.is_some()).count();
    println!(
        "enriched {} rows; velocity available for {} ({:.1}%); {} pitcher requests failed",
        rows.len(),
        velocity_rows,
        velocity_rows as f64 * 100.0 / rows.len() as f64,
        failed
    );
    Ok(())
}
