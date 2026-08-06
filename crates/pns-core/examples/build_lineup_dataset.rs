//! Backfills the opposing starting nine onto every training start, and writes the
//! per-batter box-score lines the batter simulator folds over.
//!
//! One box score answers both starters of a game, so games — not rows — are the
//! unit of work: ~8k fetches for a ~16k-row dataset. Final box scores are cached
//! for 30 days, so a resumed run costs almost nothing.
//!
//! Usage: `cargo run -p pns-core --example build_lineup_dataset [-- --limit N]`.
//! `--limit N` processes only the first N games and writes `.sample` files, so the
//! pipeline can be smoke-tested without touching the real outputs.

use pns_core::data::batters::{BatterGameLine, BatterLineRow};
use pns_core::data::lineups::GameBoxscore;
use std::collections::{BTreeMap, BTreeSet};

/// Concurrent box-score fetches. The MLB feed needs no auth, so stay polite.
const CHUNK: usize = 10;

/// Below this share of starts carrying a full nine, the backfill is broken rather
/// than merely incomplete — mirrors the Savant coverage guard in `enrich_dataset`.
const MINIMUM_LINEUP_COVERAGE: f64 = 0.95;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let limit = parse_limit()?;
    // Never the v2 path: it is the frozen input of the shipped model.
    let (starts_out, lines_out) = match limit {
        Some(_) => (
            "data/processed/training_starts_v3.sample.csv",
            "data/processed/batter_lines.sample.csv",
        ),
        None => (
            "data/processed/training_starts_v3.csv",
            "data/processed/batter_lines.csv",
        ),
    };

    let mut rows =
        pns_core::data::dataset::read_csv(root.join("data/processed/training_starts.csv"))?;
    let mut games: Vec<u32> = rows
        .iter()
        .map(|row| row.game_pk)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if let Some(limit) = limit {
        games.truncate(limit);
        let selected: BTreeSet<u32> = games.iter().copied().collect();
        rows.retain(|row| selected.contains(&row.game_pk));
    }
    let input_rows = rows.len();
    eprintln!(
        "backfilling {input_rows} starts across {} games",
        games.len()
    );

    let client = pns_core::data::MlbStatsClient::new()?;
    let mut boxscores: BTreeMap<u32, GameBoxscore> = BTreeMap::new();
    // A postponed or suspended game can have no batting order at all. One such game
    // must not kill a 10k-game run, so failures are counted and skipped.
    let mut failures: Vec<String> = Vec::new();
    for (batch, chunk) in games.chunks(CHUNK).enumerate() {
        if batch % 25 == 0 {
            eprintln!(
                "box scores: {}/{} games ({} failed so far)",
                batch * CHUNK,
                games.len(),
                failures.len()
            );
        }
        let results: Vec<_> = std::thread::scope(|scope| {
            chunk
                .iter()
                .map(|&game_pk| {
                    let client = client.clone();
                    scope.spawn(move || (game_pk, client.game_boxscore(game_pk)))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("box score worker panicked"))
                .collect()
        });
        for (game_pk, result) in results {
            match result {
                Ok(boxscore) => {
                    boxscores.insert(game_pk, boxscore);
                }
                Err(error) => failures.push(format!("{game_pk}: {error}")),
            }
        }
    }

    // The lineup a starter faced is the *other* side of his game's box score.
    let mut team_mismatches = 0;
    for row in &mut rows {
        let Some(boxscore) = boxscores.get(&row.game_pk) else {
            continue;
        };
        let opponent = if boxscore.home.team_id == row.team_id {
            &boxscore.away
        } else if boxscore.away.team_id == row.team_id {
            &boxscore.home
        } else {
            team_mismatches += 1;
            continue;
        };
        row.opponent_batters = opponent
            .batting_order
            .iter()
            .map(|line| line.batter_id)
            .collect();
    }

    // Both starters of a game map to the same box score, so key by (game, batter)
    // and process each game once.
    let dates: BTreeMap<u32, _> = rows
        .iter()
        .map(|row| (row.game_pk, row.game_date))
        .collect();
    let mut lines: BTreeMap<(u32, u32), BatterLineRow> = BTreeMap::new();
    for (&game_pk, boxscore) in &boxscores {
        let Some(&game_date) = dates.get(&game_pk) else {
            continue;
        };
        for line in BatterGameLine::from_boxscore(boxscore, game_date) {
            lines.insert((game_pk, line.batter_id), BatterLineRow::new(game_pk, line));
        }
    }
    let lines: Vec<_> = lines.into_values().collect();

    let covered = rows
        .iter()
        .filter(|row| row.opponent_batters.len() == 9)
        .count();
    let coverage = covered as f64 / input_rows.max(1) as f64;
    eprintln!(
        "fetched {} of {} games ({} failed, {team_mismatches} team-id mismatches); \
         lineup coverage {:.2}%",
        boxscores.len(),
        games.len(),
        failures.len(),
        coverage * 100.0
    );
    for failure in failures.iter().take(20) {
        eprintln!("  failed: {failure}");
    }

    if rows.len() != input_rows {
        return Err(format!(
            "refusing to write: {} output rows from {input_rows} input rows",
            rows.len()
        )
        .into());
    }
    if coverage < MINIMUM_LINEUP_COVERAGE {
        return Err(format!(
            "refusing to write backfill with {:.2}% lineup coverage ({covered}/{input_rows} starts)",
            coverage * 100.0
        )
        .into());
    }

    pns_core::data::dataset::write_csv(&rows, root.join(starts_out))?;
    pns_core::data::batters::write_batter_lines(&lines, root.join(lines_out))?;
    let plate_appearances: u64 = lines.iter().map(|line| line.plate_appearances as u64).sum();
    let strikeouts: u64 = lines.iter().map(|line| line.strikeouts as u64).sum();
    println!(
        "wrote {} starts to {starts_out} and {} batter lines to {lines_out}\n\
         lineup coverage {:.2}%; league K/PA {:.4}; PA per batter-game {:.2}",
        rows.len(),
        lines.len(),
        coverage * 100.0,
        strikeouts as f64 / plate_appearances.max(1) as f64,
        plate_appearances as f64 / lines.len().max(1) as f64,
    );
    Ok(())
}

fn parse_limit() -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(arg) = args.next() else {
        return Ok(None);
    };
    let value = match arg.strip_prefix("--limit") {
        Some("") => args.next().ok_or("--limit needs a game count")?,
        Some(rest) => rest.trim_start_matches('=').to_string(),
        None => return Err(format!("unknown argument {arg}").into()),
    };
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument {extra}").into());
    }
    Ok(Some(value.parse::<usize>()?))
}
