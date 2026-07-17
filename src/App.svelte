<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import Learn from './Learn.svelte';
  import PitcherDetail from './PitcherDetail.svelte';
  import ValidationPage from './Validation.svelte';
  import type { Card, Validation, ScoredSlate, TrackedPrediction } from './types';

  let page: 'projections' | 'learn' | 'pitcher' | 'validation' = 'projections';
  let selectedCard: Card | null = null;
  let date = new Date().toISOString().slice(0, 10);
  let cards: Card[] = [];
  let validation: Validation | null = null;
  let error = '';
  let notice = '';
  let running = false;
  let generatedAt = '';
  let search = '';
  let team = 'all';
  let history: TrackedPrediction[] = [];
  let reconciling = false;

  async function score() {
    running = true; error = ''; notice = '';
    try { const result = await invoke<ScoredSlate>('score_date', { date }); cards = result.cards; validation = result.validation; generatedAt = result.generated_at; search = ''; team = 'all'; }
    catch (cause) { error = String(cause); }
    finally { running = false; }
  }

  async function exportCsv() {
    error = ''; notice = '';
    try { notice = `Exported to ${await invoke<string>('export_predictions', { cards })}`; }
    catch (cause) { error = String(cause); }
  }
  function openPitcher(card: Card) { selectedCard = card; page = 'pitcher'; window.scrollTo(0, 0); }
  function closePitcher() { page = 'projections'; selectedCard = null; window.scrollTo(0, 0); }
  function filteredCards() { const query = search.trim().toLowerCase(); return cards.filter(card => (team === 'all' || card.pitcher_team_name === team) && (!query || `${card.pitcher_name} ${card.pitcher_team_name} ${card.opponent_team_name}`.toLowerCase().includes(query))); }
  function teams() { return [...new Set(cards.map(card => card.pitcher_team_name))].sort(); }
  function adjacentCard(offset: number) { if (!selectedCard) return null; const index = cards.findIndex(card => card.pitcher_id === selectedCard?.pitcher_id); return cards[index + offset] ?? null; }
  function movePitcher(offset: number) { const card = adjacentCard(offset); if (card) { selectedCard = card; window.scrollTo(0, 0); } }
  async function openValidation() { page = 'validation'; try { history = await invoke<TrackedPrediction[]>('prediction_history'); } catch (cause) { error = String(cause); } window.scrollTo(0, 0); }
  async function reconcileHistory() { reconciling = true; error = ''; try { history = await invoke<TrackedPrediction[]>('reconcile_prediction_history'); } catch (cause) { error = String(cause); } finally { reconciling = false; } }
</script>

<div class="app-shell">
  <header class="topbar">
    <a class="brand" href="#projections" onclick={() => page = 'projections'} aria-label="Pitcher Next-Start home">
      <span class="brand-mark">P</span><span><b>Pitcher</b><small>Next Start</small></span>
    </a>
    <nav aria-label="Primary">
      <a class:active={page === 'projections' || page === 'pitcher'} href="#projections" onclick={() => page = 'projections'}>Projections</a>
      <a class:active={page === 'validation'} href="#validation" onclick={openValidation}>Validation</a>
      <a class:active={page === 'learn'} href="#learn" onclick={() => page = 'learn'}>Learn the models</a>
    </nav>
    <div class="model-status"><i></i><span>Bayesian models</span><b>Live MLB data</b></div>
  </header>

  {#if page === 'pitcher' && selectedCard}
    <PitcherDetail card={selectedCard} {validation} onback={closePitcher} onprevious={adjacentCard(-1) ? () => movePitcher(-1) : null} onnext={adjacentCard(1) ? () => movePitcher(1) : null} {generatedAt} />
  {:else if page === 'learn'}
    <Learn />
  {:else if page === 'validation'}
    <ValidationPage metrics={validation} {history} loading={reconciling} onreconcile={reconcileHistory} />
  {:else}
    <main class="workspace">
      <div class="page-heading">
        <div><p class="kicker">Next-start workspace</p><h1>Pitcher projections</h1><p>Score probable starters with recent form, matchup context, and visible uncertainty.</p></div>
        <button class="export-button" onclick={exportCsv} disabled={!cards.length} aria-label="Export predictions to CSV"><span>↓</span> Export CSV</button>
      </div>

      <section class="control-bar" aria-label="Projection controls">
        <div class="date-field"><label for="date">Slate date</label><input id="date" type="date" bind:value={date}></div>
        <div class="control-note"><span>Data source</span><b>MLB Stats API</b></div>
        <div class="control-note"><span>Rolling window</span><b>Last 5 starts</b></div>
        <button class="run-button" onclick={score} disabled={running}>{running ? 'Running model…' : 'Run projections'} <span>→</span></button>
      </section>

      {#if error}<div class="alert error"><b>Projection failed</b><span>{error}</span></div>{/if}
      {#if notice}<div class="alert success"><b>Export complete</b><span>{notice}</span></div>{/if}

      <div class="results-heading"><div><h2>Starter outlook</h2><span>{cards.length ? `${cards.length} pitchers scored` : 'Awaiting a model run'}</span></div><div class="legend"><span><i class="good"></i> Strong</span><span><i class="neutral"></i> Average</span><span><i class="risk"></i> Risk</span></div></div>

      {#if cards.length}<section class="slate-tools" aria-label="Filter projections"><label><span>Search</span><input type="search" bind:value={search} placeholder="Pitcher, team, or opponent"></label><label><span>Team</span><select bind:value={team}><option value="all">All teams</option>{#each teams() as teamName}<option value={teamName}>{teamName}</option>{/each}</select></label><p>{filteredCards().length} of {cards.length} shown</p></section>{/if}

      {#if validation}
        <section class="validation-strip" aria-label="Walk-forward validation"><div><span>Walk-forward sample</span><b>{validation.held_out_rows.toLocaleString()} starts · {validation.held_out_dates} dates</b></div><div><span>Model FIP MAE</span><b>{validation.fip_mae.toFixed(3)}</b></div><div><span>Best baseline</span><b>{Math.min(validation.league_fip_mae, validation.pitcher_history_fip_mae, validation.last_five_fip_mae).toFixed(3)}</b></div><div><span>Run-band coverage</span><b>{(validation.calibrated_runs_interval_coverage * 100).toFixed(1)}%</b></div></section>
      {/if}

      {#if cards.length}
        <section class="cards">
          {#each filteredCards() as card}
            <article class="projection-card compact-card">
              <div class="card-topline"><span>MLBAM {card.pitcher_id}</span><span class="confidence confidence-{card.confidence}">{card.confidence} confidence</span></div>
              <div class="pitcher"><div class="avatar">{card.pitcher_name.split(' ').map(part => part[0]).join('')}</div><div><h3>{card.pitcher_name}</h3><p>{card.pitcher_team_name} · {card.home ? 'vs' : '@'} {card.opponent_team_name}</p></div></div>
              <div class="compact-metrics"><div><span>Projected FIP</span><strong>{card.projected_fip.toFixed(2)}</strong></div><div><span>Expected runs</span><strong>{card.expected_runs_base.toFixed(1)}</strong><small>{card.expected_runs_low.toFixed(1)}–{card.expected_runs_high.toFixed(1)}</small></div></div>
              <p class="compact-read">{card.projected_fip < 3.7 ? 'Strong run-prevention projection.' : card.projected_fip < 4.5 ? 'League-average run-prevention projection.' : 'Elevated run-risk projection.'}</p>
              <button class="analysis-button" onclick={() => openPitcher(card)}>View full analysis <span>→</span></button>
            </article>
          {/each}
        </section>{#if !filteredCards().length}<div class="no-filter-results">No pitchers match these filters.</div>{/if}
      {:else}
        <section class="empty-state"><div class="empty-icon"><span></span><span></span><span></span></div><h3>No projections yet</h3><p>Select a slate date and run the model to generate starter cards.</p><button onclick={score}>Run for {date}</button></section>
      {/if}
    </main>
  {/if}
</div>
