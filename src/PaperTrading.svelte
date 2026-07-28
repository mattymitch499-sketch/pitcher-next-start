<script lang="ts">
  import type { PaperEntry, PaperLog } from './types';
  export let log: PaperLog | null;
  export let loading = false;
  export let onupdate: () => void;
  const dollars = (cents: number) => `${cents < 0 ? '-' : ''}$${Math.abs(cents / 100).toFixed(2)}`;
  const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  const sideLabel = (entry: PaperEntry) => entry.side == null ? 'Logged' : entry.side === 'yes' ? `YES @ ${entry.entry_price_cents}¢` : `NO @ ${entry.entry_price_cents}¢`;
  const resultLabel = (entry: PaperEntry) => entry.voided ? 'Scratched' : entry.result_yes == null ? 'Open' : `${entry.actual_strikeouts} K · ${entry.result_yes ? 'YES' : 'NO'}`;
  const profitLabel = (entry: PaperEntry) => entry.profit_cents == null ? '—' : dollars(entry.profit_cents);
  const recent = () => (log?.entries ?? []).slice().sort((a, b) => b.event_date.localeCompare(a.event_date) || a.pitcher_name.localeCompare(b.pitcher_name) || a.threshold - b.threshold).slice(0, 200);
</script>

<main class="validation-page">
  <header><p class="kicker">Market accountability</p><h1>Paper trading</h1><p>Simulated one-contract positions against Kalshi's pitcher strikeout markets. Nothing is ever wagered — this page measures whether the model's probabilities beat live market prices.</p></header>

  {#if log}
    <section class="validation-overview">
      <div><span>Paper P&amp;L</span><strong>{dollars(log.summary.profit_cents)}</strong><small>{log.summary.roi == null ? 'No graded bets yet' : `${(log.summary.roi * 100).toFixed(1)}% return on ${dollars(log.summary.staked_cents)} staked`}</small></div>
      <div><span>Record</span><strong>{log.summary.wins}–{log.summary.graded_bets - log.summary.wins}</strong><small>{log.summary.open_positions} open positions</small></div>
      <div><span>Markets logged</span><strong>{log.summary.logged_markets.toLocaleString()}</strong><small>{log.summary.graded_logged.toLocaleString()} settled</small></div>
      <div><span>Calibration</span><strong>{percent(log.summary.mean_model_probability)} vs {percent(log.summary.observed_yes_rate)}</strong><small>Mean model probability vs observed YES rate</small></div>
      <div><span>Last logged</span><strong>{log.summary.last_recorded_at ? log.summary.last_recorded_at.slice(0, 10) : '—'}</strong><small>Run the daily log every game day</small></div>
    </section>
  {/if}

  <section class="live-results">
    <div class="panel-title"><div><h2>Daily log</h2><p>Run once a day: records today's markets with model probabilities, opens paper positions where the edge is at least 5 points, and settles finished games.</p></div><button onclick={onupdate} disabled={loading}>{loading ? 'Scoring slate and fetching markets…' : "Log today's slate & grade"}</button></div>
    {#if log && log.entries.length}
      <div class="live-table paper-table">
        <div><b>Date</b><b>Pitcher</b><b>Market</b><b>Position</b><b>Model</b><b>Bid/Ask</b><b>Result</b><b>P/L</b></div>
        {#each recent() as entry}
          <div class:paper-bet={entry.side != null}>
            <span>{entry.event_date}</span>
            <strong>{entry.pitcher_name}</strong>
            <span>{entry.threshold}+ Ks</span>
            <span>{sideLabel(entry)}</span>
            <span>{percent(entry.model_probability)}</span>
            <span>{entry.yes_bid ?? '—'}/{entry.yes_ask ?? '—'}</span>
            <span>{resultLabel(entry)}</span>
            <span>{profitLabel(entry)}</span>
          </div>
        {/each}
      </div>
    {:else}
      <div class="validation-empty compact">No paper trades recorded yet. Run the daily log on a day with MLB games.</div>
    {/if}
  </section>
</main>
