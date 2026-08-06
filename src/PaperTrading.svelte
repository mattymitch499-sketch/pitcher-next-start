<script lang="ts">
  import type { Interval, PaperEntry, PaperLog } from './types';
  export let log: PaperLog | null;
  export let loading = false;
  export let onupdate: () => void;
  const dollars = (cents: number) => `${cents < 0 ? '-' : ''}$${Math.abs(cents / 100).toFixed(2)}`;
  const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  // Per-start P&L is a few cents, so cents read better than dollars here.
  const cents = (value: number | null | undefined) => value == null ? '—' : `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(1)}¢`;
  const points = (value: number | null | undefined) => value == null ? '—' : `${value < 0 ? '−' : '+'}${Math.abs(value * 100).toFixed(1)}pp`;
  const round = (value: number | null | undefined, places = 2) => value == null ? '—' : value.toFixed(places);
  // An interval straddling zero means the sign of the effect is undetermined.
  const spansZero = (interval: Interval | null | undefined) => interval?.low != null && interval?.high != null && interval.low <= 0 && interval.high >= 0;
  const band = (interval: Interval | null | undefined, format: (value: number) => string) =>
    interval?.low == null || interval?.high == null ? 'Not enough starts yet for an interval' : `95% CI ${format(interval.low)} to ${format(interval.high)}`;
  const sideLabel = (entry: PaperEntry) => entry.side == null ? 'Logged' : entry.side === 'yes' ? `YES @ ${entry.entry_price_cents}¢` : `NO @ ${entry.entry_price_cents}¢`;
  const resultLabel = (entry: PaperEntry) => entry.voided ? 'Scratched' : entry.result_yes == null ? 'Open' : `${entry.actual_strikeouts} K · ${entry.result_yes ? 'YES' : 'NO'}`;
  const profitLabel = (entry: PaperEntry) => entry.profit_cents == null ? '—' : dollars(entry.profit_cents);
  const recent = () => (log?.entries ?? []).filter((entry) => entry.side != null).slice().sort((a, b) => b.event_date.localeCompare(a.event_date) || a.pitcher_name.localeCompare(b.pitcher_name) || a.threshold - b.threshold).slice(0, 200);
</script>

<main class="validation-page">
  <header><p class="kicker">Market accountability</p><h1>Paper trading</h1><p>Simulated one-contract positions against Kalshi's pitcher strikeout markets. Nothing is ever wagered — this page measures whether the model's probabilities beat live market prices.</p></header>

  {#if log}
    <section class="validation-overview">
      <div><span>Paper P&amp;L (net)</span><strong>{dollars(log.summary.net_profit_cents)}</strong><small>{log.summary.net_roi == null ? 'No graded bets yet' : `${(log.summary.net_roi * 100).toFixed(1)}% return on ${dollars(log.summary.staked_cents)} staked`}</small></div>
      <div><span>Kalshi fees</span><strong>−{dollars(log.summary.fee_cents)}</strong><small>{dollars(log.summary.profit_cents)} gross before fees</small></div>
      <div><span>Record</span><strong>{log.summary.wins}–{log.summary.graded_bets - log.summary.wins}</strong><small>{log.summary.open_positions} open positions</small></div>
      <div><span>Markets logged</span><strong>{log.summary.logged_markets.toLocaleString()}</strong><small>{log.summary.graded_logged.toLocaleString()} settled</small></div>
      <div><span>Calibration</span><strong>{percent(log.summary.mean_model_probability)} vs {percent(log.summary.observed_yes_rate)}</strong><small>Mean model probability vs observed YES rate</small></div>
      <div><span>Last logged</span><strong>{log.summary.last_recorded_at ? log.summary.last_recorded_at.slice(0, 10) : '—'}</strong><small>Run the daily log every game day</small></div>
    </section>
  {/if}

  {#if log && log.clustering && log.clustering.graded_starts > 0}
    {@const stats = log.clustering}
    <section class="reality-check">
      <div class="panel-title"><div><h2>Statistical reality check</h2><p>Every bet on one pitcher-start resolves together, so the tiles above count far more observations than actually exist. These numbers count starts.</p></div></div>

      {#if stats.skill}
        {@const beat = stats.skill.market_brier - stats.skill.model_brier}
        <div class="reality-headline" class:reality-inconclusive={spansZero(stats.skill.advantage)} class:reality-bad={stats.skill.advantage?.high != null && stats.skill.advantage.high < 0}>
          <span>Model vs market as forecasters</span>
          <strong>{beat > 0 ? 'Model ahead' : 'Market ahead'} by {Math.abs(beat).toFixed(4)} Brier</strong>
          <small>
            Model {stats.skill.model_brier.toFixed(4)} · Market {stats.skill.market_brier.toFixed(4)} over {stats.skill.legs.toLocaleString()} settled legs — lower is better.
            {band(stats.skill.advantage, (value) => value.toFixed(4))}{spansZero(stats.skill.advantage) ? ' — spans zero, so neither is a proven better forecaster yet' : stats.skill.advantage?.high != null && stats.skill.advantage.high < 0 ? ' — entirely below zero: the market is forecasting these better than the model.' : ' — entirely above zero: the model is genuinely beating the price.'}
          </small>
        </div>
      {/if}

      <div class="reality-headline" class:reality-inconclusive={spansZero(stats.mean_start_pnl_cents)}>
        <span>Mean profit per start</span>
        <strong>{cents(stats.mean_start_pnl_cents?.point)}</strong>
        <small>{band(stats.mean_start_pnl_cents, cents)}{spansZero(stats.mean_start_pnl_cents) ? ' — spans zero, so no edge has been demonstrated yet' : ''}</small>
      </div>

      <div class="reality-grid">
        <div><span>Sample</span><strong>{stats.graded_starts} starts</strong><small>{stats.graded_legs.toLocaleString()} settled legs · {round(stats.legs_per_start, 1)} per start</small></div>
        <div><span>Effective sample</span><strong>≈{stats.effective_legs == null ? '—' : Math.round(stats.effective_legs).toLocaleString()} legs</strong><small>{stats.design_effect == null ? 'Not enough data' : `${stats.graded_legs.toLocaleString()} legs carry this much information (${round(stats.design_effect, 1)}× inflation)`}</small></div>
        <div><span>Calibration gap</span><strong>{points(stats.calibration_gap?.point)}</strong><small>{band(stats.calibration_gap, points)}</small></div>
        <div><span>Scoring</span><strong>Brier {round(stats.brier, 3)}</strong><small>Log {round(stats.log_score, 3)} · ECE {round(stats.ece, 3)}</small></div>
      </div>

      {#if stats.by_threshold.length}
        <div class="live-table ladder-table">
          <div><b>Line</b><b>Legs</b><b>Model</b><b>Market</b><b>Actually happened</b><b>Closer</b></div>
          {#each stats.by_threshold as rung}
            <div>
              <span>{rung.threshold}+ Ks</span>
              <span>{rung.legs}</span>
              <span>{percent(rung.model_probability)}</span>
              <span>{percent(rung.market_probability)}</span>
              <span>{percent(rung.observed_rate)}{rung.observed_low != null && rung.observed_high != null ? ` (${percent(rung.observed_low)}–${percent(rung.observed_high)})` : ''}</span>
              {#if rung.market_error == null}
                <span>—</span>
              {:else if rung.model_error < rung.market_error}
                <strong>Model</strong>
              {:else}
                <span>Market</span>
              {/if}
            </div>
          {/each}
        </div>
        <p class="reality-note reality-quiet">Where the model is closer than the market, that rung is where an edge could live. Rungs with few legs are noise — read the whole shape, not one row.</p>
      {/if}

      {#if stats.per_date.length}
        <div class="live-table date-table">
          <div><b>Date</b><b>Starts</b><b>Legs</b><b>P/L</b><b>Calibration gap</b></div>
          {#each stats.per_date as day}
            <div>
              <span>{day.event_date}</span>
              <span>{day.starts}</span>
              <span>{day.legs}</span>
              <strong>{dollars(day.profit_cents)}</strong>
              <span>{points(day.calibration_gap)}</span>
            </div>
          {/each}
        </div>
      {/if}

      <p class="reality-note">
        {#if stats.same_direction_rate != null}<b>{percent(stats.same_direction_rate)}</b> of multi-bet starts had every leg resolve the same way — that clustering is what shrinks the effective sample.{/if}
        {#if stats.worst_start}Worst start: {stats.worst_start.pitcher_name} on {stats.worst_start.event_date}, {dollars(stats.worst_start.profit_cents)} across {stats.worst_start.legs} legs.{/if}
        {#if stats.date_sensitivity}Dropping any single date moves the per-start mean to somewhere between {cents(stats.date_sensitivity[0])} and {cents(stats.date_sensitivity[1])} — a range, not a confidence interval.{/if}
      </p>
      {#if stats.contradictory_starts > 0}
        <p class="reality-note reality-warn"><b>{stats.contradictory_starts}</b> historical start{stats.contradictory_starts === 1 ? ' holds' : 's hold'} a NO at a low threshold together with a YES at a higher one. Those cannot both win, so at least one leg was a guaranteed loss. New slates no longer place these.</p>
      {/if}
      {#if stats.arbitrage_starts > 0}
        <p class="reality-note reality-warn"><b>{stats.arbitrage_starts}</b> start{stats.arbitrage_starts === 1 ? '' : 's'} had a ladder where a lower line's ask sat below a higher line's bid — buying YES low and NO high there costs under $1 and pays at least $1, risk-free.</p>
      {/if}
      <p class="reality-note reality-quiet">Intervals cluster by pitcher-start only. Starts on the same date share a model version and a league-wide run environment, so real uncertainty is wider than shown — treat these as a lower bound until more dates accumulate.</p>
    </section>
  {/if}

  <section class="live-results">
    <div class="panel-title"><div><h2>Daily log</h2><p>Run once a day: records today's markets with model probabilities, opens paper positions where the edge is at least 5 points, and settles finished games. The <b>Sim</b> column is the per-plate-appearance simulator's read on the same market — logged for comparison only, never used to size or place a position, and blank until the opposing lineup is posted. Row-level P/L is gross; Kalshi's fee is charged per fill and is netted out in the summary above.</p></div><button onclick={onupdate} disabled={loading}>{loading ? 'Scoring slate and fetching markets…' : "Log today's slate & grade"}</button></div>
    {#if log && log.entries.length}
      <div class="live-table paper-table">
        <div><b>Date</b><b>Pitcher</b><b>Market</b><b>Position</b><b>Model</b><b>Sim</b><b>Bid/Ask</b><b>Result</b><b>P/L gross</b></div>
        {#each recent() as entry}
          <div class:paper-bet={entry.side != null}>
            <span>{entry.event_date}</span>
            <strong>{entry.pitcher_name}</strong>
            <span>{entry.threshold}+ Ks</span>
            <span>{sideLabel(entry)}</span>
            <span>{percent(entry.model_probability)}</span>
            <span>{entry.sim_probability == null ? '—' : percent(entry.sim_probability)}</span>
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
