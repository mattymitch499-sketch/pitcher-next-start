import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import PaperTrading from './PaperTrading.svelte';
import PitcherDetail from './PitcherDetail.svelte';
import ValidationPage from './Validation.svelte';
import type { Card, ClusterStats, PaperEntry, PaperLog, ScoredSlate, Validation } from './types';

const invoke = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

const card = (id: number, name: string, team: string): Card => ({
  pitcher_id:id,pitcher_name:name,target_game_date:'2026-07-17',projected_fip:3.8,
  expected_runs_low:0.2,expected_runs_p25:1.1,expected_runs_base:2.3,expected_runs_p75:3.4,expected_runs_high:5.1,
  expected_strikeouts:6.1,expected_innings:5.6,expected_strikeouts_low:2,expected_strikeouts_high:10,
  expected_innings_low:3.2,expected_innings_high:7.1,expected_strikeouts_compositional:6.0,
  strikeout_over_probabilities:[{line:3.5,probability:.85},{line:4.5,probability:.72},{line:5.5,probability:.55},{line:6.5,probability:.38},{line:7.5,probability:.22},{line:8.5,probability:.11}],
  strikeout_probabilities:[.01,.02,.04,.08,.12,.15,.17,.15,.11,.08,.04,.02,.01,0,0,0,0,0,0,0,0,0,0,0,0,0],
  confidence:'medium',judgment:'Average outlook.',outlook_summary:'Average outlook.',
  team_model_input:2.3,starts_used:5,recent_k_rate:.25,recent_bb_rate:.07,recent_hr_per_9:1,recent_whip:1.1,
  recent_er_per_9:3.5,recent_ground_ball_pct:.44,recent_fastball_velocity:94,rest_days:5,home:true,park_factor:1,
  opponent_offense_factor:1,pitcher_team_name:team,opponent_team_name:'Opponent',venue_name:'Park',pitcher_hand:'R',drivers:[{label:'Neutral',impact:'neutral'}],
  season_form:{starts:12,k_rate:.24,bb_rate:.08,hr_per_9:1.1,whip:1.2,er_per_9:3.8},
  recent_starts:[{game_date:'2026-07-10',opponent_name:'Opponent',home:true,innings_pitched:6,earned_runs:2,strikeouts:7,walks:1,pitch_count:91,batters_faced:23,hits:5,home_runs:1,fip:3.2}],
  starter_status:'Probable',data_through:'2026-07-10',history_starts:20,data_warnings:[]
});

const metrics: Validation = {held_out_rows:100,held_out_dates:10,fip_mae:2,runs_mae:1.5,strikeouts_mae:1.8,innings_mae:.9,
  calibrated_runs_interval_coverage:.9,calibrated_strikeouts_interval_coverage:.9,calibrated_innings_interval_coverage:.9,
  league_fip_mae:2.1,pitcher_history_fip_mae:2.05,last_five_fip_mae:2.2,league_runs_mae:1.6,pitcher_history_runs_mae:1.55,last_five_runs_mae:1.7,
  league_strikeouts_mae:2,pitcher_history_strikeouts_mae:1.9,last_five_strikeouts_mae:2.1,league_innings_mae:1.1,pitcher_history_innings_mae:1,last_five_innings_mae:1.2,
  confidence_calibration:{high_max_normalized_epistemic_score:.1,medium_max_normalized_epistemic_score:.2,threshold_training_rows:70,training_period_dates:7,tiers_validated:true,minimum_audit_rows_per_tier:100,required_mae_ratio:.95},
  confidence_audit:{rows:35,dates:3,high:{rows:10,runs_mae:1.2},medium:{rows:10,runs_mae:1.5},low:{rows:10,runs_mae:1.8},short_history:{rows:5,runs_mae:2.1}}};
const slate: ScoredSlate = {cards:[card(1,'Alpha Pitcher','Brewers'),card(2,'Beta Pitcher','Cubs')],validation:metrics,model:{artifact_id:'pns-blr-v1-2025-09-28',schema_version:1,trained_through:'2025-09-28',training_rows:1000},generated_at:'2026-07-16T12:00:00Z'};

const paperEntry = (name: string, side: PaperEntry['side']): PaperEntry => ({
  market_ticker:`KXMLBKS-26JUL281910AAABBB-${name}-6`,event_date:'2026-07-28',pitcher_id:1,pitcher_name:name,
  threshold:6,model_probability:.55,yes_bid:40,yes_ask:45,side,entry_price_cents:side?45:null,edge:side?.1:null,
  recorded_at:'2026-07-28T12:00:00Z',actual_strikeouts:7,result_yes:true,profit_cents:side?55:null,voided:false,
  model_expected_strikeouts:6,model_sd_strikeouts:2,model_artifact_id:'m'
});

const clustering = (): ClusterStats => ({
  graded_legs:931,graded_starts:128,legs_per_start:7.3,effective_legs:453,design_effect:2.05,design_effect_icc:2.02,
  same_direction_rate:.42,contradictory_starts:0,
  calibration_gap:{point:.0087,low:-.038,high:.055},
  // Straddles zero: the app must say so rather than imply a positive edge.
  mean_start_pnl_cents:{point:.77,low:-31.9,high:36.2},
  brier:.151,log_score:.466,ece:.022,
  per_date:[{event_date:'2026-07-28',starts:27,legs:199,profit_cents:-731,calibration_gap:-.0015}],
  worst_start:{pitcher_id:2,pitcher_name:'Gavin Williams',event_date:'2026-07-28',legs:8,graded_legs:8,profit_cents:-239,staked_cents:239,all_same_direction:true,contradictory:false},
  best_start:null,date_sensitivity:[-9.1,16.3],
  // Interval entirely below zero: the market forecast better, and the UI must say so.
  skill:{legs:722,starts:99,model_brier:.15671,market_brier:.14566,advantage:{point:-.01105,low:-.02415,high:-.00072}},
  by_threshold:[
    {threshold:6,legs:128,starts:128,model_probability:.386,market_probability:.370,observed_rate:.375,observed_low:.29,observed_high:.46,model_error:.011,market_error:.005},
    {threshold:9,legs:78,starts:78,model_probability:.122,market_probability:null,observed_rate:.103,observed_low:null,observed_high:null,model_error:.019,market_error:null}
  ],
  arbitrage_starts:0
});

const paperLog = (): PaperLog => ({
  entries:[paperEntry('Traded Ace','yes'),paperEntry('Untraded Arm',null)],
  // 45c fill: ceil(7 * .45 * .55) = 2c taker fee, so net is 55 - 2 = 53.
  summary:{logged_markets:2,open_positions:0,graded_bets:1,wins:1,staked_cents:45,profit_cents:55,roi:1.2,fee_cents:2,net_profit_cents:53,net_roi:1.178,graded_logged:2,mean_model_probability:.55,observed_yes_rate:.5,last_recorded_at:'2026-07-28T12:00:00Z'},
  clustering:clustering()
});

describe('projection workflow', () => {
  beforeEach(() => invoke.mockReset());

  it('runs, filters, opens, navigates, returns, and exports a slate', async () => {
    invoke.mockImplementation((command: string) => command === 'score_date' ? Promise.resolve(slate) : Promise.resolve('/tmp/predictions.csv'));
    render(App);
    await fireEvent.click(screen.getByRole('button',{name:/run projections/i}));
    expect(await screen.findByText('Alpha Pitcher')).toBeTruthy();
    expect(screen.getByText('Beta Pitcher')).toBeTruthy();
    await fireEvent.input(screen.getByPlaceholderText(/pitcher, team/i), {target:{value:'Alpha'}});
    expect(screen.queryByText('Beta Pitcher')).toBeNull();
    await fireEvent.input(screen.getByPlaceholderText(/pitcher, team/i), {target:{value:''}});
    await fireEvent.click(screen.getAllByRole('button',{name:/view full analysis/i})[0]);
    expect(screen.getByRole('heading',{name:'Alpha Pitcher'})).toBeTruthy();
    await fireEvent.click(screen.getByRole('button',{name:/next/i}));
    expect(screen.getByRole('heading',{name:'Beta Pitcher'})).toBeTruthy();
    await fireEvent.click(screen.getByRole('button',{name:/back to slate/i}));
    await fireEvent.click(screen.getByRole('button',{name:/export predictions/i}));
    expect(await screen.findByText(/\/tmp\/predictions.csv/)).toBeTruthy();
  });

  it('explains when confidence tiers are withheld', () => {
    const withheld = structuredClone(metrics);
    withheld.confidence_calibration.tiers_validated = false;
    render(PitcherDetail, { card: card(1, 'Alpha Pitcher', 'Brewers'), validation: withheld, onback: vi.fn() });
    expect(screen.getByText(/high confidence is withheld/i)).toBeTruthy();
  });

  it('labels confidence as provisional when validation is unavailable', () => {
    render(PitcherDetail, { card: card(1, 'Alpha Pitcher', 'Brewers'), validation: null, onback: vi.fn() });
    expect(screen.getByText(/confidence calibration is unavailable/i)).toBeTruthy();
  });

  it('shows only placed trades and says plainly when the edge is undetermined', () => {
    render(PaperTrading, { log: paperLog(), loading: false, onupdate: vi.fn() });
    // The no-bet market is logged for calibration but must not clutter the table.
    expect(screen.getByText('Traded Ace')).toBeTruthy();
    expect(screen.queryByText('Untraded Arm')).toBeNull();
    // An interval straddling zero must not be presented as an edge.
    expect(screen.getByText(/no edge has been demonstrated yet/i)).toBeTruthy();
    expect(screen.getByText(/lower bound until more dates accumulate/i)).toBeTruthy();
  });

  it('says outright when the market is the better forecaster', () => {
    render(PaperTrading, { log: paperLog(), loading: false, onupdate: vi.fn() });
    expect(screen.getByText(/market is forecasting these better than the model/i)).toBeTruthy();
    // The ladder breakdown shows which rung each side won.
    expect(screen.getByText('Actually happened')).toBeTruthy();
    expect(screen.getByText('9+ Ks')).toBeTruthy();
    // Column header plus the 6+ rung's verdict, where the market was closer.
    expect(screen.getAllByText('Market').length).toBeGreaterThan(1);
  });

  it('does not claim a winner when the skill interval straddles zero', () => {
    const unclear = paperLog();
    unclear.clustering.skill = { legs: 722, starts: 99, model_brier: .15, market_brier: .15, advantage: { point: 0, low: -.01, high: .01 } };
    render(PaperTrading, { log: unclear, loading: false, onupdate: vi.fn() });
    expect(screen.getByText(/neither is a proven better forecaster yet/i)).toBeTruthy();
  });

  it('warns about positions that cannot both win', () => {
    const clashing = paperLog();
    clashing.clustering.contradictory_starts = 1;
    render(PaperTrading, { log: clashing, loading: false, onupdate: vi.fn() });
    expect(screen.getByText(/cannot both win/i)).toBeTruthy();
  });

  it('withholds an interval instead of faking one on thin data', () => {
    const thin = paperLog();
    thin.clustering.mean_start_pnl_cents = { point: 0.8, low: null, high: null };
    render(PaperTrading, { log: thin, loading: false, onupdate: vi.fn() });
    expect(screen.getByText(/not enough starts yet for an interval/i)).toBeTruthy();
  });

  it('renders missing confidence audit MAE without inventing a value', () => {
    const missingMae = structuredClone(metrics);
    missingMae.confidence_calibration.tiers_validated = false;
    missingMae.confidence_audit.high.runs_mae = null;
    render(ValidationPage, { metrics: missingMae, history: [], onreconcile: vi.fn() });
    expect(screen.getByRole('heading', { name: /confidence tier audit/i })).toBeTruthy();
    expect(screen.getAllByText(/not reliable on later games/i).length).toBeGreaterThan(0);
    expect(screen.getByText('—')).toBeTruthy();
  });
});
