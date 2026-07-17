import { fireEvent, render, screen } from '@testing-library/svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.svelte';
import PitcherDetail from './PitcherDetail.svelte';
import ValidationPage from './Validation.svelte';
import type { Card, ScoredSlate, Validation } from './types';

const invoke = vi.fn();
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

const card = (id: number, name: string, team: string): Card => ({
  pitcher_id:id,pitcher_name:name,target_game_date:'2026-07-17',projected_fip:3.8,
  expected_runs_low:0.2,expected_runs_p25:1.1,expected_runs_base:2.3,expected_runs_p75:3.4,expected_runs_high:5.1,
  expected_strikeouts:6.1,expected_innings:5.6,expected_strikeouts_low:2,expected_strikeouts_high:10,
  expected_innings_low:3.2,expected_innings_high:7.1,confidence:'medium',judgment:'Average outlook.',outlook_summary:'Average outlook.',
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
const slate: ScoredSlate = {cards:[card(1,'Alpha Pitcher','Brewers'),card(2,'Beta Pitcher','Cubs')],validation:metrics,generated_at:'2026-07-16T12:00:00Z'};

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

  it('renders missing confidence audit MAE without inventing a value', () => {
    const missingMae = structuredClone(metrics);
    missingMae.confidence_calibration.tiers_validated = false;
    missingMae.confidence_audit.high.runs_mae = null;
    render(ValidationPage, { metrics: missingMae, history: [], onreconcile: vi.fn() });
    expect(screen.getByRole('heading', { name: /confidence tier audit/i })).toBeTruthy();
    expect(screen.getByText('Gate failed · high withheld')).toBeTruthy();
    expect(screen.getByText('—')).toBeTruthy();
  });
});
