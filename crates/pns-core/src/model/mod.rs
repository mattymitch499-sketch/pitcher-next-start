use crate::features::FeatureRow;
use crate::{PnsError, Projection, Result};

pub mod artifact;
pub mod whiff;

/// Normal-Inverse-Gamma posterior. `covariance` stores the noise-scaled-out
/// matrix V_n, so the predictive variance is `noise * (1 + x'V_n x)` where
/// `noise = noise_rate / noise_shape` is the posterior mean of sigma^2.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Posterior {
    mean: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    noise_shape: f64,
    noise_rate: f64,
}

impl Posterior {
    fn noise_variance(&self) -> f64 {
        self.noise_rate / self.noise_shape
    }
}

pub const MAX_STRIKEOUTS: usize = 25;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BayesianLinearModel {
    fip: Posterior,
    runs: Posterior,
    strikeouts: Posterior,
    innings: Posterior,
    k_rate: Posterior,
    batters_faced: Posterior,
    interval_scale: f64,
    strikeouts_interval_scale: f64,
    innings_interval_scale: f64,
    k_concentration: f64,
    bf_residual_quantiles: Vec<f64>,
    feature_means: Vec<f64>,
    feature_scales: Vec<f64>,
    training_row_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationMetrics {
    pub held_out_rows: usize,
    pub held_out_dates: usize,
    pub fip_mae: f64,
    pub runs_mae: f64,
    pub strikeouts_mae: f64,
    pub innings_mae: f64,
    pub strikeouts_interval_coverage: f64,
    pub innings_interval_coverage: f64,
    pub calibrated_strikeouts_interval_coverage: f64,
    pub calibrated_innings_interval_coverage: f64,
    pub recommended_strikeouts_interval_scale: f64,
    pub recommended_innings_interval_scale: f64,
    pub league_strikeouts_mae: f64,
    pub pitcher_history_strikeouts_mae: f64,
    pub last_five_strikeouts_mae: f64,
    pub league_innings_mae: f64,
    pub pitcher_history_innings_mae: f64,
    pub last_five_innings_mae: f64,
    pub runs_interval_coverage: f64,
    pub calibrated_runs_interval_coverage: f64,
    pub recommended_interval_scale: f64,
    pub league_fip_mae: f64,
    pub pitcher_history_fip_mae: f64,
    pub last_five_fip_mae: f64,
    pub league_runs_mae: f64,
    pub pitcher_history_runs_mae: f64,
    pub last_five_runs_mae: f64,
    pub confidence_calibration: ConfidenceCalibration,
    pub confidence_audit: ConfidenceAudit,
    pub k_distribution: KDistributionCalibration,
}

/// Calibration of the discrete strikeout distribution. The beta-binomial
/// concentration and batters-faced residual spread are selected on the earlier
/// 70% of held-out dates; the audit fields grade the later 30% only.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KDistributionCalibration {
    pub concentration: f64,
    pub bf_residual_quantiles: Vec<f64>,
    pub training_rows: usize,
    pub audit_rows: usize,
    pub audit_log_score: Option<f64>,
    pub gaussian_baseline_log_score: Option<f64>,
    pub compositional_strikeouts_mae: Option<f64>,
    pub direct_strikeouts_mae: Option<f64>,
    pub line_reliability: Vec<KLineReliability>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct KLineReliability {
    pub line: f64,
    pub rows: usize,
    pub mean_predicted_over: f64,
    pub observed_over_rate: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ConfidenceCalibration {
    pub high_max_normalized_epistemic_score: f64,
    pub medium_max_normalized_epistemic_score: f64,
    pub threshold_training_rows: usize,
    pub training_period_dates: usize,
    pub tiers_validated: bool,
    pub minimum_audit_rows_per_tier: usize,
    pub required_mae_ratio: f64,
}

impl ConfidenceCalibration {
    pub fn has_valid_thresholds(&self) -> bool {
        self.high_max_normalized_epistemic_score.is_finite()
            && self.medium_max_normalized_epistemic_score.is_finite()
            && self.high_max_normalized_epistemic_score >= 0.0
            && self.medium_max_normalized_epistemic_score >= 0.0
            && self.high_max_normalized_epistemic_score
                <= self.medium_max_normalized_epistemic_score
    }

    pub fn epistemic_score(
        &self,
        runs_epistemic_sd: f64,
        model_training_rows: usize,
        starts_used: usize,
    ) -> f64 {
        runs_epistemic_sd
            * (model_training_rows.max(1) as f64).sqrt()
            * (5.0 / starts_used.max(1) as f64).sqrt()
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct ConfidenceTierAudit {
    pub rows: usize,
    pub runs_mae: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct ConfidenceAudit {
    pub rows: usize,
    pub dates: usize,
    pub high: ConfidenceTierAudit,
    pub medium: ConfidenceTierAudit,
    pub low: ConfidenceTierAudit,
    /// Later-period rows with fewer than five prior starts. These are audited
    /// separately because they cannot validate thresholds trained on full histories.
    pub short_history: ConfidenceTierAudit,
}

#[derive(Debug, Clone, Copy)]
struct ConfidenceObservation {
    date: chrono::NaiveDate,
    runs_epistemic_sd: f64,
    starts_used: usize,
    model_training_rows: usize,
    absolute_error: f64,
}

#[derive(Debug, Clone, Copy)]
struct KObservation {
    date: chrono::NaiveDate,
    k_rate_mean: f64,
    batters_faced_mean: f64,
    direct_mean: f64,
    direct_sd: f64,
    actual_strikeouts: f64,
    actual_batters: f64,
}

impl BayesianLinearModel {
    pub fn fit(rows: &[FeatureRow]) -> Result<Self> {
        if rows.len() < 4 {
            return Err(PnsError::Model(
                "at least four historical rows are required".into(),
            ));
        }
        let (standardized, feature_means, feature_scales) = standardize_rows(rows)?;
        Ok(Self {
            fip: fit_target_nig(&standardized, |r| r.target_fip, 4.20, 2.25)?,
            runs: fit_target_nig(&standardized, |r| r.target_runs, 3.0, 2.25)?,
            strikeouts: fit_target_nig(&standardized, |r| r.target_strikeouts, 5.2, 6.25)?,
            innings: fit_target_nig(&standardized, |r| r.target_innings, 5.3, 1.0)?,
            k_rate: fit_target_nig(&standardized, |r| r.target_k_rate, 0.22, 0.01)?,
            batters_faced: fit_target_nig(&standardized, |r| r.target_batters_faced, 24.0, 25.0)?,
            interval_scale: 1.0,
            strikeouts_interval_scale: 1.0,
            innings_interval_scale: 1.0,
            k_concentration: 150.0,
            bf_residual_quantiles: vec![0.0],
            feature_means,
            feature_scales,
            training_row_count: rows.len(),
        })
    }

    pub fn with_interval_scale(mut self, scale: f64) -> Self {
        self.interval_scale = scale.clamp(0.75, 2.0);
        self
    }

    pub fn with_aux_interval_scales(mut self, strikeouts: f64, innings: f64) -> Self {
        self.strikeouts_interval_scale = strikeouts.clamp(0.75, 3.0);
        self.innings_interval_scale = innings.clamp(0.75, 3.0);
        self
    }

    pub fn with_k_distribution(mut self, calibration: &KDistributionCalibration) -> Self {
        self.k_concentration = calibration.concentration.clamp(10.0, 5000.0);
        if !calibration.bf_residual_quantiles.is_empty() {
            self.bf_residual_quantiles = calibration.bf_residual_quantiles.clone();
        }
        self
    }

    pub fn score(&self, row: &FeatureRow) -> Result<Projection> {
        let values = standardize_values(&row.values, &self.feature_means, &self.feature_scales)?;
        let (fip, fip_sd, _) = predict(&self.fip, &values)?;
        let (runs, runs_sd, runs_epistemic_sd) = predict(&self.runs, &values)?;
        let (strikeouts, strikeouts_sd, _) = predict(&self.strikeouts, &values)?;
        let (innings, innings_sd, _) = predict(&self.innings, &values)?;
        let (k_rate_mean, _, _) = predict(&self.k_rate, &values)?;
        let (batters_faced_mean, _, _) = predict(&self.batters_faced, &values)?;
        let strikeout_probabilities = strikeout_pmf(
            k_rate_mean,
            batters_faced_mean,
            &self.bf_residual_quantiles,
            self.k_concentration,
        );
        let expected_strikeouts_compositional = strikeout_probabilities
            .iter()
            .enumerate()
            .map(|(k, probability)| k as f64 * probability)
            .sum::<f64>();
        let half_width = 1.645 * runs_sd * self.interval_scale;
        let low = (runs - half_width).max(0.0);
        let high = (runs + half_width).max(low + 0.1);
        let quartile_width = 0.67449 * runs_sd * self.interval_scale;
        let strikeouts_half_width = 1.645 * strikeouts_sd * self.strikeouts_interval_scale;
        let innings_half_width = 1.645 * innings_sd * self.innings_interval_scale;
        Ok(Projection {
            projected_fip: fip.clamp(1.5, 7.5),
            expected_runs_low: low,
            expected_runs_p25: (runs - quartile_width).max(0.0),
            expected_runs_base: runs.max(0.0),
            expected_runs_p75: (runs + quartile_width).max(0.0),
            expected_runs_high: high,
            posterior_width: 2.0 * half_width,
            starts_used: row.starts_used,
            fip_posterior_sd: fip_sd,
            runs_posterior_sd: runs_sd,
            runs_epistemic_sd,
            model_training_rows: self.training_row_count,
            expected_strikeouts: strikeouts.clamp(0.0, 15.0),
            expected_innings: innings.clamp(1.0, 9.0),
            expected_strikeouts_low: (strikeouts - strikeouts_half_width).max(0.0),
            expected_strikeouts_high: (strikeouts + strikeouts_half_width).clamp(0.1, 20.0),
            expected_innings_low: (innings - innings_half_width).max(0.0),
            expected_innings_high: (innings + innings_half_width).clamp(0.1, 9.0),
            strikeouts_posterior_sd: strikeouts_sd,
            innings_posterior_sd: innings_sd,
            k_rate_mean,
            batters_faced_mean,
            expected_strikeouts_compositional,
            strikeout_probabilities,
        })
    }

    fn validate(&self, feature_width: usize) -> Result<()> {
        if self.feature_means.len() != feature_width
            || self.feature_scales.len() != feature_width
            || self.training_row_count < 4
        {
            return Err(PnsError::Model(
                "artifact model metadata does not match the feature contract".into(),
            ));
        }
        if self
            .feature_means
            .iter()
            .chain(&self.feature_scales)
            .any(|value| !value.is_finite())
            || self.feature_scales.iter().any(|scale| *scale <= 0.0)
            || !self.interval_scale.is_finite()
            || !self.strikeouts_interval_scale.is_finite()
            || !self.innings_interval_scale.is_finite()
            || !(0.75..=2.0).contains(&self.interval_scale)
            || !(0.75..=3.0).contains(&self.strikeouts_interval_scale)
            || !(0.75..=3.0).contains(&self.innings_interval_scale)
            || !(10.0..=5000.0).contains(&self.k_concentration)
            || self.bf_residual_quantiles.is_empty()
            || self.bf_residual_quantiles.len() > 99
            || self
                .bf_residual_quantiles
                .iter()
                .any(|value| !value.is_finite())
            || self
                .bf_residual_quantiles
                .windows(2)
                .any(|pair| pair[0] > pair[1])
        {
            return Err(PnsError::Model(
                "artifact model contains invalid standardization or interval values".into(),
            ));
        }
        for posterior in [
            &self.fip,
            &self.runs,
            &self.strikeouts,
            &self.innings,
            &self.k_rate,
            &self.batters_faced,
        ] {
            if posterior.mean.len() != feature_width
                || posterior.covariance.len() != feature_width
                || posterior
                    .covariance
                    .iter()
                    .any(|row| row.len() != feature_width)
                || !posterior.noise_shape.is_finite()
                || !posterior.noise_rate.is_finite()
                || posterior.noise_shape <= 0.0
                || posterior.noise_rate <= 0.0
                || posterior.mean.iter().any(|value| !value.is_finite())
                || posterior
                    .covariance
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
                || posterior
                    .covariance
                    .iter()
                    .enumerate()
                    .any(|(index, row)| row[index] < 0.0)
                || (0..feature_width).any(|row| {
                    (0..feature_width).any(|column| {
                        let left = posterior.covariance[row][column];
                        let right = posterior.covariance[column][row];
                        (left - right).abs() > 1e-10 * left.abs().max(right.abs()).max(1.0)
                    })
                })
            {
                return Err(PnsError::Model(
                    "artifact posterior dimensions or values are invalid".into(),
                ));
            }
        }
        Ok(())
    }
}

impl ValidationMetrics {
    fn validate(&self) -> Result<()> {
        let metrics = [
            self.fip_mae,
            self.runs_mae,
            self.strikeouts_mae,
            self.innings_mae,
            self.recommended_interval_scale,
            self.recommended_strikeouts_interval_scale,
            self.recommended_innings_interval_scale,
            self.league_fip_mae,
            self.pitcher_history_fip_mae,
            self.last_five_fip_mae,
            self.league_runs_mae,
            self.pitcher_history_runs_mae,
            self.last_five_runs_mae,
            self.league_strikeouts_mae,
            self.pitcher_history_strikeouts_mae,
            self.last_five_strikeouts_mae,
            self.league_innings_mae,
            self.pitcher_history_innings_mae,
            self.last_five_innings_mae,
        ];
        let coverage = [
            self.runs_interval_coverage,
            self.calibrated_runs_interval_coverage,
            self.strikeouts_interval_coverage,
            self.calibrated_strikeouts_interval_coverage,
            self.innings_interval_coverage,
            self.calibrated_innings_interval_coverage,
        ];
        if self.held_out_rows == 0
            || self.held_out_dates == 0
            || metrics
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || coverage
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || self.confidence_audit.rows
                != self.confidence_audit.high.rows
                    + self.confidence_audit.medium.rows
                    + self.confidence_audit.low.rows
            || [
                self.confidence_audit.high.runs_mae,
                self.confidence_audit.medium.runs_mae,
                self.confidence_audit.low.runs_mae,
                self.confidence_audit.short_history.runs_mae,
            ]
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite() || value < 0.0)
        {
            return Err(PnsError::Model(
                "model artifact validation metrics are internally inconsistent".into(),
            ));
        }
        let k = &self.k_distribution;
        if !(10.0..=5000.0).contains(&k.concentration)
            || k.bf_residual_quantiles.is_empty()
            || k.bf_residual_quantiles.len() > 99
            || k.bf_residual_quantiles
                .iter()
                .any(|value| !value.is_finite())
            || k.bf_residual_quantiles
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || k.training_rows == 0
            || [k.audit_log_score, k.gaussian_baseline_log_score]
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite())
            || [k.compositional_strikeouts_mae, k.direct_strikeouts_mae]
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite() || value < 0.0)
            || k.line_reliability.iter().any(|reliability| {
                !reliability.line.is_finite()
                    || !(0.0..=1.0).contains(&reliability.mean_predicted_over)
                    || !(0.0..=1.0).contains(&reliability.observed_over_rate)
            })
        {
            return Err(PnsError::Model(
                "model artifact strikeout distribution calibration is invalid".into(),
            ));
        }
        Ok(())
    }
}

fn standardize_rows(rows: &[FeatureRow]) -> Result<(Vec<FeatureRow>, Vec<f64>, Vec<f64>)> {
    let width = rows
        .first()
        .ok_or_else(|| PnsError::Model("training rows are empty".into()))?
        .values
        .len();
    if rows.iter().any(|row| row.values.len() != width) {
        return Err(PnsError::Model("inconsistent feature widths".into()));
    }
    let mut means = vec![0.0; width];
    let mut scales = vec![1.0; width];
    for feature in 1..width {
        let observed: Vec<_> = rows
            .iter()
            .map(|row| row.values[feature])
            .filter(|value| feature != 7 || *value > 0.0)
            .collect();
        if observed.is_empty() {
            continue;
        }
        means[feature] = observed.iter().sum::<f64>() / observed.len() as f64;
        let variance = observed
            .iter()
            .map(|value| (*value - means[feature]).powi(2))
            .sum::<f64>()
            / observed.len() as f64;
        scales[feature] = if variance < 1e-8 {
            1.0
        } else {
            variance.sqrt()
        };
    }
    let standardized = rows
        .iter()
        .cloned()
        .map(|mut row| {
            row.values =
                standardize_values(&row.values, &means, &scales).expect("validated feature width");
            row
        })
        .collect();
    Ok((standardized, means, scales))
}

fn standardize_values(values: &[f64], means: &[f64], scales: &[f64]) -> Result<Vec<f64>> {
    if values.len() != means.len() || values.len() != scales.len() {
        return Err(PnsError::Model(
            "feature width differs from standardizer".into(),
        ));
    }
    Ok(values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            if index == 0 {
                *value
            } else if index == 7 && *value <= 0.0 {
                0.0
            } else {
                (*value - means[index]) / scales[index]
            }
        })
        .collect())
}

/// Walk-forward validation. Every test date is scored using only earlier dates.
pub fn walk_forward_validate(
    rows: &[FeatureRow],
    minimum_training_rows: usize,
) -> Result<ValidationMetrics> {
    if rows.len() < 10 {
        return Err(PnsError::Model(
            "at least ten rows are required for time-split validation".into(),
        ));
    }
    let mut ordered = rows.to_vec();
    ordered.sort_by_key(|row| row.as_of);
    let mut fip_error = 0.0;
    let mut runs_error = 0.0;
    let mut strikeouts_error = 0.0;
    let mut innings_error = 0.0;
    let mut strikeouts_covered = 0usize;
    let mut innings_covered = 0usize;
    let mut standardized_strikeout_errors = Vec::new();
    let mut standardized_innings_errors = Vec::new();
    let mut league_strikeouts_error = 0.0;
    let mut pitcher_strikeouts_error = 0.0;
    let mut last_five_strikeouts_error = 0.0;
    let mut league_innings_error = 0.0;
    let mut pitcher_innings_error = 0.0;
    let mut last_five_innings_error = 0.0;
    let mut covered = 0usize;
    let mut standardized_run_errors = Vec::new();
    let mut league_fip_error = 0.0;
    let mut pitcher_fip_error = 0.0;
    let mut last_five_fip_error = 0.0;
    let mut league_runs_error = 0.0;
    let mut pitcher_runs_error = 0.0;
    let mut last_five_runs_error = 0.0;
    let mut held_out_rows = 0usize;
    let mut held_out_dates = 0usize;
    let mut confidence_observations = Vec::new();
    let mut k_observations = Vec::new();
    let mut index = minimum_training_rows.max(4).min(ordered.len() - 1);
    let remaining_dates = ordered[index..]
        .iter()
        .map(|row| row.as_of)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let date_stride = remaining_dates.div_ceil(120).max(1);
    let mut date_number = 0usize;
    while index < ordered.len() {
        let test_date = ordered[index].as_of;
        let date_end = ordered[index..]
            .iter()
            .position(|row| row.as_of > test_date)
            .map(|offset| index + offset)
            .unwrap_or(ordered.len());
        let should_evaluate = date_number.is_multiple_of(date_stride);
        date_number += 1;
        if !should_evaluate {
            index = date_end;
            continue;
        }
        let model = BayesianLinearModel::fit(&ordered[..index])?;
        let training = &ordered[..index];
        let league_fip = mean_target(training, |row| row.target_fip)?;
        let league_runs = mean_target(training, |row| row.target_runs)?;
        let league_strikeouts = mean_target(training, |row| row.target_strikeouts)?;
        let league_innings = mean_target(training, |row| row.target_innings)?;
        for row in &ordered[index..date_end] {
            let projection = model.score(row)?;
            let fip = row
                .target_fip
                .ok_or_else(|| PnsError::Model("validation FIP target missing".into()))?;
            let runs = row
                .target_runs
                .ok_or_else(|| PnsError::Model("validation runs target missing".into()))?;
            let strikeouts = row
                .target_strikeouts
                .ok_or_else(|| PnsError::Model("validation strikeouts target missing".into()))?;
            let innings = row
                .target_innings
                .ok_or_else(|| PnsError::Model("validation innings target missing".into()))?;
            fip_error += (projection.projected_fip - fip).abs();
            runs_error += (projection.expected_runs_base - runs).abs();
            strikeouts_error += (projection.expected_strikeouts - strikeouts).abs();
            innings_error += (projection.expected_innings - innings).abs();
            if strikeouts >= projection.expected_strikeouts_low
                && strikeouts <= projection.expected_strikeouts_high
            {
                strikeouts_covered += 1;
            }
            if innings >= projection.expected_innings_low
                && innings <= projection.expected_innings_high
            {
                innings_covered += 1;
            }
            standardized_strikeout_errors.push(
                (projection.expected_strikeouts - strikeouts).abs()
                    / projection.strikeouts_posterior_sd.max(0.001),
            );
            standardized_innings_errors.push(
                (projection.expected_innings - innings).abs()
                    / projection.innings_posterior_sd.max(0.001),
            );
            let pitcher_rows: Vec<_> = training
                .iter()
                .filter(|prior| prior.pitcher_id == row.pitcher_id)
                .collect();
            let pitcher_fip =
                mean_refs(&pitcher_rows, |prior| prior.target_fip).unwrap_or(league_fip);
            let pitcher_runs =
                mean_refs(&pitcher_rows, |prior| prior.target_runs).unwrap_or(league_runs);
            let pitcher_strikeouts = mean_refs(&pitcher_rows, |prior| prior.target_strikeouts)
                .unwrap_or(league_strikeouts);
            let pitcher_innings =
                mean_refs(&pitcher_rows, |prior| prior.target_innings).unwrap_or(league_innings);
            let recent: Vec<_> = pitcher_rows.iter().rev().take(5).copied().collect();
            let last_five_fip = mean_refs(&recent, |prior| prior.target_fip).unwrap_or(pitcher_fip);
            let last_five_runs =
                mean_refs(&recent, |prior| prior.target_runs).unwrap_or(pitcher_runs);
            let last_five_strikeouts =
                mean_refs(&recent, |prior| prior.target_strikeouts).unwrap_or(pitcher_strikeouts);
            let last_five_innings =
                mean_refs(&recent, |prior| prior.target_innings).unwrap_or(pitcher_innings);
            league_fip_error += (league_fip - fip).abs();
            pitcher_fip_error += (pitcher_fip - fip).abs();
            last_five_fip_error += (last_five_fip - fip).abs();
            league_runs_error += (league_runs - runs).abs();
            pitcher_runs_error += (pitcher_runs - runs).abs();
            last_five_runs_error += (last_five_runs - runs).abs();
            league_strikeouts_error += (league_strikeouts - strikeouts).abs();
            pitcher_strikeouts_error += (pitcher_strikeouts - strikeouts).abs();
            last_five_strikeouts_error += (last_five_strikeouts - strikeouts).abs();
            league_innings_error += (league_innings - innings).abs();
            pitcher_innings_error += (pitcher_innings - innings).abs();
            last_five_innings_error += (last_five_innings - innings).abs();
            if runs >= projection.expected_runs_low && runs <= projection.expected_runs_high {
                covered += 1;
            }
            standardized_run_errors.push(
                (projection.expected_runs_base - runs).abs()
                    / projection.runs_posterior_sd.max(0.001),
            );
            confidence_observations.push(ConfidenceObservation {
                date: row.as_of,
                runs_epistemic_sd: projection.runs_epistemic_sd,
                starts_used: projection.starts_used,
                model_training_rows: projection.model_training_rows,
                absolute_error: (projection.expected_runs_base - runs).abs(),
            });
            k_observations.push(KObservation {
                date: row.as_of,
                k_rate_mean: projection.k_rate_mean,
                batters_faced_mean: projection.batters_faced_mean,
                direct_mean: projection.expected_strikeouts,
                direct_sd: projection.strikeouts_posterior_sd,
                actual_strikeouts: strikeouts,
                actual_batters: row.target_batters_faced.ok_or_else(|| {
                    PnsError::Model("validation batters-faced target missing".into())
                })?,
            });
            held_out_rows += 1;
        }
        held_out_dates += 1;
        index = date_end;
    }
    if held_out_rows == 0 {
        return Err(PnsError::Model(
            "walk-forward validation produced no held-out rows".into(),
        ));
    }
    standardized_run_errors.sort_by(f64::total_cmp);
    standardized_strikeout_errors.sort_by(f64::total_cmp);
    standardized_innings_errors.sort_by(f64::total_cmp);
    let quantile_index = ((standardized_run_errors.len() as f64 * 0.90).ceil() as usize)
        .saturating_sub(1)
        .min(standardized_run_errors.len() - 1);
    let recommended_interval_scale =
        (standardized_run_errors[quantile_index] / 1.645).clamp(0.75, 2.0);
    let strikeout_quantile =
        standardized_strikeout_errors[quantile_index.min(standardized_strikeout_errors.len() - 1)];
    let innings_quantile =
        standardized_innings_errors[quantile_index.min(standardized_innings_errors.len() - 1)];
    let recommended_strikeouts_interval_scale = (strikeout_quantile / 1.645).clamp(0.75, 3.0);
    let recommended_innings_interval_scale = (innings_quantile / 1.645).clamp(0.75, 3.0);
    let calibrated_covered = standardized_run_errors
        .iter()
        .filter(|error| **error <= 1.645 * recommended_interval_scale)
        .count();
    let calibrated_strikeouts_covered = standardized_strikeout_errors
        .iter()
        .filter(|error| **error <= 1.645 * recommended_strikeouts_interval_scale)
        .count();
    let calibrated_innings_covered = standardized_innings_errors
        .iter()
        .filter(|error| **error <= 1.645 * recommended_innings_interval_scale)
        .count();
    let (confidence_calibration, confidence_audit) =
        confidence_calibration(&confidence_observations)?;
    let k_distribution = calibrate_k_distribution(&k_observations)?;
    Ok(ValidationMetrics {
        held_out_rows,
        held_out_dates,
        fip_mae: fip_error / held_out_rows as f64,
        runs_mae: runs_error / held_out_rows as f64,
        strikeouts_mae: strikeouts_error / held_out_rows as f64,
        innings_mae: innings_error / held_out_rows as f64,
        strikeouts_interval_coverage: strikeouts_covered as f64 / held_out_rows as f64,
        innings_interval_coverage: innings_covered as f64 / held_out_rows as f64,
        calibrated_strikeouts_interval_coverage: calibrated_strikeouts_covered as f64
            / held_out_rows as f64,
        calibrated_innings_interval_coverage: calibrated_innings_covered as f64
            / held_out_rows as f64,
        recommended_strikeouts_interval_scale,
        recommended_innings_interval_scale,
        league_strikeouts_mae: league_strikeouts_error / held_out_rows as f64,
        pitcher_history_strikeouts_mae: pitcher_strikeouts_error / held_out_rows as f64,
        last_five_strikeouts_mae: last_five_strikeouts_error / held_out_rows as f64,
        league_innings_mae: league_innings_error / held_out_rows as f64,
        pitcher_history_innings_mae: pitcher_innings_error / held_out_rows as f64,
        last_five_innings_mae: last_five_innings_error / held_out_rows as f64,
        runs_interval_coverage: covered as f64 / held_out_rows as f64,
        calibrated_runs_interval_coverage: calibrated_covered as f64 / held_out_rows as f64,
        recommended_interval_scale,
        league_fip_mae: league_fip_error / held_out_rows as f64,
        pitcher_history_fip_mae: pitcher_fip_error / held_out_rows as f64,
        last_five_fip_mae: last_five_fip_error / held_out_rows as f64,
        league_runs_mae: league_runs_error / held_out_rows as f64,
        pitcher_history_runs_mae: pitcher_runs_error / held_out_rows as f64,
        last_five_runs_mae: last_five_runs_error / held_out_rows as f64,
        confidence_calibration,
        confidence_audit,
        k_distribution,
    })
}

/// Chooses the beta-binomial concentration and batters-faced spread on the
/// earlier 70% of held-out dates, then grades the later 30% that had no say in
/// the choice: mean predictive log score against the old Gaussian treatment,
/// point MAE for both estimators, and over-line reliability at common lines.
fn calibrate_k_distribution(observations: &[KObservation]) -> Result<KDistributionCalibration> {
    if observations.is_empty() {
        return Err(PnsError::Model(
            "strikeout distribution calibration produced no held-out rows".into(),
        ));
    }
    let dates: Vec<_> = observations
        .iter()
        .map(|observation| observation.date)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let training_date_count = (dates.len() * 7).div_ceil(10).max(1);
    let training_through = dates[training_date_count.min(dates.len()) - 1];
    let (training, audit): (Vec<&KObservation>, Vec<&KObservation>) = observations
        .iter()
        .partition(|observation| observation.date <= training_through);
    let mut residuals: Vec<_> = training
        .iter()
        .map(|observation| observation.actual_batters - observation.batters_faced_mean)
        .collect();
    residuals.sort_by(f64::total_cmp);
    let bf_residual_quantiles: Vec<_> = (1..=19)
        .map(|step| percentile(&residuals, step as f64 * 0.05))
        .collect();
    const CONCENTRATION_GRID: [f64; 11] = [
        25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1200.0, 2000.0,
    ];
    let mut concentration = CONCENTRATION_GRID[0];
    let mut best_log_likelihood = f64::NEG_INFINITY;
    for candidate in CONCENTRATION_GRID {
        let log_likelihood: f64 = training
            .iter()
            .map(|observation| {
                observation_log_score(observation, &bf_residual_quantiles, candidate)
            })
            .sum();
        if log_likelihood > best_log_likelihood {
            best_log_likelihood = log_likelihood;
            concentration = candidate;
        }
    }
    let mut calibration = KDistributionCalibration {
        concentration,
        bf_residual_quantiles,
        training_rows: training.len(),
        audit_rows: audit.len(),
        audit_log_score: None,
        gaussian_baseline_log_score: None,
        compositional_strikeouts_mae: None,
        direct_strikeouts_mae: None,
        line_reliability: Vec::new(),
    };
    if audit.is_empty() {
        return Ok(calibration);
    }
    let rows = audit.len() as f64;
    let mut log_score = 0.0;
    let mut gaussian_log_score = 0.0;
    let mut compositional_error = 0.0;
    let mut direct_error = 0.0;
    let mut over_predictions: Vec<(f64, f64, usize)> = [3.5, 4.5, 5.5, 6.5, 7.5]
        .map(|line| (line, 0.0, 0usize))
        .into();
    for observation in &audit {
        let pmf = strikeout_pmf(
            observation.k_rate_mean,
            observation.batters_faced_mean,
            &calibration.bf_residual_quantiles,
            calibration.concentration,
        );
        let actual = observation.actual_strikeouts;
        let k_index = (actual.round().max(0.0) as usize).min(MAX_STRIKEOUTS);
        log_score += pmf[k_index].max(1e-12).ln();
        let sd = observation.direct_sd.max(0.01);
        let gaussian_probability = (normal_cdf((actual + 0.5 - observation.direct_mean) / sd)
            - normal_cdf((actual - 0.5 - observation.direct_mean) / sd))
        .max(1e-12);
        gaussian_log_score += gaussian_probability.ln();
        let pmf_mean: f64 = pmf
            .iter()
            .enumerate()
            .map(|(k, probability)| k as f64 * probability)
            .sum();
        compositional_error += (pmf_mean - actual).abs();
        direct_error += (observation.direct_mean - actual).abs();
        for (line, predicted_sum, observed_count) in &mut over_predictions {
            let predicted_over: f64 = pmf
                .iter()
                .enumerate()
                .filter(|(k, _)| *k as f64 > *line)
                .map(|(_, probability)| probability)
                .sum();
            *predicted_sum += predicted_over;
            if actual > *line {
                *observed_count += 1;
            }
        }
    }
    calibration.audit_log_score = Some(log_score / rows);
    calibration.gaussian_baseline_log_score = Some(gaussian_log_score / rows);
    calibration.compositional_strikeouts_mae = Some(compositional_error / rows);
    calibration.direct_strikeouts_mae = Some(direct_error / rows);
    calibration.line_reliability = over_predictions
        .into_iter()
        .map(|(line, predicted_sum, observed_count)| KLineReliability {
            line,
            rows: audit.len(),
            mean_predicted_over: predicted_sum / rows,
            observed_over_rate: observed_count as f64 / rows,
        })
        .collect();
    Ok(calibration)
}

fn observation_log_score(
    observation: &KObservation,
    bf_residual_quantiles: &[f64],
    concentration: f64,
) -> f64 {
    let pmf = strikeout_pmf(
        observation.k_rate_mean,
        observation.batters_faced_mean,
        bf_residual_quantiles,
        concentration,
    );
    let k = (observation.actual_strikeouts.round().max(0.0) as usize).min(MAX_STRIKEOUTS);
    pmf[k].max(1e-12).ln()
}

fn confidence_calibration(
    observations: &[ConfidenceObservation],
) -> Result<(ConfidenceCalibration, ConfidenceAudit)> {
    let dates: Vec<_> = observations
        .iter()
        .map(|observation| observation.date)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let training_date_count = (dates.len() * 7).div_ceil(10).max(1);
    let training_through = dates[training_date_count.min(dates.len()) - 1];
    let training: Vec<_> = observations
        .iter()
        .filter(|observation| observation.date <= training_through && observation.starts_used == 5)
        .collect();
    let mut scores: Vec<_> = training
        .iter()
        .map(|observation| {
            observation.runs_epistemic_sd
                * (observation.model_training_rows.max(1) as f64).sqrt()
                * (5.0 / observation.starts_used.max(1) as f64).sqrt()
        })
        .collect();
    scores.sort_by(f64::total_cmp);
    if scores.is_empty() {
        return Err(PnsError::Model(
            "confidence calibration produced no training rows".into(),
        ));
    }
    let mut calibration = ConfidenceCalibration {
        high_max_normalized_epistemic_score: percentile(&scores, 0.33),
        medium_max_normalized_epistemic_score: percentile(&scores, 0.67),
        threshold_training_rows: training.len(),
        training_period_dates: training_date_count.min(dates.len()),
        tiers_validated: false,
        minimum_audit_rows_per_tier: 100,
        required_mae_ratio: 0.95,
    };
    let later_observations: Vec<_> = observations
        .iter()
        .filter(|observation| observation.date > training_through)
        .collect();
    let audit_observations: Vec<_> = later_observations
        .iter()
        .copied()
        .filter(|observation| observation.starts_used == 5)
        .collect();
    let short_history: Vec<_> = later_observations
        .iter()
        .copied()
        .filter(|observation| observation.starts_used < 5)
        .collect();
    let mut audit = ConfidenceAudit {
        rows: audit_observations.len(),
        dates: dates
            .len()
            .saturating_sub(calibration.training_period_dates),
        ..ConfidenceAudit::default()
    };
    let mut high = Vec::new();
    let mut medium = Vec::new();
    let mut low = Vec::new();
    for observation in audit_observations {
        let score = calibration.epistemic_score(
            observation.runs_epistemic_sd,
            observation.model_training_rows,
            observation.starts_used,
        );
        if score <= calibration.high_max_normalized_epistemic_score {
            high.push(observation);
        } else if score <= calibration.medium_max_normalized_epistemic_score {
            medium.push(observation);
        } else {
            low.push(observation);
        }
    }
    audit.high = audit_tier(&high);
    audit.medium = audit_tier(&medium);
    audit.low = audit_tier(&low);
    audit.short_history = audit_tier(&short_history);
    calibration.tiers_validated = tiers_are_validated(&audit, calibration);
    Ok((calibration, audit))
}

fn tiers_are_validated(audit: &ConfidenceAudit, calibration: ConfidenceCalibration) -> bool {
    if !calibration.has_valid_thresholds()
        || !calibration.required_mae_ratio.is_finite()
        || !(0.0..=1.0).contains(&calibration.required_mae_ratio)
    {
        return false;
    }
    let Some((high_mae, medium_mae, low_mae)) = audit
        .high
        .runs_mae
        .zip(audit.medium.runs_mae)
        .zip(audit.low.runs_mae)
        .map(|((high, medium), low)| (high, medium, low))
    else {
        return false;
    };
    audit.high.rows >= calibration.minimum_audit_rows_per_tier
        && audit.medium.rows >= calibration.minimum_audit_rows_per_tier
        && audit.low.rows >= calibration.minimum_audit_rows_per_tier
        && high_mae <= calibration.required_mae_ratio * medium_mae
        && medium_mae <= calibration.required_mae_ratio * low_mae
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn audit_tier(observations: &[&ConfidenceObservation]) -> ConfidenceTierAudit {
    if observations.is_empty() {
        return ConfidenceTierAudit::default();
    }
    let rows = observations.len();
    let absolute_error = observations
        .iter()
        .map(|observation| observation.absolute_error)
        .sum::<f64>();
    ConfidenceTierAudit {
        rows,
        runs_mae: Some(absolute_error / rows as f64),
    }
}

fn mean_target(rows: &[FeatureRow], target: impl Fn(&FeatureRow) -> Option<f64>) -> Result<f64> {
    let values: Vec<_> = rows.iter().filter_map(target).collect();
    if values.is_empty() {
        return Err(PnsError::Model("baseline target is empty".into()));
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_refs(rows: &[&FeatureRow], target: impl Fn(&FeatureRow) -> Option<f64>) -> Option<f64> {
    let values: Vec<_> = rows.iter().filter_map(|row| target(row)).collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

pub fn time_split_validate(rows: &[FeatureRow]) -> Result<ValidationMetrics> {
    walk_forward_validate(rows, rows.len() * 4 / 5)
}

/// Conjugate Normal-Inverse-Gamma regression. The observation noise sigma^2 is
/// learned from the data instead of fixed: sigma^2 ~ InvGamma(a, b) with the
/// prior mean set to `prior_noise_mean`, and beta | sigma^2 ~ N(m0, sigma^2 V0).
/// With thousands of rows the data dominate and b_n/a_n approaches the
/// residual variance, so intervals are honest at the source.
fn fit_target_nig(
    rows: &[FeatureRow],
    target: impl Fn(&FeatureRow) -> Option<f64>,
    prior_intercept: f64,
    prior_noise_mean: f64,
) -> Result<Posterior> {
    let p = rows[0].values.len();
    let prior_shape = 3.0;
    let prior_rate = prior_noise_mean * (prior_shape - 1.0);
    // V0 is scaled so the marginal prior variance of each coefficient stays
    // near the previous fixed value of 4.0 at the prior mean of sigma^2.
    let prior_variance = 4.0 / prior_noise_mean;
    let mut precision = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for (j, row) in precision.iter_mut().enumerate() {
        row[j] = 1.0 / prior_variance;
    }
    // Intercept prior centers on league-average outcomes; other coefficients center at zero.
    rhs[0] = prior_intercept / prior_variance;
    let mut quadratic = prior_intercept * prior_intercept / prior_variance;
    let mut count = 0.0;
    for row in rows {
        let y = target(row).ok_or_else(|| PnsError::Model("training target missing".into()))?;
        quadratic += y * y;
        count += 1.0;
        for (j, precision_row) in precision.iter_mut().enumerate() {
            rhs[j] += row.values[j] * y;
            for (k, cell) in precision_row.iter_mut().enumerate() {
                *cell += row.values[j] * row.values[k];
            }
        }
    }
    let covariance = invert(precision)?;
    let mean = mat_vec(&covariance, &rhs);
    let noise_shape = prior_shape + count / 2.0;
    let noise_rate = (prior_rate + 0.5 * (quadratic - dot(&mean, &rhs))).max(1e-9);
    Ok(Posterior {
        mean,
        covariance,
        noise_shape,
        noise_rate,
    })
}

fn predict(p: &Posterior, x: &[f64]) -> Result<(f64, f64, f64)> {
    if p.mean.len() != x.len() {
        return Err(PnsError::Model(
            "feature width differs from fitted model".into(),
        ));
    }
    let mean = dot(&p.mean, x);
    let covariance_x = mat_vec(&p.covariance, x);
    let scaled_epistemic = dot(x, &covariance_x).max(0.0);
    let noise = p.noise_variance();
    let epistemic_variance = noise * scaled_epistemic;
    let predictive_variance = noise * (1.0 + scaled_epistemic);
    Ok((
        mean,
        predictive_variance.max(0.0).sqrt(),
        epistemic_variance.sqrt(),
    ))
}

/// Discrete strikeout distribution: a beta-binomial in the per-batter
/// strikeout probability, mixed over an empirical distribution of batters
/// faced. `bf_residual_quantiles` are walk-forward residuals of the
/// batters-faced model, so workload uncertainty (early hooks, deep starts)
/// shapes the tails instead of a symmetric bell curve.
pub(crate) fn strikeout_pmf(
    rate: f64,
    batters: f64,
    bf_residual_quantiles: &[f64],
    concentration: f64,
) -> Vec<f64> {
    let rate = rate.clamp(0.03, 0.55);
    let alpha = rate * concentration;
    let beta = (1.0 - rate) * concentration;
    let mut pmf = vec![0.0; MAX_STRIKEOUTS + 1];
    for quantile in bf_residual_quantiles {
        let faced = (batters + quantile).round().clamp(3.0, 45.0) as usize;
        for (k, slot) in pmf
            .iter_mut()
            .enumerate()
            .take(faced.min(MAX_STRIKEOUTS) + 1)
        {
            *slot += beta_binomial_pmf(k, faced, alpha, beta);
        }
    }
    let total: f64 = pmf.iter().sum();
    if total > 0.0 {
        for value in &mut pmf {
            *value /= total;
        }
    }
    pmf
}

fn beta_binomial_pmf(k: usize, n: usize, alpha: f64, beta: f64) -> f64 {
    let (k, n) = (k as f64, n as f64);
    let ln_choose = ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0);
    (ln_choose + ln_beta(k + alpha, n - k + beta) - ln_beta(alpha, beta)).exp()
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation (g = 7, n = 9), accurate to ~1e-13 for x > 0.5.
/// All call sites pass arguments >= 1 or positive beta-binomial shapes.
fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut series = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (x + index as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

/// Abramowitz & Stegun 7.1.26, absolute error < 1.5e-7. Used only for the
/// Gaussian-baseline log score in validation, never for pricing.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn invert(mut a: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let mut inv = vec![vec![0.0; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        if a[pivot][col].abs() < 1e-10 {
            return Err(PnsError::Model("posterior precision is singular".into()));
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let scale = a[col][col];
        for j in 0..n {
            a[col][j] /= scale;
            inv[col][j] /= scale;
        }
        for i in 0..n {
            if i == col {
                continue;
            }
            let factor = a[i][col];
            for j in 0..n {
                a[i][j] -= factor * a[col][j];
                inv[i][j] -= factor * inv[col][j];
            }
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    #[test]
    fn validates_only_on_newest_rows() {
        let starts = crate::data::pitcher_logs::fixture_starts().unwrap();
        let rows = crate::features::historical_rows(&starts, 5).unwrap();
        let metrics = super::time_split_validate(&rows).unwrap();
        assert!(metrics.held_out_rows >= 1);
        assert!(metrics.fip_mae.is_finite() && metrics.runs_mae.is_finite());
        assert!(metrics.strikeouts_mae.is_finite() && metrics.innings_mae.is_finite());
        assert!((0.0..=1.0).contains(&metrics.runs_interval_coverage));
        assert!((0.0..=1.0).contains(&metrics.calibrated_runs_interval_coverage));
        assert!(
            metrics
                .confidence_calibration
                .high_max_normalized_epistemic_score
                .is_finite()
        );
        assert!(
            metrics
                .confidence_calibration
                .high_max_normalized_epistemic_score
                <= metrics
                    .confidence_calibration
                    .medium_max_normalized_epistemic_score
        );
        assert!(
            metrics.confidence_calibration.threshold_training_rows
                <= metrics.held_out_rows - metrics.confidence_audit.rows
        );
    }

    #[test]
    fn posterior_width_is_not_reduced_by_zero_clipping() {
        let starts = crate::data::pitcher_logs::fixture_starts().unwrap();
        let rows = crate::features::historical_rows(&starts, 5).unwrap();
        let model = super::BayesianLinearModel::fit(&rows).unwrap();
        let projection = model.score(&rows[0]).unwrap();
        assert!(projection.runs_epistemic_sd < projection.runs_posterior_sd);
        // Allow one ULP of float rounding when the band is not clipped at zero
        // and the width equals posterior_width exactly.
        assert!(
            projection.posterior_width
                >= projection.expected_runs_high - projection.expected_runs_low - 1e-12
        );
        if projection.expected_runs_low == 0.0 {
            assert!(
                projection.posterior_width
                    > projection.expected_runs_high - projection.expected_runs_low
            );
        }
    }

    #[test]
    fn nig_regression_learns_noise_variance_from_residuals() {
        let rows: Vec<_> = (0..200)
            .map(|index| crate::features::FeatureRow {
                pitcher_id: 1,
                as_of: NaiveDate::from_ymd_opt(2025, 4, 1).unwrap(),
                values: vec![1.0],
                starts_used: 5,
                target_fip: None,
                target_runs: Some(5.0 + if index % 2 == 0 { 1.5 } else { -1.5 }),
                target_strikeouts: None,
                target_innings: None,
                target_batters_faced: None,
                target_k_rate: None,
            })
            .collect();
        let posterior = super::fit_target_nig(&rows, |r| r.target_runs, 5.0, 10.0).unwrap();
        // Residual variance is exactly 2.25; the learned noise must land near it
        // even though the prior mean was 10.0.
        assert!((posterior.noise_variance() - 2.25).abs() < 0.5);
        assert!((posterior.mean[0] - 5.0).abs() < 0.05);
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        assert!((super::ln_gamma(5.0) - 24f64.ln()).abs() < 1e-10);
        assert!((super::ln_gamma(1.0)).abs() < 1e-12);
        assert!((super::ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }

    #[test]
    fn beta_binomial_pmf_is_a_distribution_with_the_expected_mean() {
        let total: f64 = (0..=20)
            .map(|k| super::beta_binomial_pmf(k, 20, 5.0, 20.0))
            .sum();
        assert!((total - 1.0).abs() < 1e-9);
        let mean: f64 = (0..=20)
            .map(|k| k as f64 * super::beta_binomial_pmf(k, 20, 5.0, 20.0))
            .sum();
        assert!((mean - 4.0).abs() < 1e-9);
    }

    #[test]
    fn strikeout_pmf_normalizes_and_over_probabilities_decrease_with_the_line() {
        let pmf = super::strikeout_pmf(0.25, 24.0, &[-6.0, 0.0, 5.0], 150.0);
        assert_eq!(pmf.len(), super::MAX_STRIKEOUTS + 1);
        assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let over = |line: f64| {
            pmf.iter()
                .enumerate()
                .filter(|(k, _)| *k as f64 > line)
                .map(|(_, probability)| probability)
                .sum::<f64>()
        };
        assert!(over(3.5) > over(5.5));
        assert!(over(5.5) > over(8.5));
        let mean: f64 = pmf
            .iter()
            .enumerate()
            .map(|(k, probability)| k as f64 * probability)
            .sum();
        assert!((mean - 0.25 * 24.0).abs() < 1.5);
    }

    #[test]
    fn normal_cdf_is_symmetric_and_accurate() {
        assert!((super::normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((super::normal_cdf(1.645) - 0.95).abs() < 0.001);
        assert!((super::normal_cdf(-1.645) - 0.05).abs() < 0.001);
    }

    #[test]
    fn missing_velocity_is_imputed_to_the_training_mean() {
        let starts = crate::data::pitcher_logs::fixture_starts().unwrap();
        let mut rows = crate::features::historical_rows(&starts, 5).unwrap();
        for (index, row) in rows.iter_mut().enumerate() {
            row.values[7] = if index == 0 { 0.0 } else { 90.0 + index as f64 };
        }
        let (standardized, means, _) = super::standardize_rows(&rows).unwrap();
        assert!(means[7] > 90.0);
        assert_eq!(standardized[0].values[7], 0.0);
    }

    #[test]
    fn confidence_thresholds_use_early_dates_and_audit_later_dates() {
        let observations: Vec<_> = (0..10)
            .map(|day| super::ConfidenceObservation {
                date: NaiveDate::from_ymd_opt(2025, 4, day + 1).unwrap(),
                runs_epistemic_sd: (day + 1) as f64,
                starts_used: if day < 2 { 3 } else { 5 },
                model_training_rows: 1,
                absolute_error: day as f64,
            })
            .collect();
        let (calibration, audit) = super::confidence_calibration(&observations).unwrap();
        assert_eq!(calibration.training_period_dates, 7);
        assert_eq!(calibration.threshold_training_rows, 5);
        assert_eq!(calibration.high_max_normalized_epistemic_score, 4.0);
        assert_eq!(calibration.medium_max_normalized_epistemic_score, 6.0);
        assert_eq!(audit.dates, 3);
        assert_eq!(audit.rows, 3);
        assert_eq!(audit.high.rows + audit.medium.rows + audit.low.rows, 3);
        assert_eq!(audit.low.runs_mae, Some(8.0));
        assert!(!calibration.tiers_validated);
    }

    #[test]
    fn confidence_tiers_require_sample_size_and_ordered_mae_separation() {
        let calibration = super::ConfidenceCalibration {
            high_max_normalized_epistemic_score: 1.0,
            medium_max_normalized_epistemic_score: 2.0,
            threshold_training_rows: 500,
            training_period_dates: 50,
            tiers_validated: false,
            minimum_audit_rows_per_tier: 100,
            required_mae_ratio: 0.95,
        };
        let tier = |rows, mae| super::ConfidenceTierAudit {
            rows,
            runs_mae: Some(mae),
        };
        let validated = super::ConfidenceAudit {
            rows: 300,
            dates: 20,
            high: tier(100, 1.0),
            medium: tier(100, 1.1),
            low: tier(100, 1.2),
            short_history: super::ConfidenceTierAudit::default(),
        };
        assert!(super::tiers_are_validated(&validated, calibration));
        let too_small = super::ConfidenceAudit {
            high: tier(99, 1.0),
            ..validated
        };
        assert!(!super::tiers_are_validated(&too_small, calibration));
        let unordered = super::ConfidenceAudit {
            high: tier(100, 1.1),
            medium: tier(100, 1.0),
            ..validated
        };
        assert!(!super::tiers_are_validated(&unordered, calibration));

        let invalid_thresholds = super::ConfidenceCalibration {
            high_max_normalized_epistemic_score: 3.0,
            medium_max_normalized_epistemic_score: 2.0,
            ..calibration
        };
        assert!(!super::tiers_are_validated(&validated, invalid_thresholds));
    }

    #[test]
    fn confidence_audit_separates_short_histories_from_candidate_tiers() {
        let mut observations: Vec<_> = (0..10)
            .map(|day| super::ConfidenceObservation {
                date: NaiveDate::from_ymd_opt(2025, 4, day + 1).unwrap(),
                runs_epistemic_sd: 1.0,
                starts_used: 5,
                model_training_rows: 1,
                absolute_error: 1.0,
            })
            .collect();
        observations[8].starts_used = 4;
        observations[8].absolute_error = 9.0;

        let (_, audit) = super::confidence_calibration(&observations).unwrap();

        assert_eq!(audit.rows, 2);
        assert_eq!(audit.high.rows + audit.medium.rows + audit.low.rows, 2);
        assert_eq!(audit.short_history.rows, 1);
        assert_eq!(audit.short_history.runs_mae, Some(9.0));
    }
}
