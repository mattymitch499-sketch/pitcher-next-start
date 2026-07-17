use crate::features::FeatureRow;
use crate::{PnsError, Projection, Result};

#[derive(Debug, Clone)]
struct Posterior {
    mean: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    noise_variance: f64,
}

#[derive(Debug, Clone)]
pub struct BayesianLinearModel {
    fip: Posterior,
    runs: Posterior,
    strikeouts: Posterior,
    innings: Posterior,
    interval_scale: f64,
    strikeouts_interval_scale: f64,
    innings_interval_scale: f64,
    feature_means: Vec<f64>,
    feature_scales: Vec<f64>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
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
            fip: fit_target(&standardized, |r| r.target_fip, 4.20)?,
            runs: fit_target(&standardized, |r| r.target_runs, 3.0)?,
            strikeouts: fit_target_with_noise(&standardized, |r| r.target_strikeouts, 5.2, 6.25)?,
            innings: fit_target_with_noise(&standardized, |r| r.target_innings, 5.3, 1.0)?,
            interval_scale: 1.0,
            strikeouts_interval_scale: 1.0,
            innings_interval_scale: 1.0,
            feature_means,
            feature_scales,
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

    pub fn score(&self, row: &FeatureRow) -> Result<Projection> {
        let values = standardize_values(&row.values, &self.feature_means, &self.feature_scales)?;
        let (fip, fip_sd) = predict(&self.fip, &values)?;
        let (runs, runs_sd) = predict(&self.runs, &values)?;
        let (strikeouts, strikeouts_sd) = predict(&self.strikeouts, &values)?;
        let (innings, innings_sd) = predict(&self.innings, &values)?;
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
            posterior_width: high - low,
            starts_used: row.starts_used,
            fip_posterior_sd: fip_sd,
            runs_posterior_sd: runs_sd,
            expected_strikeouts: strikeouts.clamp(0.0, 15.0),
            expected_innings: innings.clamp(1.0, 9.0),
            expected_strikeouts_low: (strikeouts - strikeouts_half_width).max(0.0),
            expected_strikeouts_high: (strikeouts + strikeouts_half_width).clamp(0.1, 20.0),
            expected_innings_low: (innings - innings_half_width).max(0.0),
            expected_innings_high: (innings + innings_half_width).clamp(0.1, 9.0),
            strikeouts_posterior_sd: strikeouts_sd,
            innings_posterior_sd: innings_sd,
        })
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
        means[feature] =
            rows.iter().map(|row| row.values[feature]).sum::<f64>() / rows.len() as f64;
        let variance = rows
            .iter()
            .map(|row| (row.values[feature] - means[feature]).powi(2))
            .sum::<f64>()
            / rows.len() as f64;
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
    })
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

#[cfg(test)]
mod tests {
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
    }
}

fn fit_target(
    rows: &[FeatureRow],
    target: impl Fn(&FeatureRow) -> Option<f64>,
    prior_intercept: f64,
) -> Result<Posterior> {
    fit_target_with_noise(rows, target, prior_intercept, 2.25)
}

fn fit_target_with_noise(
    rows: &[FeatureRow],
    target: impl Fn(&FeatureRow) -> Option<f64>,
    prior_intercept: f64,
    noise_variance: f64,
) -> Result<Posterior> {
    let p = rows[0].values.len();
    let prior_variance = 4.0;
    let mut precision = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for j in 0..p {
        precision[j][j] = 1.0 / prior_variance;
    }
    // Intercept prior centers on league-average outcomes; other coefficients center at zero.
    rhs[0] = prior_intercept / prior_variance;
    for row in rows {
        let y = target(row).ok_or_else(|| PnsError::Model("training target missing".into()))?;
        for j in 0..p {
            rhs[j] += row.values[j] * y / noise_variance;
            for k in 0..p {
                precision[j][k] += row.values[j] * row.values[k] / noise_variance;
            }
        }
    }
    let covariance = invert(precision)?;
    let mean = mat_vec(&covariance, &rhs);
    Ok(Posterior {
        mean,
        covariance,
        noise_variance,
    })
}

fn predict(p: &Posterior, x: &[f64]) -> Result<(f64, f64)> {
    if p.mean.len() != x.len() {
        return Err(PnsError::Model(
            "feature width differs from fitted model".into(),
        ));
    }
    let mean = dot(&p.mean, x);
    let covariance_x = mat_vec(&p.covariance, x);
    let variance = p.noise_variance + dot(x, &covariance_x);
    Ok((mean, variance.max(0.0).sqrt()))
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
    for i in 0..n {
        inv[i][i] = 1.0;
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
