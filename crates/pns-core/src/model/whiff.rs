//! Does swinging-strike rate tell the k-rate model anything its existing
//! features do not?
//!
//! The model predicts strikeouts from *past strikeouts*. Swinging-strike rate
//! stabilises faster than strikeout rate and is the better-known leading
//! indicator, and it has never been in the feature set. This module answers
//! whether adding it is worth a feature-contract bump, a schema version, and a
//! retrain — *before* paying for any of that.
//!
//! # Why it lives here
//!
//! A child module can reach its parent's private items, so this reuses the real
//! estimator ([`super::fit_target_nig`]), the real standardizer, and the real
//! predictive distribution. A study that re-implemented the regression would be
//! measuring its own arithmetic, not the model's.
//!
//! # What is compared, and on what
//!
//! Two fits of the same target on the *same rows*, differing only in feature
//! width. Both are fitted on one window and scored on a later one, so a feature
//! that merely memorises history shows up as no held-out gain — the exact trap
//! the removal-hazard work fell into, where a 2.25x in-sample improvement made
//! out-of-sample forecasting slightly worse.
//!
//! The primary statistic is **held-out predictive log-likelihood**, not point
//! error. A feature can shave the mean while leaving the model just as unsure,
//! and for a forecaster that prices distributions the second thing is what
//! matters.

use super::{Posterior, fit_target_nig, ln_gamma, predict, standardize_rows, standardize_values};
use crate::features::FeatureRow;
use crate::paper_stats::{self, Interval};
use crate::{PnsError, Result};
use serde::Serialize;

/// Held-out quality of one fitted target.
#[derive(Debug, Clone, Serialize)]
pub struct HeldOutFit {
    pub rows: usize,
    /// Mean log density the fit assigned to what actually happened. Higher is
    /// better; this is the primary statistic.
    pub mean_log_likelihood: f64,
    pub mae: f64,
    pub rmse: f64,
}

/// One added feature's posterior, in standardized units.
#[derive(Debug, Clone, Serialize)]
pub struct AddedCoefficient {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    /// `|mean| / sd`. A coefficient the data cannot separate from zero sits near
    /// 0; roughly 2 or more is clearly separated.
    pub z: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WhiffComparison {
    pub fit_rows: usize,
    pub baseline: HeldOutFit,
    pub augmented: HeldOutFit,
    /// Augmented minus baseline. Positive means the added features helped.
    pub log_likelihood_gain: f64,
    /// The same gain with a 95% interval, resampling **pitchers** rather than
    /// rows. A pitcher's starts share whatever the features get wrong about him,
    /// so treating his ten starts as ten independent pieces of evidence would
    /// invent precision — the same design effect measured at 2.05x on the live
    /// paper log.
    pub log_likelihood_gain_interval: Option<Interval>,
    /// Augmented minus baseline. Negative means the added features helped.
    pub mae_change: f64,
    pub added_coefficients: Vec<AddedCoefficient>,
}

/// Exact predictive log density under the Normal-Inverse-Gamma posterior.
///
/// The predictive is Student-t with `2 * noise_shape` degrees of freedom, not a
/// Gaussian — [`super::predict`] returns its *scale*, and at the shapes this
/// model reaches the difference in the tails is exactly where a distributional
/// forecaster earns or loses its score. Using the real density avoids quietly
/// grading two fits on an approximation that flatters whichever is more
/// confident.
fn student_t_log_density(actual: f64, location: f64, scale: f64, degrees_of_freedom: f64) -> f64 {
    let scale = scale.max(1e-12);
    let v = degrees_of_freedom.max(1e-6);
    let z = (actual - location) / scale;
    ln_gamma((v + 1.0) / 2.0)
        - ln_gamma(v / 2.0)
        - 0.5 * (v * std::f64::consts::PI).ln()
        - scale.ln()
        - (v + 1.0) / 2.0 * (1.0 + z * z / v).ln()
}

/// One evaluation row's log density, tagged with the pitcher it belongs to so
/// the interval can resample by pitcher rather than by row.
type ScoredRow = (u32, f64);

/// Fits `target_k_rate` on `fit` and scores it on `eval`.
fn fit_and_score(
    fit: &[FeatureRow],
    eval: &[FeatureRow],
) -> Result<(HeldOutFit, Posterior, Vec<ScoredRow>)> {
    if fit.len() < 4 {
        return Err(PnsError::Model("not enough rows to fit".into()));
    }
    let (standardized, means, scales) = standardize_rows(fit)?;
    // Same target, prior mean and prior noise the production fit uses, so the
    // only thing that differs between the two arms is the feature width.
    let posterior = fit_target_nig(&standardized, |row| row.target_k_rate, 0.22, 0.01)?;
    let degrees_of_freedom = 2.0 * posterior.noise_shape;

    let (mut log_likelihood, mut absolute, mut squared) = (0.0, 0.0, 0.0);
    let mut per_row = Vec::with_capacity(eval.len());
    for row in eval {
        let Some(actual) = row.target_k_rate else {
            continue;
        };
        let values = standardize_values(&row.values, &means, &scales)?;
        let (mean, scale, _) = predict(&posterior, &values)?;
        let density = student_t_log_density(actual, mean, scale, degrees_of_freedom);
        log_likelihood += density;
        absolute += (actual - mean).abs();
        squared += (actual - mean).powi(2);
        per_row.push((row.pitcher_id, density));
    }
    if per_row.is_empty() {
        return Err(PnsError::Model(
            "no evaluation rows carried a target".into(),
        ));
    }
    let counted_f = per_row.len() as f64;
    Ok((
        HeldOutFit {
            rows: per_row.len(),
            mean_log_likelihood: log_likelihood / counted_f,
            mae: absolute / counted_f,
            rmse: (squared / counted_f).sqrt(),
        },
        posterior,
        per_row,
    ))
}

/// Per-pitcher paired log-likelihood differences, the resampling unit for the
/// interval.
struct PitcherDifferences {
    differences: Vec<f64>,
}

fn mean_difference(clusters: &[&PitcherDifferences]) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for cluster in clusters {
        for difference in &cluster.differences {
            total += difference;
            count += 1;
        }
    }
    (count > 0).then(|| total / count as f64)
}

/// Compares a baseline feature set against one with extra columns appended.
///
/// `baseline_*` and `augmented_*` must describe the **same starts in the same
/// order**, differing only in feature width — otherwise the comparison measures
/// a change of sample rather than a change of information.
pub fn compare_k_rate(
    baseline_fit: &[FeatureRow],
    baseline_eval: &[FeatureRow],
    augmented_fit: &[FeatureRow],
    augmented_eval: &[FeatureRow],
    added_names: &[String],
) -> Result<WhiffComparison> {
    if baseline_fit.len() != augmented_fit.len() || baseline_eval.len() != augmented_eval.len() {
        return Err(PnsError::Model(
            "baseline and augmented arms must cover the same rows".into(),
        ));
    }
    let baseline_width = baseline_fit
        .first()
        .ok_or_else(|| PnsError::Model("no fit rows".into()))?
        .values
        .len();
    let augmented_width = augmented_fit[0].values.len();
    if augmented_width != baseline_width + added_names.len() {
        return Err(PnsError::Model(format!(
            "augmented width {augmented_width} does not equal baseline {baseline_width} plus \
             {} added features",
            added_names.len()
        )));
    }

    let (baseline, _, baseline_rows) = fit_and_score(baseline_fit, baseline_eval)?;
    let (augmented, posterior, augmented_rows) = fit_and_score(augmented_fit, augmented_eval)?;

    // Pair the two arms row by row. They were scored on the same eval rows in
    // the same order, so a mismatch here means the arms diverged and the whole
    // comparison is void.
    if baseline_rows.len() != augmented_rows.len() {
        return Err(PnsError::Model(
            "arms scored different numbers of evaluation rows".into(),
        ));
    }
    let mut by_pitcher: std::collections::BTreeMap<u32, Vec<f64>> =
        std::collections::BTreeMap::new();
    for ((pitcher, base), (other_pitcher, aug)) in baseline_rows.iter().zip(&augmented_rows) {
        if pitcher != other_pitcher {
            return Err(PnsError::Model(
                "arms scored evaluation rows in a different order".into(),
            ));
        }
        by_pitcher.entry(*pitcher).or_default().push(aug - base);
    }
    let clusters: Vec<PitcherDifferences> = by_pitcher
        .into_values()
        .map(|differences| PitcherDifferences { differences })
        .collect();
    let cluster_refs: Vec<&PitcherDifferences> = clusters.iter().collect();
    let point = mean_difference(&cluster_refs);
    let log_likelihood_gain_interval = point.and_then(|point| {
        let mut rng = paper_stats::Rng::new(paper_stats::SEED);
        let mut replicates = Vec::with_capacity(paper_stats::BOOTSTRAP_DRAWS);
        let mut draw: Vec<&PitcherDifferences> = Vec::with_capacity(clusters.len());
        for _ in 0..paper_stats::BOOTSTRAP_DRAWS {
            draw.clear();
            for _ in 0..clusters.len() {
                draw.push(&clusters[rng.below(clusters.len())]);
            }
            if let Some(value) = mean_difference(&draw) {
                replicates.push(value);
            }
        }
        paper_stats::bca_interval(point, &replicates, &clusters, mean_difference)
    });

    let added_coefficients = added_names
        .iter()
        .enumerate()
        .map(|(offset, name)| {
            let index = baseline_width + offset;
            let mean = posterior.mean[index];
            let sd = (posterior.noise_variance() * posterior.covariance[index][index])
                .max(0.0)
                .sqrt();
            AddedCoefficient {
                name: name.clone(),
                mean,
                sd,
                z: if sd > 0.0 { mean.abs() / sd } else { 0.0 },
            }
        })
        .collect();

    Ok(WhiffComparison {
        fit_rows: baseline_fit.len(),
        log_likelihood_gain: augmented.mean_log_likelihood - baseline.mean_log_likelihood,
        log_likelihood_gain_interval,
        mae_change: augmented.mae - baseline.mae,
        baseline,
        augmented,
        added_coefficients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Student-t must approach the Gaussian as degrees of freedom grow, and be
    /// fatter-tailed below it. Getting this backwards would systematically
    /// mis-score whichever fit is more confident.
    #[test]
    fn student_t_density_behaves_like_a_t() {
        let gaussian = |z: f64| -0.5 * ((2.0 * std::f64::consts::PI).ln() + z * z);
        // Far out in the tail, few degrees of freedom must assign MORE density.
        let heavy = student_t_log_density(3.0, 0.0, 1.0, 3.0);
        let light = student_t_log_density(3.0, 0.0, 1.0, 1_000_000.0);
        assert!(
            heavy > light,
            "t should be fatter tailed: {heavy} vs {light}"
        );
        // With huge degrees of freedom it must converge on the normal.
        assert!(
            (light - gaussian(3.0)).abs() < 1e-4,
            "t with huge df was {light}, normal is {}",
            gaussian(3.0)
        );
        // Scale enters as a log term: doubling it costs ln(2) at the centre.
        let tight = student_t_log_density(0.0, 0.0, 1.0, 50.0);
        let wide = student_t_log_density(0.0, 0.0, 2.0, 50.0);
        assert!((tight - wide - 2f64.ln()).abs() < 1e-9);
    }

    fn row(values: Vec<f64>, target: f64) -> FeatureRow {
        FeatureRow {
            pitcher_id: 1,
            as_of: chrono::NaiveDate::from_ymd_opt(2026, 5, 1).unwrap(),
            values,
            starts_used: 5,
            target_fip: None,
            target_runs: None,
            target_strikeouts: None,
            target_innings: None,
            target_batters_faced: None,
            target_k_rate: Some(target),
        }
    }

    /// A feature that genuinely drives the target must show a clear coefficient
    /// and improve held-out likelihood. This is the study's positive control: if
    /// it fails, a null result elsewhere means nothing.
    #[test]
    fn a_genuinely_predictive_feature_is_detected() {
        let mut baseline_fit = Vec::new();
        let mut augmented_fit = Vec::new();
        let mut baseline_eval = Vec::new();
        let mut augmented_eval = Vec::new();
        for index in 0..400 {
            // `signal` drives the target; `noise_feature` does not.
            let signal = f64::from(index % 23) / 23.0;
            let noise_feature = f64::from((index * 7) % 13) / 13.0;
            let target = 0.15 + 0.20 * signal;
            let base = vec![1.0, noise_feature];
            let mut augmented = base.clone();
            augmented.push(signal);
            if index % 4 == 0 {
                baseline_eval.push(row(base, target));
                augmented_eval.push(row(augmented, target));
            } else {
                baseline_fit.push(row(base, target));
                augmented_fit.push(row(augmented, target));
            }
        }
        let result = compare_k_rate(
            &baseline_fit,
            &baseline_eval,
            &augmented_fit,
            &augmented_eval,
            &["signal".to_string()],
        )
        .unwrap();

        assert!(
            result.log_likelihood_gain > 0.0,
            "a real feature must raise held-out likelihood, got {}",
            result.log_likelihood_gain
        );
        assert!(result.mae_change < 0.0, "and must lower error");
        let coefficient = &result.added_coefficients[0];
        assert!(
            coefficient.z > 2.0,
            "coefficient should be clearly non-zero, z was {}",
            coefficient.z
        );
    }

    /// A pure-noise feature must NOT look like a win. Without this the study
    /// could report a gain for anything at all.
    #[test]
    fn a_useless_feature_shows_no_meaningful_gain() {
        let mut baseline_fit = Vec::new();
        let mut augmented_fit = Vec::new();
        let mut baseline_eval = Vec::new();
        let mut augmented_eval = Vec::new();
        for index in 0..400 {
            let signal = f64::from(index % 23) / 23.0;
            // Deterministic but unrelated to the target.
            let junk = f64::from((index * 31) % 17) / 17.0;
            let target = 0.15 + 0.20 * signal;
            let base = vec![1.0, signal];
            let mut augmented = base.clone();
            augmented.push(junk);
            if index % 4 == 0 {
                baseline_eval.push(row(base, target));
                augmented_eval.push(row(augmented, target));
            } else {
                baseline_fit.push(row(base, target));
                augmented_fit.push(row(augmented, target));
            }
        }
        let result = compare_k_rate(
            &baseline_fit,
            &baseline_eval,
            &augmented_fit,
            &augmented_eval,
            &["junk".to_string()],
        )
        .unwrap();
        assert!(
            result.added_coefficients[0].z < 2.0,
            "noise should not produce a separated coefficient, z was {}",
            result.added_coefficients[0].z
        );
    }

    /// Mismatched arms are a bug, not a result: refuse rather than silently
    /// comparing different samples.
    #[test]
    fn mismatched_arms_are_rejected() {
        let fit = vec![row(vec![1.0, 0.5], 0.2); 10];
        let eval = vec![row(vec![1.0, 0.5], 0.2); 4];
        let wide_fit = vec![row(vec![1.0, 0.5, 0.1], 0.2); 9];
        assert!(
            compare_k_rate(&fit, &eval, &wide_fit, &eval, &["x".to_string()]).is_err(),
            "row-count mismatch must be refused"
        );
    }
}
