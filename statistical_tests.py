"""
Statistical testing framework
Fixes Issues: 4, 17, 40
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings


def compute_statistical_significance(
    baseline_metrics: List[float],
    curritail_metrics: List[float],
    alpha: float = 0.05,
    metric_name: str = "metric",
) -> Dict:
    """
    FIX Issue 4: Proper statistical testing with adequate sample size.

    Uses Welch's t-test (doesn't assume equal variances) or paired t-test.
    Reports p-values, effect sizes, and confidence intervals.

    Args:
        baseline_metrics: Metric values from baseline across seeds
        curritail_metrics: Metric values from CurriTail across seeds
        alpha: Significance level
        metric_name: Name for reporting

    Returns:
        Dictionary with test results
    """
    baseline_metrics = np.array(baseline_metrics)
    curritail_metrics = np.array(curritail_metrics)

    n = len(baseline_metrics)

    if n < 3:
        warnings.warn(
            f"Only {n} samples for {metric_name}. Statistical tests unreliable."
        )
        return {
            "n_samples": n,
            "significant": False,
            "p_value": 1.0,
            "warning": "Insufficient samples",
        }

    # Compute descriptive statistics
    base_mean = np.mean(baseline_metrics)
    base_std = np.std(baseline_metrics, ddof=1)
    curri_mean = np.mean(curritail_metrics)
    curri_std = np.std(curritail_metrics, ddof=1)

    # Paired t-test (same seeds used for both)
    if n >= 30:
        # Use standard t-test for large samples
        t_stat, p_value = stats.ttest_rel(curritail_metrics, baseline_metrics)
    else:
        # Use Welch's t-test for small samples (more conservative)
        t_stat, p_value = stats.ttest_ind(
            curritail_metrics, baseline_metrics, equal_var=False
        )

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((base_std**2 + curri_std**2) / 2)
    if pooled_std > 0:
        cohens_d = (curri_mean - base_mean) / pooled_std
    else:
        cohens_d = 0.0

    # Confidence interval for difference
    diff = (
        curritail_metrics - baseline_metrics
        if len(curritail_metrics) == len(baseline_metrics)
        else curritail_metrics
    )
    if len(diff) > 1:
        ci_lower, ci_upper = stats.t.interval(
            1 - alpha, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
        )
    else:
        ci_lower, ci_upper = 0, 0

    return {
        "n_samples": n,
        "baseline_mean": base_mean,
        "baseline_std": base_std,
        "curritail_mean": curri_mean,
        "curritail_std": curri_std,
        "mean_difference": curri_mean - base_mean,
        "p_value": p_value,
        "significant": p_value < alpha,
        "t_statistic": t_stat,
        "cohens_d": cohens_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": alpha,
    }


def compare_all_models(
    results_df: pd.DataFrame, reference_model: str = "Baseline", alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compare all models against reference with multiple testing correction.

    Args:
        results_df: DataFrame with columns [Dataset, Model, Seed, Metric1, Metric2, ...]
        reference_model: Model to compare against
        alpha: Significance level

    Returns:
        DataFrame with statistical test results
    """
    datasets = results_df["Dataset"].unique()
    models = [m for m in results_df["Model"].unique() if m != reference_model]
    metrics = [
        col for col in results_df.columns if col not in ["Dataset", "Model", "Seed"]
    ]

    comparisons = []

    for dataset in datasets:
        df_dataset = results_df[results_df["Dataset"] == dataset]

        reference_data = df_dataset[df_dataset["Model"] == reference_model]

        for model in models:
            model_data = df_dataset[df_dataset["Model"] == model]

            for metric in metrics:
                ref_values = reference_data[metric].values
                model_values = model_data[metric].values

                # Handle cases where models might have different numbers of seeds
                min_len = min(len(ref_values), len(model_values))
                if min_len == 0:
                    continue

                test_result = compute_statistical_significance(
                    ref_values[:min_len],
                    model_values[:min_len],
                    alpha=alpha,
                    metric_name=f"{dataset}_{model}_{metric}",
                )

                comparisons.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Metric": metric,
                        "Reference": reference_model,
                        **test_result,
                    }
                )

    comparison_df = pd.DataFrame(comparisons)

    # Bonferroni correction for multiple comparisons
    if len(comparison_df) > 0:
        comparison_df["p_value_corrected"] = comparison_df["p_value"] * len(
            comparison_df
        )
        comparison_df["p_value_corrected"] = comparison_df["p_value_corrected"].clip(
            upper=1.0
        )
        comparison_df["significant_corrected"] = (
            comparison_df["p_value_corrected"] < alpha
        )

    return comparison_df


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    FIX Issue 6: Bootstrap confidence interval for statistics.

    Args:
        data: Data array
        statistic_func: Function to compute statistic (e.g., np.mean, np.std)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level
        seed: Random seed

    Returns:
        Tuple of (statistic, ci_lower, ci_upper)
    """
    np.random.seed(seed)

    n = len(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    observed_stat = statistic_func(data)

    return observed_stat, ci_lower, ci_upper


def kurtosis_comparison_bootstrap(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    FIX Issue 6: Compare kurtosis with matched sample sizes using bootstrap.

    Args:
        real_data: Real data
        generated_data: Generated data
        n_bootstrap: Bootstrap iterations
        seed: Random seed

    Returns:
        Dictionary with comparison results
    """
    from scipy.stats import kurtosis

    np.random.seed(seed)

    n_real = len(real_data)

    # Bootstrap generated data to match real data sample size
    generated_kurtosis_bootstrap = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(generated_data, size=n_real, replace=True)
        generated_kurtosis_bootstrap.append(kurtosis(sample, fisher=True))

    generated_kurtosis_bootstrap = np.array(generated_kurtosis_bootstrap)

    real_kurt = kurtosis(real_data, fisher=True)
    gen_kurt_mean = np.mean(generated_kurtosis_bootstrap)
    gen_kurt_std = np.std(generated_kurtosis_bootstrap)

    # 95% confidence interval
    ci_lower = np.percentile(generated_kurtosis_bootstrap, 2.5)
    ci_upper = np.percentile(generated_kurtosis_bootstrap, 97.5)

    # Check if real kurtosis falls within generated CI
    real_in_ci = ci_lower <= real_kurt <= ci_upper

    return {
        "real_kurtosis": real_kurt,
        "generated_kurtosis_mean": gen_kurt_mean,
        "generated_kurtosis_std": gen_kurt_std,
        "generated_ci_lower": ci_lower,
        "generated_ci_upper": ci_upper,
        "real_in_generated_ci": real_in_ci,
        "absolute_error": abs(real_kurt - gen_kurt_mean),
    }


def validate_against_paper_results(
    results_df: pd.DataFrame,
    paper_results: Dict[str, Dict[str, float]],
    tolerance: float = 10.0,
) -> Dict:
    """
    FIX Issue 40: Validate computed results against paper's reported numbers.

    Args:
        results_df: Computed results DataFrame
        paper_results: Dictionary like {'SPX': {'Baseline_TailKL': 99.90, 'CurriTail_TailKL': 0.69}}
        tolerance: Tolerance for deviation

    Returns:
        Validation report
    """
    validation_report = []

    for dataset, metrics in paper_results.items():
        for key, paper_value in metrics.items():
            model, metric = key.rsplit("_", 1)

            # Get computed value
            df_subset = results_df[
                (results_df["Dataset"] == dataset) & (results_df["Model"] == model)
            ]

            if len(df_subset) == 0:
                validation_report.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Metric": metric,
                        "Paper": paper_value,
                        "Computed": np.nan,
                        "Match": False,
                        "Warning": "No computed results",
                    }
                )
                continue

            if metric not in df_subset.columns:
                validation_report.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Metric": metric,
                        "Paper": paper_value,
                        "Computed": np.nan,
                        "Match": False,
                        "Warning": "Metric not found",
                    }
                )
                continue

            computed_mean = df_subset[metric].mean()
            deviation = abs(computed_mean - paper_value)
            matches = deviation <= tolerance

            validation_report.append(
                {
                    "Dataset": dataset,
                    "Model": model,
                    "Metric": metric,
                    "Paper": paper_value,
                    "Computed": computed_mean,
                    "Deviation": deviation,
                    "Match": matches,
                    "Warning": None
                    if matches
                    else f"Deviation {deviation:.2f} exceeds tolerance {tolerance}",
                }
            )

    validation_df = pd.DataFrame(validation_report)

    # Summary
    n_total = len(validation_df)
    n_matched = validation_df["Match"].sum()

    summary = {
        "total_comparisons": n_total,
        "matched": n_matched,
        "match_rate": n_matched / n_total if n_total > 0 else 0,
        "details": validation_df,
    }

    if n_matched < n_total:
        warnings.warn(
            f"Only {n_matched}/{n_total} metrics matched paper results. "
            "Review data preprocessing and hyperparameters."
        )

    return summary
