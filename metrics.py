"""
Metrics module with correct implementations
Fixes Issues: 3, 9, 10, 19, 30, 34
"""
import numpy as np
from scipy.stats import entropy, ks_2samp, kurtosis
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf
from typing import Dict, Tuple
import warnings

from utils import apply_laplace_smoothing_correct, verify_data_is_raw


def compute_tail_metrics(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    threshold_q: float = 0.01,
    n_bins: int = 5
) -> Dict[str, float]:
    """
    Compute comprehensive tail metrics with statistical rigor.
    
    FIX Issue 9: Uses Wasserstein distance instead of KL for sparse tails.
    FIX Issue 10: Applies correct Laplace smoothing.
    FIX Issue 30: Validates input data is raw.
    
    Args:
        real_data: Real data samples (raw, not scaled)
        fake_data: Generated data samples (raw, not scaled)
        threshold_q: Quantile threshold for tail definition
        n_bins: Number of bins for histogram-based metrics
    
    Returns:
        Dictionary with metric names and values
    """
    real_data = real_data.flatten()
    fake_data = fake_data.flatten()
    
    # Validate data is raw
    if not verify_data_is_raw(real_data):
        warnings.warn("Real data appears to be scaled. Metrics may be incorrect.")
    if not verify_data_is_raw(fake_data):
        warnings.warn("Fake data appears to be scaled. Metrics may be incorrect.")
    
    # Compute tail threshold
    var_thresh = np.quantile(real_data, threshold_q)
    
    # 1. Tail Recall (proportion of generated samples in tail region)
    # Should be close to threshold_q (e.g., 0.01 for 1% tail)
    tail_recall = np.mean(fake_data <= var_thresh)
    
    # Extract tail samples
    real_tail = real_data[real_data <= var_thresh]
    fake_tail = fake_data[fake_data <= var_thresh]
    
    # Check if enough tail samples exist
    if len(real_tail) < 10 or len(fake_tail) < 10:
        return {
            'tail_recall': tail_recall,
            'tail_kl': 999.0,
            'tail_wasserstein': 999.0,
            'tail_ks': 1.0,
            'tail_coverage': 0.0
        }
    
    # 2. Wasserstein Distance (Earth Mover's Distance) - better for sparse distributions
    # FIX Issue 9: Use continuous metric instead of histogram-based KL
    w_dist = wasserstein_distance(real_tail, fake_tail)
    
    # 3. KL Divergence with correct Laplace smoothing (for comparison)
    min_val = min(real_tail.min(), fake_tail.min())
    max_val = max(real_tail.max(), fake_tail.max())
    bins = np.linspace(min_val, var_thresh, n_bins + 1)
    
    p_hist, _ = np.histogram(real_tail, bins=bins, density=True)
    q_hist, _ = np.histogram(fake_tail, bins=bins, density=True)
    
    # Apply correct Laplace smoothing
    p_smooth = apply_laplace_smoothing_correct(p_hist, epsilon=1e-6)
    q_smooth = apply_laplace_smoothing_correct(q_hist, epsilon=1e-6)
    
    kl_div = entropy(p_smooth, q_smooth)
    
    # 4. Kolmogorov-Smirnov statistic
    ks_stat, ks_pval = ks_2samp(real_tail, fake_tail)
    
    # 5. Mode Coverage (FIX Issue 34)
    # Divide tail into bins and check coverage
    tail_bins = np.linspace(real_tail.min(), real_tail.max(), 10)
    coverage = compute_mode_coverage(real_tail, fake_tail, tail_bins)
    
    return {
        'tail_recall': tail_recall,
        'tail_kl': kl_div,
        'tail_wasserstein': w_dist,
        'tail_ks': ks_stat,
        'tail_ks_pval': ks_pval,
        'tail_coverage': coverage
    }


def compute_mode_coverage(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    bins: np.ndarray,
    min_coverage_pct: float = 0.05
) -> float:
    """
    FIX Issue 34: Measure mode collapse by checking coverage of real data modes.
    
    Args:
        real_data: Real samples
        fake_data: Generated samples
        bins: Bin edges for mode definition
        min_coverage_pct: Minimum percentage of samples to consider mode covered
    
    Returns:
        Fraction of modes (bins) that are adequately covered
    """
    real_hist, _ = np.histogram(real_data, bins=bins)
    fake_hist, _ = np.histogram(fake_data, bins=bins)
    
    # Normalize to probabilities
    real_probs = real_hist / real_hist.sum()
    fake_probs = fake_hist / fake_hist.sum()
    
    # Count bins that have significant mass in real data and are covered in fake
    significant_bins = real_probs > 0.01  # Bins with >1% of real data
    covered = fake_probs[significant_bins] >= min_coverage_pct
    
    if significant_bins.sum() == 0:
        return 1.0  # All modes covered if no significant modes
    
    coverage = covered.sum() / significant_bins.sum()
    return coverage


def compute_extrapolation_metric_robust(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    percentile: float = 0.01
) -> Dict[str, float]:
    """
    FIX Issue 3: Robust extrapolation metric using percentiles across multiple runs.
    
    Instead of comparing single minimum values, compares extreme percentiles
    and measures systematic extrapolation.
    
    Args:
        real_data: Real data samples
        fake_data: Generated data samples (from single run)
        percentile: Percentile to use (0.01 = 1st percentile)
    
    Returns:
        Dictionary with extrapolation metrics
    """
    real_extreme = np.percentile(real_data, percentile * 100)
    fake_extreme = np.percentile(fake_data, percentile * 100)
    
    # Extrapolation: generated extreme is more extreme than historical
    extrapolates = fake_extreme < real_extreme
    extrap_magnitude = (real_extreme - fake_extreme) / abs(real_extreme) if extrapolates else 0.0
    
    # Also compute tail extension: how far beyond historical minimum
    real_min = real_data.min()
    fake_min = fake_data.min()
    tail_extension = (real_min - fake_min) / abs(real_min) if fake_min < real_min else 0.0
    
    return {
        'real_p01': real_extreme,
        'fake_p01': fake_extreme,
        'extrapolates': extrapolates,
        'extrap_pct': extrap_magnitude * 100,
        'tail_extension_pct': tail_extension * 100
    }


def test_extrapolation_significance(
    extrapolation_results: list,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    FIX Issue 17: Statistical test for systematic extrapolation across runs.
    
    Args:
        extrapolation_results: List of dicts from compute_extrapolation_metric_robust
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    extrap_flags = [r['extrapolates'] for r in extrapolation_results]
    extrap_pcts = [r['extrap_pct'] for r in extrapolation_results]
    
    n_total = len(extrap_flags)
    n_extrapolated = sum(extrap_flags)
    proportion = n_extrapolated / n_total if n_total > 0 else 0.0
    
    # Binomial test: is proportion significantly > 0.5?
    from scipy.stats import binom_test
    p_value = binom_test(n_extrapolated, n_total, 0.5, alternative='greater')
    
    mean_extrap = np.mean(extrap_pcts)
    std_extrap = np.std(extrap_pcts, ddof=1) if len(extrap_pcts) > 1 else 0.0
    
    return {
        'n_seeds': n_total,
        'n_extrapolated': n_extrapolated,
        'proportion': proportion,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_extrap_pct': mean_extrap,
        'std_extrap_pct': std_extrap
    }


def compute_calibration_metrics(
    real_data: np.ndarray,
    fake_data: np.ndarray
) -> Dict[str, float]:
    """
    FIX Issue 19: Comprehensive calibration metrics beyond tail.
    
    Checks if generated data preserves statistical properties:
    - Autocorrelation
    - Volatility clustering
    - Skewness
    - Kurtosis
    
    Args:
        real_data: Real time series
        fake_data: Generated time series
    
    Returns:
        Dictionary of calibration metrics
    """
    # 1. Autocorrelation of returns (should be near zero)
    real_acf = acf(real_data, nlags=5, fft=True)[1:]  # Skip lag 0
    fake_acf = acf(fake_data, nlags=5, fft=True)[1:]
    acf_error = np.mean(np.abs(real_acf - fake_acf))
    
    # 2. Autocorrelation of absolute returns (volatility clustering)
    real_abs_acf = acf(np.abs(real_data), nlags=5, fft=True)[1:]
    fake_abs_acf = acf(np.abs(fake_data), nlags=5, fft=True)[1:]
    abs_acf_error = np.mean(np.abs(real_abs_acf - fake_abs_acf))
    
    # 3. Skewness (should be negative for equity returns)
    from scipy.stats import skew
    real_skew = skew(real_data)
    fake_skew = skew(fake_data)
    skew_error = abs(real_skew - fake_skew)
    
    # 4. Excess Kurtosis
    real_kurt = kurtosis(real_data, fisher=True)
    fake_kurt = kurtosis(fake_data, fisher=True)
    kurt_error = abs(real_kurt - fake_kurt)
    
    return {
        'real_acf_mean': real_acf.mean(),
        'fake_acf_mean': fake_acf.mean(),
        'acf_error': acf_error,
        'real_abs_acf_mean': real_abs_acf.mean(),
        'fake_abs_acf_mean': fake_abs_acf.mean(),
        'abs_acf_error': abs_acf_error,
        'real_skewness': real_skew,
        'fake_skewness': fake_skew,
        'skewness_error': skew_error,
        'real_kurtosis': real_kurt,
        'fake_kurtosis': fake_kurt,
        'kurtosis_error': kurt_error
    }


def verify_spx_kurtosis(data: np.ndarray, expected_range: Tuple[float, float] = (5.0, 8.0)):
    """
    FIX Issue 21: Verify SPX kurtosis matches expected range from paper.
    
    Args:
        data: SPX return data
        expected_range: Expected kurtosis range
    
    Raises:
        Warning if kurtosis is outside expected range
    """
    kurt = kurtosis(data, fisher=True)
    if not (expected_range[0] <= kurt <= expected_range[1]):
        warnings.warn(
            f"SPX kurtosis {kurt:.2f} outside expected range {expected_range}. "
            "Data preprocessing may be incorrect."
        )
    else:
        print(f"SPX kurtosis validation passed: {kurt:.2f}")
    return kurt
