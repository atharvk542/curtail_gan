"""
Unit tests for critical functions
Fixes Issue: 38
"""

import numpy as np
import pytest
from scipy.stats import kurtosis

from utils import (
    compute_rarity_scores_correct,
    apply_laplace_smoothing_correct,
    verify_data_is_raw,
)
from metrics import compute_mode_coverage
from config import ExperimentConfig


def test_rarity_monotonic():
    """Test that rarity scores increase with extremeness"""
    sorted_data = np.sort(np.random.randn(1000))

    # Test samples from different quantiles
    q10 = np.percentile(sorted_data, 10)
    q50 = np.percentile(sorted_data, 50)

    batch = np.array([q10, q50])
    rarity = compute_rarity_scores_correct(batch, sorted_data)

    # Lower quantile should have higher rarity
    assert rarity[0] > rarity[1], "Rarity should be higher for more extreme values"
    assert 0 <= rarity.min() <= rarity.max() <= 1, "Rarity should be in [0, 1]"


def test_rarity_unsorted_batch():
    """Test that rarity computation works with unsorted batch"""
    sorted_data = np.sort(np.random.randn(1000))

    # Unsorted batch
    batch = np.array([0.5, -2.0, 1.0, -3.0, 0.0])
    rarity = compute_rarity_scores_correct(batch, sorted_data)

    # More extreme values should have higher rarity
    assert rarity[3] > rarity[1], "-3.0 should be rarer than -2.0"
    assert rarity[3] > rarity[2], "-3.0 should be rarer than 1.0"
    assert len(rarity) == len(batch), "Rarity should have same length as batch"


def test_laplace_smoothing_sum():
    """Test that Laplace smoothing preserves probability constraint"""
    histogram = np.array([10, 5, 0, 20, 15])

    smoothed = apply_laplace_smoothing_correct(histogram, epsilon=1e-6)

    assert np.abs(smoothed.sum() - 1.0) < 1e-5, "Smoothed histogram should sum to 1"
    assert np.all(smoothed > 0), "All bins should have positive probability"


def test_laplace_smoothing_with_zeros():
    """Test Laplace smoothing with zero bins"""
    histogram = np.array([100, 0, 0, 0, 50])

    smoothed = apply_laplace_smoothing_correct(histogram)

    assert smoothed[1] > 0, "Zero bins should get positive mass"
    assert smoothed[0] > smoothed[1], "Original high bins should remain higher"
    assert np.abs(smoothed.sum() - 1.0) < 1e-5, "Should sum to 1"


def test_metrics_on_identical():
    """Metrics should be zero/perfect when real equals fake"""
    data = np.random.randn(1000)

    from metrics import compute_tail_metrics

    metrics = compute_tail_metrics(data, data.copy(), threshold_q=0.01)

    # Tail recall should equal threshold
    assert abs(metrics["tail_recall"] - 0.01) < 0.02, (
        "Tail recall should match threshold"
    )

    # Wasserstein distance should be near zero
    assert metrics["tail_wasserstein"] < 0.1, (
        "Wasserstein should be near zero for identical distributions"
    )

    # KS p-value should be high (distributions are same)
    assert metrics["tail_ks_pval"] > 0.05, (
        "KS test should not reject when distributions are identical"
    )


def test_schedule_bounds():
    """Test that curriculum schedule has correct bounds"""
    config = ExperimentConfig()
    config.curriculum_k = 12.0
    epochs = 300

    # Sigmoid schedule
    t_start = 0 / epochs
    t_mid = 0.5
    t_end = 1.0

    S_start = 1.0 / (1.0 + np.exp(-config.curriculum_k * (t_start - 0.5)))
    S_mid = 1.0 / (1.0 + np.exp(-config.curriculum_k * (t_mid - 0.5)))
    S_end = 1.0 / (1.0 + np.exp(-config.curriculum_k * (t_end - 0.5)))

    assert S_start < 0.1, "Schedule should start near 0"
    assert 0.4 < S_mid < 0.6, "Schedule should be around 0.5 at midpoint"
    assert S_end > 0.9, "Schedule should end near 1"


def test_verify_data_raw_vs_scaled():
    """Test detection of raw vs scaled data"""
    # Raw financial returns
    raw_data = np.random.normal(-0.001, 0.02, 1000)
    assert verify_data_is_raw(raw_data), "Should detect raw financial data"

    # Scaled data (mean≈0, std≈1)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data.reshape(-1, 1)).flatten()
    assert not verify_data_is_raw(scaled_data), "Should detect scaled data"


def test_mode_coverage():
    """Test mode coverage metric"""
    real_data = np.concatenate([np.random.randn(800), np.random.randn(200) - 3])

    # Good coverage
    fake_good = np.concatenate([np.random.randn(800), np.random.randn(200) - 3])
    bins = np.linspace(real_data.min(), real_data.max(), 10)
    coverage_good = compute_mode_coverage(real_data, fake_good, bins)

    # Poor coverage (mode collapse to center)
    fake_poor = np.random.randn(1000)
    coverage_poor = compute_mode_coverage(real_data, fake_poor, bins)

    assert coverage_good > coverage_poor, "Good model should have better coverage"
    assert 0 <= coverage_good <= 1, "Coverage should be in [0, 1]"


def test_curriculum_schedule_comparison():
    """Test different curriculum schedules"""
    epochs = 300
    k = 12.0

    schedules = {"sigmoid": [], "linear": [], "step": []}

    for epoch in range(epochs):
        t = epoch / epochs

        schedules["sigmoid"].append(1.0 / (1.0 + np.exp(-k * (t - 0.5))))
        schedules["linear"].append(t)
        schedules["step"].append(0.0 if t < 0.5 else 1.0)

    # Sigmoid should be smooth
    sigmoid_grad = np.gradient(schedules["sigmoid"])
    assert np.all(sigmoid_grad >= 0), "Sigmoid should be monotonically increasing"

    # Step should have discontinuity
    step_array = np.array(schedules["step"])
    assert step_array[0] == 0.0, "Step should start at 0"
    assert step_array[-1] == 1.0, "Step should end at 1"

    # Linear should be strictly linear
    linear_array = np.array(schedules["linear"])
    assert np.allclose(linear_array, np.linspace(0, 1, epochs)), (
        "Linear should be linear"
    )


if __name__ == "__main__":
    # Run tests
    test_rarity_monotonic()
    test_rarity_unsorted_batch()
    test_laplace_smoothing_sum()
    test_laplace_smoothing_with_zeros()
    test_metrics_on_identical()
    test_schedule_bounds()
    test_verify_data_raw_vs_scaled()
    test_mode_coverage()
    test_curriculum_schedule_comparison()

    print("All tests passed!")
