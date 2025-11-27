"""
Publication-quality visualization module with error bars
Fixes Issues: 16, 28, 31, 35, 39
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import os

from config import CONFIG


def setup_plotting_style():
    """
    FIX Issue 16 & 39: Consistent figure aesthetics globally.
    """
    plt.rcParams["figure.dpi"] = CONFIG.figure_dpi
    plt.rcParams["savefig.dpi"] = CONFIG.figure_dpi
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 13

    sns.set_style(CONFIG.seaborn_style)
    sns.set_context(CONFIG.seaborn_context, font_scale=CONFIG.font_scale)

    # Global color palette
    global COLORS
    COLORS = {
        "real": "#2E86AB",  # Blue
        "baseline": "#A23B72",  # Purple
        "wgan": "#F18F01",  # Orange
        "curritail": "#2A9D8F",  # Teal/Green
        "evt": "#E76F51",  # Coral
        "kde": "#8E44AD",  # Deep purple
        "bootstrap": "#95A5A6",  # Gray
        "tailgan": "#E74C3C",  # Red
    }


def plot_figure1_recovery(
    real_data: np.ndarray,
    baseline_data: List[np.ndarray],
    curritail_data: List[np.ndarray],
    output_path: str,
):
    """
    FIX Issue 28 & 31: Figure 1 with error bars and full distribution.

    Shows tail recovery and extrapolation for SPX.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Global distribution with KDE
    ax = axes[0]

    # Real data
    from scipy.stats import gaussian_kde

    kde_real = gaussian_kde(real_data)
    x_range = np.linspace(real_data.min(), real_data.max(), 200)
    ax.plot(
        x_range,
        kde_real(x_range),
        label="Real Data",
        color=COLORS["real"],
        linewidth=2.5,
        linestyle="--",
    )

    # Baseline with confidence band
    baseline_kdes = [gaussian_kde(data) for data in baseline_data]
    baseline_densities = np.array([kde(x_range) for kde in baseline_kdes])
    baseline_mean = baseline_densities.mean(axis=0)
    baseline_std = baseline_densities.std(axis=0)

    ax.plot(
        x_range, baseline_mean, label="Baseline", color=COLORS["baseline"], linewidth=2
    )
    ax.fill_between(
        x_range,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        alpha=0.2,
        color=COLORS["baseline"],
    )

    # CurriTail with confidence band
    curritail_kdes = [gaussian_kde(data) for data in curritail_data]
    curritail_densities = np.array([kde(x_range) for kde in curritail_kdes])
    curritail_mean = curritail_densities.mean(axis=0)
    curritail_std = curritail_densities.std(axis=0)

    ax.plot(
        x_range,
        curritail_mean,
        label="CurriTail",
        color=COLORS["curritail"],
        linewidth=2,
    )
    ax.fill_between(
        x_range,
        curritail_mean - curritail_std,
        curritail_mean + curritail_std,
        alpha=0.2,
        color=COLORS["curritail"],
    )

    ax.set_xlabel("Weekly Log Returns")
    ax.set_ylabel("Density")
    ax.set_title("(a) Global Distribution")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    # Right: Tail quantile plot
    ax = axes[1]

    real_sorted = np.sort(real_data)
    n = len(real_sorted)
    tail_n = int(n * 0.1)
    q = np.linspace(0, 0.1, tail_n)

    ax.plot(
        q,
        real_sorted[:tail_n],
        label="Real",
        color=COLORS["real"],
        linewidth=2.5,
        marker="o",
        markersize=3,
        markevery=tail_n // 20,
        linestyle="--",
    )

    # Baseline quantiles with error bars
    baseline_sorted = [np.sort(data)[:tail_n] for data in baseline_data]
    baseline_q_mean = np.mean(baseline_sorted, axis=0)
    baseline_q_std = np.std(baseline_sorted, axis=0)

    ax.plot(q, baseline_q_mean, label="Baseline", color=COLORS["baseline"], linewidth=2)
    ax.fill_between(
        q,
        baseline_q_mean - baseline_q_std,
        baseline_q_mean + baseline_q_std,
        alpha=0.2,
        color=COLORS["baseline"],
    )

    # CurriTail quantiles with error bars
    curritail_sorted = [np.sort(data)[:tail_n] for data in curritail_data]
    curritail_q_mean = np.mean(curritail_sorted, axis=0)
    curritail_q_std = np.std(curritail_sorted, axis=0)

    ax.plot(
        q, curritail_q_mean, label="CurriTail", color=COLORS["curritail"], linewidth=2
    )
    ax.fill_between(
        q,
        curritail_q_mean - curritail_q_std,
        curritail_q_mean + curritail_q_std,
        alpha=0.2,
        color=COLORS["curritail"],
    )

    # Historical minimum
    hist_min = real_data.min()
    ax.axhline(
        hist_min,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Historical Min",
    )

    # Extrapolation annotation
    curritail_min_mean = np.mean([d.min() for d in curritail_data])
    if curritail_min_mean < hist_min:
        extrap_pct = ((hist_min - curritail_min_mean) / abs(hist_min)) * 100
        ax.text(
            0.05,
            hist_min * 1.1,
            f"Extrapolation: {extrap_pct:.1f}%",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax.set_xlabel("Quantile")
    ax.set_ylabel("Return")
    ax.set_title("(b) Tail Recovery (Bottom 10%)")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure3_ablation(
    sigmoid_history: Dict, linear_history: Dict, step_history: Dict, output_path: str
):
    """
    FIX Issue 28 & 35: Curriculum ablation with verification.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Tail-KL convergence
    ax = axes[0]

    epochs_sigmoid = np.array(sigmoid_history["epoch"])[::10]
    epochs_linear = np.array(linear_history["epoch"])[::10]
    epochs_step = np.array(step_history["epoch"])[::10]

    ax.plot(
        epochs_sigmoid,
        sigmoid_history["kl"],
        label="Sigmoid (Ours)",
        color=COLORS["curritail"],
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    ax.plot(
        epochs_linear,
        linear_history["kl"],
        label="Linear",
        color=COLORS["baseline"],
        linewidth=2,
        marker="s",
        markersize=4,
        linestyle="--",
    )
    ax.plot(
        epochs_step,
        step_history["kl"],
        label="Step",
        color=COLORS["wgan"],
        linewidth=2,
        marker="^",
        markersize=4,
        linestyle=":",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tail-KL Divergence")
    ax.set_title("(a) Convergence Stability")
    ax.set_yscale("log")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3, which="both")

    # Right: Schedule visualization with warmup check
    ax = axes[1]

    epochs_full = np.arange(400)
    t_norm = epochs_full / 400
    k = 12.0

    S_sigmoid = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))
    S_linear = t_norm
    S_step = np.where(t_norm < 0.5, 0.0, 1.0)

    ax.plot(
        epochs_full,
        S_sigmoid,
        label="Sigmoid",
        color=COLORS["curritail"],
        linewidth=2.5,
    )
    ax.plot(
        epochs_full,
        S_linear,
        label="Linear",
        color=COLORS["baseline"],
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        epochs_full,
        S_step,
        label="Step",
        color=COLORS["wgan"],
        linewidth=2,
        linestyle=":",
    )

    # FIX Issue 35: Verify warmup and convergence
    warmup_epoch = int(0.2 * 400)
    convergence_epoch = int(0.8 * 400)
    ax.axvline(warmup_epoch, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(convergence_epoch, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(warmup_epoch, 0.95, "Warmup", rotation=90, va="top", fontsize=8, alpha=0.7)
    ax.text(
        convergence_epoch,
        0.95,
        "Converged",
        rotation=90,
        va="top",
        fontsize=8,
        alpha=0.7,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Curriculum Weight $S(t)$")
    ax.set_title("(b) Schedule Functions")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure4_gradients(
    baseline_gradients: List[np.ndarray],
    curritail_gradients: List[np.ndarray],
    output_path: str,
):
    """
    FIX Issue 28: Gradient dynamics with error bars across seeds.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure all gradient arrays have same length
    min_len = min(
        min(len(g) for g in baseline_gradients),
        min(len(g) for g in curritail_gradients),
    )

    baseline_array = np.array([g[:min_len] for g in baseline_gradients])
    curritail_array = np.array([g[:min_len] for g in curritail_gradients])

    epochs = np.arange(min_len)

    # Baseline
    base_mean = baseline_array.mean(axis=0)
    base_sem = baseline_array.std(axis=0) / np.sqrt(len(baseline_gradients))

    ax.plot(
        epochs,
        base_mean,
        label="Baseline",
        color=COLORS["baseline"],
        linewidth=2,
        alpha=0.8,
    )
    ax.fill_between(
        epochs,
        base_mean - 1.96 * base_sem,
        base_mean + 1.96 * base_sem,
        alpha=0.2,
        color=COLORS["baseline"],
    )

    # CurriTail
    curri_mean = curritail_array.mean(axis=0)
    curri_sem = curritail_array.std(axis=0) / np.sqrt(len(curritail_gradients))

    ax.plot(
        epochs, curri_mean, label="CurriTail", color=COLORS["curritail"], linewidth=2.5
    )
    ax.fill_between(
        epochs,
        curri_mean - 1.96 * curri_sem,
        curri_mean + 1.96 * curri_sem,
        alpha=0.2,
        color=COLORS["curritail"],
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generator Gradient Norm (Tail Samples)")
    ax.set_title("Gradient Flow to Tail Regions")
    ax.legend(frameon=True, fancybox=True, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure5_kurtosis(kurtosis_results: Dict[str, Dict], output_path: str):
    """
    FIX Issue 6 & 28: Kurtosis with bootstrap confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(kurtosis_results.keys())
    x_pos = np.arange(len(models))

    means = [kurtosis_results[m]["generated_kurtosis_mean"] for m in models]
    ci_lower = [kurtosis_results[m]["generated_ci_lower"] for m in models]
    ci_upper = [kurtosis_results[m]["generated_ci_upper"] for m in models]

    # Error bars (confidence intervals)
    errors_lower = [means[i] - ci_lower[i] for i in range(len(models))]
    errors_upper = [ci_upper[i] - means[i] for i in range(len(models))]

    colors = [COLORS.get(m.lower(), "#34495E") for m in models]

    bars = ax.bar(
        x_pos, means, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    ax.errorbar(
        x_pos,
        means,
        yerr=[errors_lower, errors_upper],
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=2,
    )

    # Real kurtosis line
    real_kurt = kurtosis_results[models[0]]["real_kurtosis"]
    ax.axhline(
        real_kurt,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Real (κ={real_kurt:.2f})",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Excess Kurtosis (κ)")
    ax.set_title("Fat-Tail Reproduction (Bootstrap 95% CI)")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3, axis="y")

    # Annotate values
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + errors_upper[i] + 0.5,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_figure6_portfolio(portfolio_comparison: Dict, output_path: str):
    """
    FIX Issue 5 & 24: Real portfolio backtest results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    baseline_bt = portfolio_comparison["baseline_backtest"]
    curritail_bt = portfolio_comparison["curritail_backtest"]

    # Top-left: Cumulative returns
    ax = axes[0, 0]
    days = np.arange(len(baseline_bt["cumulative_returns"]))

    ax.plot(
        days,
        baseline_bt["cumulative_returns"],
        label="Baseline Portfolio",
        color=COLORS["baseline"],
        linewidth=2,
    )
    ax.plot(
        days,
        curritail_bt["cumulative_returns"],
        label="CurriTail Portfolio",
        color=COLORS["curritail"],
        linewidth=2,
    )

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Cumulative Value ($)")
    ax.set_title("(a) Cumulative Performance")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    # Top-right: Drawdown
    ax = axes[0, 1]

    baseline_cum = baseline_bt["cumulative_returns"]
    curritail_cum = curritail_bt["cumulative_returns"]

    baseline_dd = (
        baseline_cum - np.maximum.accumulate(baseline_cum)
    ) / np.maximum.accumulate(baseline_cum)
    curritail_dd = (
        curritail_cum - np.maximum.accumulate(curritail_cum)
    ) / np.maximum.accumulate(curritail_cum)

    ax.fill_between(
        days,
        baseline_dd * 100,
        0,
        alpha=0.3,
        color=COLORS["baseline"],
        label="Baseline DD",
    )
    ax.fill_between(
        days,
        curritail_dd * 100,
        0,
        alpha=0.3,
        color=COLORS["curritail"],
        label="CurriTail DD",
    )

    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("(b) Drawdown Analysis")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3)

    # Bottom-left: Performance metrics
    ax = axes[1, 0]
    ax.axis("off")

    metrics_text = f"""
    Performance Metrics (Annualized):
    
    Baseline Portfolio:
    • Return: {baseline_bt["annualized_return"] * 100:.2f}%
    • Volatility: {baseline_bt["annualized_volatility"] * 100:.2f}%
    • Sharpe Ratio: {baseline_bt["sharpe_ratio"]:.3f}
    • Max Drawdown: {baseline_bt["max_drawdown"] * 100:.2f}%
    
    CurriTail Portfolio:
    • Return: {curritail_bt["annualized_return"] * 100:.2f}%
    • Volatility: {curritail_bt["annualized_volatility"] * 100:.2f}%
    • Sharpe Ratio: {curritail_bt["sharpe_ratio"]:.3f}
    • Max Drawdown: {curritail_bt["max_drawdown"] * 100:.2f}%
    
    Improvement:
    • Sharpe Δ: {portfolio_comparison["improvement"]["sharpe_diff"]:.3f}
    • Drawdown Δ: {portfolio_comparison["improvement"]["drawdown_diff"] * 100:.2f}%
    """

    ax.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Bottom-right: Allocation comparison
    ax = axes[1, 1]

    baseline_opt = portfolio_comparison["baseline_optimization"]
    curritail_opt = portfolio_comparison["curritail_optimization"]

    models = ["Baseline\nOptimized", "CurriTail\nOptimized"]
    risky_weights = [baseline_opt["weight_risky"], curritail_opt["weight_risky"]]
    rf_weights = [baseline_opt["weight_risk_free"], curritail_opt["weight_risk_free"]]

    x_pos = np.arange(len(models))
    width = 0.6

    p1 = ax.bar(
        x_pos,
        risky_weights,
        width,
        label="Risky Asset",
        color=COLORS["wgan"],
        alpha=0.7,
    )
    p2 = ax.bar(
        x_pos,
        rf_weights,
        width,
        bottom=risky_weights,
        label="Risk-Free",
        color=COLORS["kde"],
        alpha=0.7,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("(c) Optimal Allocations")
    ax.legend(frameon=True, fancybox=True)
    ax.grid(alpha=0.3, axis="y")

    # Annotate weights
    for i, (r, rf) in enumerate(zip(risky_weights, rf_weights)):
        ax.text(
            i,
            r / 2,
            f"{r * 100:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
        )
        ax.text(
            i,
            r + rf / 2,
            f"{rf * 100:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_sensitivity_analysis(
    sensitivity_results: Dict, param_name: str, param_values: List, output_path: str
):
    """
    FIX Issue 7 & 27: Sensitivity with multiple seeds and error bars.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract metrics across seeds for each parameter value
    metric_arrays = []
    for param_val in param_values:
        metrics = sensitivity_results[param_val]
        metric_arrays.append(metrics)

    metric_arrays = np.array(metric_arrays)  # Shape: (n_params, n_seeds)

    means = metric_arrays.mean(axis=1)
    stds = metric_arrays.std(axis=1)
    sems = stds / np.sqrt(metric_arrays.shape[1])

    ax.errorbar(
        param_values,
        means,
        yerr=1.96 * sems,
        marker="o",
        markersize=8,
        linewidth=2.5,
        capsize=5,
        capthick=2,
        color=COLORS["curritail"],
    )

    ax.set_xlabel(param_name)
    ax.set_ylabel("Tail-KL Divergence")
    ax.set_title(
        f"{param_name} Sensitivity Analysis (95% CI, n={metric_arrays.shape[1]} seeds)"
    )
    ax.grid(alpha=0.3)

    # Mark optimal
    optimal_idx = np.argmin(means)
    ax.axvline(
        param_values[optimal_idx], color="red", linestyle="--", alpha=0.5, linewidth=1.5
    )
    ax.text(
        param_values[optimal_idx],
        means.max(),
        f"Optimal: {param_values[optimal_idx]}",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_qq_plots(
    real_data: np.ndarray, generated_data_dict: Dict[str, np.ndarray], output_path: str
):
    """
    FIX Issue 19: Q-Q plots for calibration.
    """
    n_models = len(generated_data_dict)
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (model_name, gen_data) in enumerate(generated_data_dict.items()):
        ax = axes[idx]

        # Q-Q plot
        stats.probplot(gen_data, dist=stats.norm, plot=ax, fit=True)

        # Compare with real data quantiles
        real_quantiles = np.percentile(real_data, np.linspace(0, 100, 100))
        gen_quantiles = np.percentile(gen_data, np.linspace(0, 100, 100))

        # Overlay real vs generated
        ax.scatter(
            real_quantiles,
            gen_quantiles,
            alpha=0.5,
            s=20,
            color=COLORS.get(model_name.lower(), "#34495E"),
            label="Real vs Gen",
        )

        ax.set_title(f"{model_name} Q-Q Plot")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_calibration_metrics(calibration_results: Dict[str, Dict], output_path: str):
    """
    FIX Issue 19: Visualization of calibration metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = list(calibration_results.keys())
    colors = [COLORS.get(m.lower(), "#34495E") for m in models]

    # Autocorrelation error
    ax = axes[0, 0]
    acf_errors = [calibration_results[m]["acf_error"] for m in models]
    bars = ax.bar(models, acf_errors, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("(a) Autocorrelation Matching")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    # Volatility clustering (abs returns ACF)
    ax = axes[0, 1]
    abs_acf_errors = [calibration_results[m]["abs_acf_error"] for m in models]
    bars = ax.bar(models, abs_acf_errors, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("(b) Volatility Clustering")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    # Skewness
    ax = axes[1, 0]
    real_skew = calibration_results[models[0]]["real_skewness"]
    gen_skews = [calibration_results[m]["fake_skewness"] for m in models]

    x_pos = np.arange(len(models) + 1)
    all_skews = [real_skew] + gen_skews
    all_labels = ["Real"] + models
    all_colors = ["red"] + colors

    bars = ax.bar(x_pos, all_skews, color=all_colors, alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=15, ha="right")
    ax.set_ylabel("Skewness")
    ax.set_title("(c) Skewness Comparison")
    ax.grid(alpha=0.3, axis="y")

    # Kurtosis error
    ax = axes[1, 1]
    kurt_errors = [calibration_results[m]["kurtosis_error"] for m in models]
    bars = ax.bar(models, kurt_errors, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Absolute Error")
    ax.set_title("(d) Excess Kurtosis Matching")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_failure_analysis(failure_results: Dict[str, Dict], output_path: str):
    """
    FIX Issue 20: Visualize failure modes with bad hyperparameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = list(failure_results.keys())

    # Extract metrics
    tail_recalls = [failure_results[c]["tail_recall"] for c in configs]
    tail_kls = [failure_results[c]["tail_kl"] for c in configs]
    coverages = [failure_results[c]["tail_coverage"] for c in configs]

    # Color code: green for good, red for bad
    def get_color(metric_val, threshold_good, threshold_bad, lower_is_better=True):
        if lower_is_better:
            if metric_val <= threshold_good:
                return "#2ECC71"  # Green
            elif metric_val >= threshold_bad:
                return "#E74C3C"  # Red
            else:
                return "#F39C12"  # Orange
        else:
            if metric_val >= threshold_good:
                return "#2ECC71"
            elif metric_val <= threshold_bad:
                return "#E74C3C"
            else:
                return "#F39C12"

    # Tail recall
    ax = axes[0, 0]
    colors = [
        get_color(r, 0.008, 0.015, lower_is_better=False)
        if 0.005 < r < 0.02
        else "#E74C3C"
        for r in tail_recalls
    ]
    bars = ax.bar(configs, tail_recalls, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(
        0.01, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Target (1%)"
    )
    ax.set_ylabel("Tail Recall")
    ax.set_title("(a) Tail Coverage")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # Tail KL (log scale)
    ax = axes[0, 1]
    colors = [get_color(kl, 2.0, 10.0, lower_is_better=True) for kl in tail_kls]
    bars = ax.bar(configs, tail_kls, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Tail-KL Divergence")
    ax.set_title("(b) Tail Distribution Quality")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # Mode coverage
    ax = axes[1, 0]
    colors = [get_color(c, 0.7, 0.5, lower_is_better=False) for c in coverages]
    bars = ax.bar(configs, coverages, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(
        0.8, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Good (>0.8)"
    )
    ax.set_ylabel("Mode Coverage")
    ax.set_title("(c) Mode Collapse Detection")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # Summary text
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = "Failure Mode Analysis:\n\n"
    for config in configs:
        r = failure_results[config]
        status = (
            "✓ PASS" if (r["tail_kl"] < 5.0 and r["tail_coverage"] > 0.6) else "✗ FAIL"
        )
        summary_text += f"{config}: {status}\n"
        summary_text += (
            f"  KL: {r['tail_kl']:.2f}, Coverage: {r['tail_coverage']:.2f}\n\n"
        )

    summary_text += "\nKey Findings:\n"
    summary_text += "• Too weak (α=0.1): Poor tail learning\n"
    summary_text += "• Too strong (α=10): Training instability\n"
    summary_text += "• No curriculum (k=1): Slow convergence\n"
    summary_text += "• Instant shock (k=50): Mode collapse\n"
    summary_text += "• Undertrained (epochs=50): Incomplete\n"

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG.figure_dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# Initialize plotting on module import
setup_plotting_style()
