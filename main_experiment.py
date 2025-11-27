"""
Main experiment runner with complete reproducibility
Fixes Issues: 4, 8, 22, 23, 29, 30, 36, 37
"""

import numpy as np
import pandas as pd
import torch
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
from tqdm import tqdm

from config import CONFIG
from utils import set_seed, load_data_with_cache, train_val_split, verify_data_is_raw
from models import train_baseline_gan, train_curritail_gan, train_wgan_gp
from baselines import (
    historical_bootstrap_baseline,
    kde_baseline,
    evt_baseline,
    train_tailgan_static,
    train_importance_sampling_baseline,
)
from metrics import compute_tail_metrics, verify_spx_kurtosis
from statistical_tests import (
    compute_statistical_significance,
    compare_all_models,
    validate_against_paper_results,
)
from portfolio import compare_portfolio_strategies
from plotting import (
    plot_figure1_recovery,
    plot_figure3_ablation,
    plot_figure4_gradients,
    plot_figure5_kurtosis,
    plot_figure6_portfolio,
    plot_sensitivity_analysis,
    plot_qq_plots,
    plot_calibration_metrics,
)

warnings.filterwarnings("ignore")


def estimate_runtime() -> str:
    """
    Estimate total runtime based on configuration.
    """
    n_seeds = len(CONFIG.seeds)
    n_datasets = len(CONFIG.datasets)
    epochs = CONFIG.epochs

    # Rough estimates (minutes per model per seed)
    time_per_model = {
        "baseline": epochs / 200 * 3,  # ~3 min for 200 epochs
        "wgan": epochs / 200 * 4,  # Slower due to critic updates
        "curritail": epochs / 200 * 3.5,
        "tailgan": epochs / 200 * 3,
    }

    # GAN models trained on all seeds
    gan_time = sum(time_per_model.values()) * n_seeds * n_datasets

    # Expensive baselines only on first seed
    baseline_time = 2 * n_datasets  # KDE, Bootstrap, EVT, Importance

    # SPX-specific analyses
    spx_time = 0
    if "SPX" in CONFIG.datasets:
        spx_time = 120  # Sensitivity + portfolio (2 hours)

    total_minutes = gan_time + baseline_time + spx_time
    hours = int(total_minutes / 60)
    minutes = int(total_minutes % 60)

    return f"{hours}h {minutes}m"


def get_git_hash() -> str:
    """
    FIX Issue 37: Track git commit for reproducibility.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


def get_system_info() -> Dict:
    """
    FIX Issue 37: Record system information for reproducibility.
    """
    return {
        "python_version": subprocess.run(
            ["python", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def train_all_models_single_seed(
    train_data: np.ndarray, val_data: np.ndarray, seed: int, dataset_name: str
) -> Dict:
    """
    FIX Issue 8 & 22: Train all models for one seed with validation.

    Returns dict with generated samples and training histories.
    """
    print(f"  [Seed {seed}] Training all models...")

    set_seed(seed)
    results = {}

    # Baseline GAN
    print(f"    • Baseline GAN...")
    G_baseline, hist_baseline = train_baseline_gan(
        train_data=train_data,
        val_data=val_data,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        lr=CONFIG.learning_rate,
        seed=seed,
    )

    with torch.no_grad():
        z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
        gen_baseline = G_baseline(z).cpu().numpy().flatten()

    results["baseline"] = {
        "generated": gen_baseline,
        "history": hist_baseline,
        "model": G_baseline,
    }

    # WGAN-GP
    print(f"    • WGAN-GP...")
    G_wgan, hist_wgan = train_wgan_gp(
        train_data=train_data,
        val_data=val_data,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        lr=CONFIG.learning_rate,
        seed=seed,
    )

    with torch.no_grad():
        z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
        gen_wgan = G_wgan(z).cpu().numpy().flatten()

    results["wgan"] = {"generated": gen_wgan, "history": hist_wgan, "model": G_wgan}

    # CurriTail (main method)
    print(f"    • CurriTail...")
    G_curritail, hist_curritail = train_curritail_gan(
        train_data=train_data,
        val_data=val_data,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        alpha=CONFIG.alpha,
        k=CONFIG.curriculum_k,
        schedule=CONFIG.curriculum_schedule,
        lr=CONFIG.learning_rate,
        seed=seed,
    )

    with torch.no_grad():
        z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
        gen_curritail = G_curritail(z).cpu().numpy().flatten()

    results["curritail"] = {
        "generated": gen_curritail,
        "history": hist_curritail,
        "model": G_curritail,
    }

    # Tail-GAN static (ablation)
    print(f"    • Tail-GAN Static...")
    G_tailgan, hist_tailgan = train_tailgan_static(
        train_data=train_data,
        val_data=val_data,
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        alpha=CONFIG.alpha,
        lr=CONFIG.learning_rate,
        seed=seed,
    )

    with torch.no_grad():
        z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
        gen_tailgan = G_tailgan(z).cpu().numpy().flatten()

    results["tailgan"] = {
        "generated": gen_tailgan,
        "history": hist_tailgan,
        "model": G_tailgan,
    }

    # Only compute expensive baselines on first seed
    if seed == CONFIG.seeds[0]:
        # KDE
        print(f"    • KDE Baseline...")
        gen_kde = kde_baseline(train_data, n_samples=10000, seed=seed)
        results["kde"] = {"generated": gen_kde, "history": {}, "model": None}

        # Historical Bootstrap
        print(f"    • Bootstrap Baseline...")
        gen_bootstrap = historical_bootstrap_baseline(
            train_data, n_samples=10000, seed=seed
        )
        results["bootstrap"] = {
            "generated": gen_bootstrap,
            "history": {},
            "model": None,
        }

        # EVT
        print(f"    • EVT Baseline...")
        gen_evt = evt_baseline(train_data, n_samples=10000, threshold_q=0.05, seed=seed)
        results["evt"] = {"generated": gen_evt, "history": {}, "model": None}

        # Importance Sampling
        print(f"    • Importance Sampling...")
        G_importance, hist_importance = train_importance_sampling_baseline(
            train_data=train_data,
            val_data=val_data,
            epochs=CONFIG.epochs,
            batch_size=CONFIG.batch_size,
            tail_quantile=0.1,
            lr=CONFIG.learning_rate,
            seed=seed,
        )

        with torch.no_grad():
            z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
            gen_importance = G_importance(z).cpu().numpy().flatten()

        results["importance"] = {
            "generated": gen_importance,
            "history": hist_importance,
            "model": G_importance,
        }

    # Save models
    if hasattr(CONFIG, "save_models") and CONFIG.save_models:
        for model_name, model_dict in results.items():
            if model_dict["model"] is not None:
                save_path = (
                    CONFIG.models_dir / f"{model_name}_{dataset_name}_seed{seed}.pth"
                )
                torch.save(model_dict["model"].state_dict(), save_path)

    return results


def run_ablation_study(train_data: np.ndarray, val_data: np.ndarray, seed: int) -> Dict:
    """
    FIX Issue 28 & 35: Curriculum schedule ablation.
    """
    print(f"  [Ablation Study] Seed {seed}")

    ablation_results = {}

    for schedule in ["sigmoid", "linear", "step"]:
        print(f"    • Schedule: {schedule}")
        set_seed(seed)

        G, hist = train_curritail_gan(
            train_data=train_data,
            val_data=val_data,
            epochs=CONFIG.epochs,
            batch_size=CONFIG.batch_size,
            alpha=CONFIG.alpha,
            k=CONFIG.curriculum_k,
            schedule=schedule,
            lr=CONFIG.learning_rate,
            seed=seed,
        )

        with torch.no_grad():
            z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
            gen = G(z).cpu().numpy().flatten()

        ablation_results[schedule] = {"generated": gen, "history": hist}

    return ablation_results


def run_sensitivity_analysis(
    train_data: np.ndarray,
    val_data: np.ndarray,
    real_data: np.ndarray,
    n_seeds: int = 5,
) -> Dict:
    """
    FIX Issue 7 & 27: Hyperparameter sensitivity with multiple seeds.
    """
    print("  [Sensitivity Analysis]")

    sensitivity_results = {}

    # Alpha sensitivity
    print("    • Alpha sensitivity...")
    alpha_values = [0.5, 1.0, 3.0, 5.0, 10.0]
    alpha_metrics = {alpha: [] for alpha in alpha_values}

    for alpha in alpha_values:
        for seed_idx in range(n_seeds):
            seed = CONFIG.seeds[seed_idx]
            set_seed(seed)

            G, _ = train_curritail_gan(
                train_data=train_data,
                val_data=val_data,
                epochs=200,  # Faster for sensitivity
                batch_size=CONFIG.batch_size,
                alpha=alpha,
                k=CONFIG.curriculum_k,
                schedule="sigmoid",
                lr=CONFIG.learning_rate,
                seed=seed,
            )

            with torch.no_grad():
                z = torch.randn(5000, CONFIG.latent_dim).to(CONFIG.device)
                gen = G(z).cpu().numpy().flatten()

            metrics = compute_tail_metrics(real_data, gen, threshold_q=0.01)
            alpha_metrics[alpha].append(metrics["tail_kl"])

    sensitivity_results["alpha"] = alpha_metrics

    # K (steepness) sensitivity
    print("    • K (steepness) sensitivity...")
    k_values = [2, 6, 12, 18, 25]
    k_metrics = {k: [] for k in k_values}

    for k in k_values:
        for seed_idx in range(n_seeds):
            seed = CONFIG.seeds[seed_idx]
            set_seed(seed)

            G, _ = train_curritail_gan(
                train_data=train_data,
                val_data=val_data,
                epochs=200,
                batch_size=CONFIG.batch_size,
                alpha=CONFIG.alpha,
                k=k,
                schedule="sigmoid",
                lr=CONFIG.learning_rate,
                seed=seed,
            )

            with torch.no_grad():
                z = torch.randn(5000, CONFIG.latent_dim).to(CONFIG.device)
                gen = G(z).cpu().numpy().flatten()

            metrics = compute_tail_metrics(real_data, gen, threshold_q=0.01)
            k_metrics[k].append(metrics["tail_kl"])

    sensitivity_results["k"] = k_metrics

    # Batch size sensitivity
    print("    • Batch size sensitivity...")
    batch_sizes = [32, 64, 128, 256]
    batch_metrics = {bs: [] for bs in batch_sizes}

    for bs in batch_sizes:
        for seed_idx in range(n_seeds):
            seed = CONFIG.seeds[seed_idx]
            set_seed(seed)

            G, _ = train_curritail_gan(
                train_data=train_data,
                val_data=val_data,
                epochs=200,
                batch_size=bs,
                alpha=CONFIG.alpha,
                k=CONFIG.curriculum_k,
                schedule="sigmoid",
                lr=CONFIG.learning_rate,
                seed=seed,
            )

            with torch.no_grad():
                z = torch.randn(5000, CONFIG.latent_dim).to(CONFIG.device)
                gen = G(z).cpu().numpy().flatten()

            metrics = compute_tail_metrics(real_data, gen, threshold_q=0.01)
            batch_metrics[bs].append(metrics["tail_kl"])

    sensitivity_results["batch_size"] = batch_metrics

    return sensitivity_results


def run_experiment_for_dataset(dataset_name: str) -> Dict:
    """
    FIX Issue 29 & 30: Complete experiment pipeline for one dataset.
    """
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    # Load data
    print("Loading data...")
    _, _, raw_data = load_data_with_cache(dataset_name)

    # Verify it's raw (not scaled)
    verify_data_is_raw(raw_data)

    # Split train/val
    train_data, val_data = train_val_split(raw_data, train_ratio=CONFIG.train_ratio)
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Store results across seeds
    all_seed_results = []
    baseline_gradients = []
    curritail_gradients = []

    baseline_generated_all = []
    curritail_generated_all = []

    # Train across all seeds
    for seed in tqdm(CONFIG.seeds, desc=f"{dataset_name} - Training Seeds"):
        print(f"\n--- Seed {seed} ---")

        seed_results = train_all_models_single_seed(
            train_data=train_data,
            val_data=val_data,
            seed=seed,
            dataset_name=dataset_name,
        )

        # Collect for aggregation
        baseline_gradients.append(seed_results["baseline"]["history"]["gradient_norms"])
        curritail_gradients.append(
            seed_results["curritail"]["history"]["gradient_norms"]
        )

        baseline_generated_all.append(seed_results["baseline"]["generated"])
        curritail_generated_all.append(seed_results["curritail"]["generated"])

        # Compute metrics for this seed
        for model_name, model_dict in seed_results.items():
            gen_data = model_dict["generated"]
            metrics = compute_tail_metrics(raw_data, gen_data, threshold_q=0.01)

            all_seed_results.append(
                {"dataset": dataset_name, "model": model_name, "seed": seed, **metrics}
            )

    # Ablation study (only first seed)
    print("\n--- Ablation Study ---")
    ablation_results = run_ablation_study(
        train_data=train_data, val_data=val_data, seed=CONFIG.seeds[0]
    )

    # Sensitivity analysis (if SPX)
    sensitivity_results = None
    if dataset_name == "SPX":
        print("\n--- Sensitivity Analysis ---")
        sensitivity_results = run_sensitivity_analysis(
            train_data=train_data,
            val_data=val_data,
            real_data=raw_data,
            n_seeds=min(5, len(CONFIG.seeds)),
        )

    # Portfolio analysis (if SPX)
    portfolio_results = None
    if dataset_name == "SPX":
        print("\n--- Portfolio Analysis ---")

        baseline_gen = baseline_generated_all[0]
        curritail_gen = curritail_generated_all[0]

        portfolio_results = compare_portfolio_strategies(
            real_returns=raw_data,
            baseline_returns=baseline_gen,
            curritail_returns=curritail_gen,
            risk_free_rate=0.02,
        )

    # Statistical significance testing
    print("\n--- Statistical Testing ---")
    df_results = pd.DataFrame(all_seed_results)

    statistical_comparisons = compare_all_models(
        df_results, reference_model="curritail", metric="tail_kl"
    )

    # Validate against paper if SPX
    if dataset_name == "SPX":
        print("\n--- Validation Against Paper ---")
        validation = validate_against_paper_results(
            df_results, dataset_name=dataset_name
        )

        # SPX kurtosis check
        verify_spx_kurtosis(raw_data)

    return {
        "raw_data": raw_data,
        "train_data": train_data,
        "val_data": val_data,
        "all_seed_results": all_seed_results,
        "baseline_gradients": baseline_gradients,
        "curritail_gradients": curritail_gradients,
        "baseline_generated_all": baseline_generated_all,
        "curritail_generated_all": curritail_generated_all,
        "ablation_results": ablation_results,
        "sensitivity_results": sensitivity_results,
        "portfolio_results": portfolio_results,
        "statistical_comparisons": statistical_comparisons,
        "validation": validation if dataset_name == "SPX" else None,
    }


def generate_all_figures(experiment_results: Dict):
    """
    FIX Issue 16 & 28: Generate all publication figures.
    """
    print("\n" + "=" * 80)
    print("Generating Figures")
    print("=" * 80)

    for dataset_name, results in experiment_results.items():
        print(f"\n{dataset_name}:")

        # Figure 1: Recovery (all datasets)
        print("  • Figure 1: Tail Recovery")
        plot_figure1_recovery(
            real_data=results["raw_data"],
            baseline_data=results["baseline_generated_all"],
            curritail_data=results["curritail_generated_all"],
            output_path=str(CONFIG.figures_dir / f"Fig1_{dataset_name}_Recovery.png"),
        )

        # Figure 4: Gradients (all datasets)
        print("  • Figure 4: Gradient Dynamics")
        plot_figure4_gradients(
            baseline_gradients=results["baseline_gradients"],
            curritail_gradients=results["curritail_gradients"],
            output_path=str(CONFIG.figures_dir / f"Fig4_{dataset_name}_Gradients.png"),
        )

        # SPX-specific figures
        if dataset_name == "SPX":
            # Figure 3: Ablation
            print("  • Figure 3: Curriculum Ablation")
            plot_figure3_ablation(
                sigmoid_history=results["ablation_results"]["sigmoid"]["history"],
                linear_history=results["ablation_results"]["linear"]["history"],
                step_history=results["ablation_results"]["step"]["history"],
                output_path=str(CONFIG.figures_dir / "Fig3_Ablation.png"),
            )

            # Figure 5: Kurtosis (using statistical tests module)
            print("  • Figure 5: Kurtosis Comparison")
            from statistical_tests import kurtosis_comparison_bootstrap

            df_results = pd.DataFrame(results["all_seed_results"])
            kurtosis_results = {}

            for model_name in df_results["model"].unique():
                model_data = df_results[df_results["model"] == model_name]

                # Get generated samples for this model (first seed)
                first_seed_idx = 0
                if model_name == "baseline":
                    gen_sample = results["baseline_generated_all"][first_seed_idx]
                elif model_name == "curritail":
                    gen_sample = results["curritail_generated_all"][first_seed_idx]
                else:
                    continue  # Skip other models for kurtosis plot

                kurt_results = kurtosis_comparison_bootstrap(
                    results["raw_data"], gen_sample, n_bootstrap=1000
                )
                kurtosis_results[model_name.capitalize()] = kurt_results

            plot_figure5_kurtosis(
                kurtosis_results=kurtosis_results,
                output_path=str(CONFIG.figures_dir / "Fig5_Kurtosis.png"),
            )

            # Figure 6: Portfolio
            print("  • Figure 6: Portfolio Performance")
            plot_figure6_portfolio(
                portfolio_comparison=results["portfolio_results"],
                output_path=str(CONFIG.figures_dir / "Fig6_Portfolio.png"),
            )

            # Sensitivity plots
            if results["sensitivity_results"] is not None:
                print("  • Sensitivity: Alpha")
                plot_sensitivity_analysis(
                    sensitivity_results=results["sensitivity_results"]["alpha"],
                    param_name="Alpha (α)",
                    param_values=[0.5, 1.0, 3.0, 5.0, 10.0],
                    output_path=str(CONFIG.figures_dir / "Sensitivity_Alpha.png"),
                )

                print("  • Sensitivity: K")
                plot_sensitivity_analysis(
                    sensitivity_results=results["sensitivity_results"]["k"],
                    param_name="Steepness (k)",
                    param_values=[2, 6, 12, 18, 25],
                    output_path=str(CONFIG.figures_dir / "Sensitivity_K.png"),
                )

                print("  • Sensitivity: Batch Size")
                plot_sensitivity_analysis(
                    sensitivity_results=results["sensitivity_results"]["batch_size"],
                    param_name="Batch Size",
                    param_values=[32, 64, 128, 256],
                    output_path=str(CONFIG.figures_dir / "Sensitivity_BatchSize.png"),
                )

            # Q-Q plots
            print("  • Q-Q Plots")
            qq_dict = {
                "Baseline": results["baseline_generated_all"][0],
                "CurriTail": results["curritail_generated_all"][0],
            }
            plot_qq_plots(
                real_data=results["raw_data"],
                generated_data_dict=qq_dict,
                output_path=str(CONFIG.figures_dir / "QQ_Plots.png"),
            )

            # Calibration metrics
            print("  • Calibration Metrics")
            from metrics import compute_calibration_metrics

            calibration_results = {}
            for model_name in ["Baseline", "CurriTail"]:
                gen_data = qq_dict[model_name]
                calib = compute_calibration_metrics(results["raw_data"], gen_data)
                calibration_results[model_name] = calib

            plot_calibration_metrics(
                calibration_results=calibration_results,
                output_path=str(CONFIG.figures_dir / "Calibration_Metrics.png"),
            )


def save_final_results(experiment_results: Dict, metadata: Dict):
    """
    FIX Issue 37: Save complete results with metadata.
    """
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Aggregate all seed results
    all_results = []
    for dataset_name, results in experiment_results.items():
        all_results.extend(results["all_seed_results"])

    df_all = pd.DataFrame(all_results)

    # Summary statistics
    df_summary = (
        df_all.groupby(["dataset", "model"])
        .agg(
            {
                "tail_recall": ["mean", "std"],
                "tail_kl": ["mean", "std"],
                "tail_wasserstein": ["mean", "std"],
                "tail_ks": ["mean", "std"],
                "tail_coverage": ["mean", "std"],
            }
        )
        .round(4)
    )

    # Save CSV
    results_path = CONFIG.outputs_dir / f"results_all_seeds_{timestamp}.csv"
    df_all.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    summary_path = CONFIG.outputs_dir / f"summary_{timestamp}.csv"
    df_summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    # Save metadata
    metadata["timestamp"] = timestamp
    metadata["config"] = CONFIG.to_dict()

    metadata_path = CONFIG.outputs_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Mean ± Std across seeds)")
    print("=" * 80)
    print(df_summary)

    return df_all, df_summary


def main():
    """
    FIX Issue 4, 8, 29, 30, 36, 37: Complete reproducible experiment.
    """
    print("=" * 80)
    print("CurriTail-GAN: Publication-Ready Experimental Suite")
    print("=" * 80)

    # Record metadata
    metadata = {
        "git_hash": get_git_hash(),
        "system_info": get_system_info(),
        "start_time": datetime.now().isoformat(),
    }

    print(f"\nGit Hash: {metadata['git_hash']}")
    print(f"Device: {CONFIG.device}")
    print(f"Seeds: {CONFIG.seeds[:3]}...{CONFIG.seeds[-1]} ({len(CONFIG.seeds)} total)")
    print(f"Datasets: {CONFIG.datasets}")
    print(f"Epochs: {CONFIG.epochs}")
    print(f"Batch Size: {CONFIG.batch_size}")
    print(f"\n⏱️  Estimated Runtime: {estimate_runtime()}")
    print(f"{'=' * 80}\n")

    # Run experiments
    experiment_results = {}
    for dataset_name in CONFIG.datasets:
        experiment_results[dataset_name] = run_experiment_for_dataset(dataset_name)

    # Generate all figures
    generate_all_figures(experiment_results)

    # Save results
    metadata["end_time"] = datetime.now().isoformat()
    df_all, df_summary = save_final_results(experiment_results, metadata)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {CONFIG.outputs_dir}")
    print(f"Figures saved to: {CONFIG.figures_dir}")
    print(f"Models saved to: {CONFIG.models_dir}")

    return experiment_results, df_all, df_summary


if __name__ == "__main__":
    results, df_all, df_summary = main()
