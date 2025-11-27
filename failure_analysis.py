"""
Failure analysis module to test robustness with bad hyperparameters
Fixes Issue: 20
"""

import numpy as np
import torch
from typing import Dict

from config import CONFIG
from utils import set_seed, load_data_with_cache, train_val_split
from models import train_curritail_gan
from metrics import compute_tail_metrics, compute_mode_coverage


def test_failure_mode(
    train_data: np.ndarray,
    val_data: np.ndarray,
    real_data: np.ndarray,
    config_name: str,
    alpha: float,
    k: float,
    epochs: int,
    batch_size: int,
    seed: int = 42,
) -> Dict:
    """
    Test a specific configuration and measure degradation.

    Args:
        train_data: Training data
        val_data: Validation data
        real_data: Full real data for comparison
        config_name: Descriptive name for this configuration
        alpha: Curriculum strength parameter
        k: Curriculum steepness parameter
        epochs: Number of training epochs
        batch_size: Batch size
        seed: Random seed

    Returns:
        Dict with configuration name and all metrics
    """
    print(f"  Testing: {config_name}")

    set_seed(seed)

    try:
        G, history = train_curritail_gan(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            alpha=alpha,
            k=k,
            schedule="sigmoid",
            lr=CONFIG.learning_rate,
            seed=seed,
        )

        # Generate samples
        with torch.no_grad():
            z = torch.randn(10000, CONFIG.latent_dim).to(CONFIG.device)
            gen_data = G(z).cpu().numpy().flatten()

        # Compute metrics
        tail_metrics = compute_tail_metrics(real_data, gen_data, threshold_q=0.01)
        mode_coverage = compute_mode_coverage(real_data, gen_data, n_bins=20)

        # Check for training issues
        val_losses = history.get("val_loss", [])
        converged = len(val_losses) > 0 and val_losses[-1] < 10.0

        result = {
            "config_name": config_name,
            "alpha": alpha,
            "k": k,
            "epochs": epochs,
            "batch_size": batch_size,
            "tail_recall": tail_metrics["tail_recall"],
            "tail_kl": tail_metrics["tail_kl"],
            "tail_wasserstein": tail_metrics["tail_wasserstein"],
            "tail_ks": tail_metrics["tail_ks"],
            "tail_coverage": tail_metrics["tail_coverage"],
            "mode_coverage": mode_coverage,
            "converged": converged,
            "final_val_loss": val_losses[-1] if len(val_losses) > 0 else float("inf"),
            "status": "success",
        }

    except Exception as e:
        print(f"    ERROR: {str(e)}")
        result = {
            "config_name": config_name,
            "alpha": alpha,
            "k": k,
            "epochs": epochs,
            "batch_size": batch_size,
            "tail_recall": 0.0,
            "tail_kl": float("inf"),
            "tail_wasserstein": float("inf"),
            "tail_ks": 1.0,
            "tail_coverage": 0.0,
            "mode_coverage": 0.0,
            "converged": False,
            "final_val_loss": float("inf"),
            "status": f"failed: {str(e)}",
        }

    return result


def run_failure_analysis(dataset_name: str = "SPX") -> Dict:
    """
    FIX Issue 20: Test bad hyperparameters to show degradation.

    Tests:
    1. Too weak curriculum (alpha too small)
    2. Too strong curriculum (alpha too large)
    3. Too slow curriculum (k too small)
    4. Too fast curriculum (k too large)
    5. Undertrained (too few epochs)
    6. Wrong batch size (too small/large)

    Returns:
        Dict mapping config names to results
    """
    print("=" * 80)
    print("Failure Analysis: Testing Bad Hyperparameters")
    print("=" * 80)

    # Load data
    print(f"\nLoading {dataset_name} data...")
    raw_data = load_data_with_cache(dataset_name)
    train_data, val_data = train_val_split(raw_data, val_ratio=CONFIG.val_ratio)

    seed = CONFIG.random_seeds[0]

    # Define failure configurations
    configs = [
        # Baseline (good config)
        {
            "config_name": "Baseline (Good)",
            "alpha": 3.0,
            "k": 12.0,
            "epochs": 400,
            "batch_size": 64,
        },
        # Alpha failures
        {
            "config_name": "Alpha Too Weak (0.1)",
            "alpha": 0.1,
            "k": 12.0,
            "epochs": 400,
            "batch_size": 64,
        },
        {
            "config_name": "Alpha Too Strong (10.0)",
            "alpha": 10.0,
            "k": 12.0,
            "epochs": 400,
            "batch_size": 64,
        },
        # K (steepness) failures
        {
            "config_name": "K Too Slow (1.0)",
            "alpha": 3.0,
            "k": 1.0,
            "epochs": 400,
            "batch_size": 64,
        },
        {
            "config_name": "K Too Fast (50.0)",
            "alpha": 3.0,
            "k": 50.0,
            "epochs": 400,
            "batch_size": 64,
        },
        # Training failures
        {
            "config_name": "Undertrained (50 epochs)",
            "alpha": 3.0,
            "k": 12.0,
            "epochs": 50,
            "batch_size": 64,
        },
        {
            "config_name": "Overtrained (800 epochs)",
            "alpha": 3.0,
            "k": 12.0,
            "epochs": 800,
            "batch_size": 64,
        },
        # Batch size failures
        {
            "config_name": "Batch Too Small (16)",
            "alpha": 3.0,
            "k": 12.0,
            "epochs": 400,
            "batch_size": 16,
        },
        {
            "config_name": "Batch Too Large (512)",
            "alpha": 3.0,
            "k": 12.0,
            "epochs": 400,
            "batch_size": 512,
        },
        # Combined failures
        {
            "config_name": "Multiple Bad (α=0.1, k=1, epochs=50)",
            "alpha": 0.1,
            "k": 1.0,
            "epochs": 50,
            "batch_size": 64,
        },
    ]

    # Run all tests
    results = {}
    for config in configs:
        result = test_failure_mode(
            train_data=train_data,
            val_data=val_data,
            real_data=raw_data,
            seed=seed,
            **config,
        )
        results[config["config_name"]] = result

    # Print summary
    print("\n" + "=" * 80)
    print("Failure Analysis Summary")
    print("=" * 80)
    print(f"{'Configuration':<40} {'Tail-KL':<12} {'Coverage':<10} {'Status'}")
    print("-" * 80)

    for config_name, result in results.items():
        tail_kl = result["tail_kl"]
        coverage = result["tail_coverage"]
        status = result["status"]

        # Format output
        if tail_kl == float("inf"):
            kl_str = "INF"
        else:
            kl_str = f"{tail_kl:.2f}"

        print(f"{config_name:<40} {kl_str:<12} {coverage:<10.2f} {status}")

    # Analysis
    print("\n" + "=" * 80)
    print("Key Findings")
    print("=" * 80)

    baseline_kl = results["Baseline (Good)"]["tail_kl"]

    print(f"\nBaseline (Good Config): Tail-KL = {baseline_kl:.2f}")

    for config_name, result in results.items():
        if config_name == "Baseline (Good)":
            continue

        kl = result["tail_kl"]
        if kl != float("inf"):
            degradation = ((kl - baseline_kl) / baseline_kl) * 100
            print(f"\n{config_name}:")
            print(f"  • Tail-KL: {kl:.2f} ({degradation:+.1f}% vs baseline)")
            print(f"  • Mode Coverage: {result['mode_coverage']:.2f}")
            print(f"  • Converged: {result['converged']}")

            if degradation > 50:
                print(f"  ⚠ SEVERE DEGRADATION")
            elif degradation > 20:
                print(f"  ⚠ Moderate degradation")
            elif degradation < -10:
                print(f"  ✓ Unexpectedly good (possible overfitting)")
        else:
            print(f"\n{config_name}:")
            print(f"  ✗ TRAINING FAILED")

    return results


def save_failure_analysis_results(results: Dict, output_path: str):
    """
    Save failure analysis results to JSON.
    """
    import json

    # Convert to serializable format
    serializable_results = {}
    for config_name, result in results.items():
        serializable_results[config_name] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in result.items()
            if k != "status" or isinstance(v, str)
        }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nSaved failure analysis to: {output_path}")


if __name__ == "__main__":
    # Run failure analysis
    results = run_failure_analysis(dataset_name="SPX")

    # Save results
    output_path = CONFIG.outputs_dir / "failure_analysis_results.json"
    save_failure_analysis_results(results, str(output_path))

    # Generate visualization
    from plotting import plot_failure_analysis

    plot_failure_analysis(
        failure_results=results,
        output_path=str(CONFIG.figures_dir / "Failure_Analysis.png"),
    )

    print("\nFailure analysis complete!")
