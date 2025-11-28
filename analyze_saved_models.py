"""
Analyze saved model checkpoints without retraining
Loads .pth files and generates samples for statistical analysis
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List
import warnings
from tqdm import tqdm

from config import CONFIG
from utils import set_seed, load_data_with_cache, train_val_split
from models import Generator
from metrics import compute_tail_metrics
from statistical_tests import compare_all_models, validate_against_paper_results
from plotting import (
    plot_figure1_recovery,
    plot_figure4_gradients,
    plot_qq_plots,
    plot_calibration_metrics,
)

warnings.filterwarnings("ignore")


def load_model_from_checkpoint(model_path: Path, latent_dim: int = 10) -> Generator:
    """Load a saved Generator model from checkpoint"""
    G = Generator(
        latent_dim=latent_dim,
        hidden_dims=CONFIG.g_hidden_dims,
        dropout_rate=CONFIG.dropout_rate,
    ).to(CONFIG.device)

    G.load_state_dict(torch.load(model_path, map_location=CONFIG.device))
    G.eval()
    return G


def generate_samples_from_model(model: Generator, n_samples: int = 10000) -> np.ndarray:
    """Generate samples from a trained model"""
    with torch.no_grad():
        z = torch.randn(n_samples, CONFIG.latent_dim).to(CONFIG.device)
        samples = model(z).cpu().numpy().flatten()
    return samples


def find_model_files(models_dir: Path, dataset_name: str) -> Dict[str, List[Path]]:
    """
    Find all model checkpoint files for a dataset.

    Returns:
        Dict mapping model_name -> list of checkpoint paths (one per seed)
    """
    models_dir = Path(models_dir)

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    model_files = {
        "baseline": [],
        "wgan": [],
        "curritail": [],
        "tailgan": [],
        "importance": [],
    }

    # Find all .pth files for this dataset
    for model_name in model_files.keys():
        pattern = f"{model_name}_{dataset_name}_seed*.pth"
        files = sorted(models_dir.glob(pattern))
        model_files[model_name] = files
        print(f"  Found {len(files)} checkpoint(s) for {model_name}")

    return model_files


def analyze_dataset_from_checkpoints(
    dataset_name: str,
    models_dir: Path = Path("saved_models"),
) -> Dict:
    """
    Analyze a dataset using saved model checkpoints.

    Args:
        dataset_name: Name of dataset (e.g., "Synthetic", "SPX", "BTC")
        models_dir: Directory containing .pth checkpoint files

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'=' * 80}")
    print(f"Analyzing Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    # Load real data
    print("Loading data...")
    _, _, raw_data = load_data_with_cache(dataset_name)
    train_data, val_data = train_val_split(raw_data, train_ratio=CONFIG.train_ratio)
    print(f"Real data: {len(raw_data)} samples")

    # Find checkpoint files
    print("\nFinding model checkpoints...")
    model_files = find_model_files(models_dir, dataset_name)

    # Check if we have any models
    total_models = sum(len(files) for files in model_files.values())
    if total_models == 0:
        raise FileNotFoundError(
            f"No model checkpoints found for {dataset_name} in {models_dir}\n"
            f"Expected files like: baseline_{dataset_name}_seed42.pth"
        )

    # Generate samples from each model checkpoint
    print("\nGenerating samples from checkpoints...")
    all_seed_results = []
    baseline_generated_all = []
    curritail_generated_all = []

    # Get list of seeds from checkpoint filenames
    baseline_files = model_files["baseline"]
    seeds_found = []
    for filepath in baseline_files:
        # Extract seed number from filename like "baseline_SPX_seed42.pth"
        seed_str = filepath.stem.split("seed")[-1]
        try:
            seed = int(seed_str)
            seeds_found.append(seed)
        except ValueError:
            print(f"Warning: Could not extract seed from {filepath.name}")

    seeds_found = sorted(seeds_found)
    print(
        f"Found checkpoints for {len(seeds_found)} seeds: {seeds_found[:5]}{'...' if len(seeds_found) > 5 else ''}"
    )

    # Process each seed
    for seed in tqdm(seeds_found, desc=f"{dataset_name} - Loading Models"):
        set_seed(seed)

        for model_name in ["baseline", "wgan", "curritail", "tailgan", "importance"]:
            # Find checkpoint for this seed
            checkpoint_path = models_dir / f"{model_name}_{dataset_name}_seed{seed}.pth"

            if not checkpoint_path.exists():
                print(f"  Warning: Missing {model_name} checkpoint for seed {seed}")
                continue

            # Load model and generate samples
            try:
                model = load_model_from_checkpoint(checkpoint_path)
                gen_samples = generate_samples_from_model(model, n_samples=10000)

                # Compute metrics
                metrics = compute_tail_metrics(raw_data, gen_samples, threshold_q=0.01)

                all_seed_results.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        **metrics,
                    }
                )

                # Store for plotting
                if model_name == "baseline":
                    baseline_generated_all.append(gen_samples)
                elif model_name == "curritail":
                    curritail_generated_all.append(gen_samples)

            except Exception as e:
                print(f"  Error loading {model_name} seed {seed}: {e}")

    # Statistical analysis
    print("\n--- Statistical Testing ---")
    df_results = pd.DataFrame(all_seed_results)

    if len(df_results) == 0:
        print("No results to analyze!")
        return None

    print(f"\nResults summary:")
    print(df_results.groupby("model")["tail_kl"].agg(["count", "mean", "std"]))

    statistical_comparisons = compare_all_models(
        df_results, reference_model="curritail"
    )

    print("\n--- Statistical Comparisons ---")
    print(statistical_comparisons)

    # Validate against paper if SPX
    validation = None
    if dataset_name == "SPX":
        print("\n--- Validation Against Paper ---")
        validation = validate_against_paper_results(
            df_results, dataset_name=dataset_name
        )
        print(validation)

    return {
        "raw_data": raw_data,
        "all_seed_results": all_seed_results,
        "df_results": df_results,
        "baseline_generated_all": baseline_generated_all,
        "curritail_generated_all": curritail_generated_all,
        "statistical_comparisons": statistical_comparisons,
        "validation": validation,
    }


def generate_figures_from_results(results: Dict, dataset_name: str):
    """Generate plots from analysis results"""
    print(f"\n--- Generating Figures for {dataset_name} ---")

    if not results or len(results["baseline_generated_all"]) == 0:
        print("Insufficient data for plotting")
        return

    # Figure 1: Recovery
    print("  • Tail Recovery Plot")
    plot_figure1_recovery(
        real_data=results["raw_data"],
        baseline_data=results["baseline_generated_all"],
        curritail_data=results["curritail_generated_all"],
        output_path=str(CONFIG.figures_dir / f"Fig1_{dataset_name}_Recovery.png"),
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
        output_path=str(CONFIG.figures_dir / f"QQ_{dataset_name}.png"),
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
        output_path=str(CONFIG.figures_dir / f"Calibration_{dataset_name}.png"),
    )


def main(
    models_dir: str = "saved_models",
    datasets: List[str] = None,
    generate_plots: bool = True,
):
    """
    Main analysis function for saved models.

    Args:
        models_dir: Directory containing .pth checkpoint files
        datasets: List of dataset names to analyze (default: from CONFIG)
        generate_plots: Whether to generate visualization plots
    """
    models_dir = Path(models_dir)

    if datasets is None:
        datasets = CONFIG.datasets

    print("=" * 80)
    print("CurriTail-GAN: Checkpoint Analysis")
    print("=" * 80)
    print(f"Models directory: {models_dir}")
    print(f"Datasets: {datasets}")
    print(f"Device: {CONFIG.device}")
    print("=" * 80)

    all_results = {}

    for dataset_name in datasets:
        try:
            results = analyze_dataset_from_checkpoints(
                dataset_name=dataset_name,
                models_dir=models_dir,
            )

            if results:
                all_results[dataset_name] = results

                if generate_plots:
                    generate_figures_from_results(results, dataset_name)

        except Exception as e:
            print(f"\nError analyzing {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save combined results
    if all_results:
        print("\n" + "=" * 80)
        print("Saving Combined Results")
        print("=" * 80)

        all_seed_results = []
        for dataset_name, results in all_results.items():
            all_seed_results.extend(results["all_seed_results"])

        df_all = pd.DataFrame(all_seed_results)

        # Summary statistics
        df_summary = (
            df_all.groupby(["dataset", "model"])
            .agg(
                {
                    "tail_recall": ["mean", "std", "count"],
                    "tail_kl": ["mean", "std"],
                    "tail_wasserstein": ["mean", "std"],
                    "tail_ks": ["mean", "std"],
                    "tail_coverage": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Save
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_path = CONFIG.outputs_dir / f"checkpoint_analysis_{timestamp}.csv"
        df_all.to_csv(results_path, index=False)
        print(f"Saved: {results_path}")

        summary_path = CONFIG.outputs_dir / f"checkpoint_summary_{timestamp}.csv"
        df_summary.to_csv(summary_path)
        print(f"Saved: {summary_path}")

        print("\n" + "=" * 80)
        print("FINAL RESULTS (Mean ± Std across seeds)")
        print("=" * 80)
        print(df_summary)

        return all_results, df_all, df_summary
    else:
        print("\nNo results to save.")
        return None, None, None


if __name__ == "__main__":
    # Example usage:
    # python analyze_saved_models.py

    # For Colab, you might run:
    # results, df_all, df_summary = main(
    #     models_dir="/content/drive/MyDrive/CurriTail_GAN/saved_models",
    #     datasets=["SPX"],  # or ["Synthetic", "SPX", "BTC"]
    # )

    results, df_all, df_summary = main()
