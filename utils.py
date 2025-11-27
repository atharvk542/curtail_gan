"""
Utility functions for data loading, caching, and preprocessing
Fixes Issues: 2, 9, 10, 14, 30
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from datetime import datetime, timedelta
import warnings
from typing import Tuple, Optional
import json


def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    import torch
    import random

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_rarity_scores_correct(
    batch_np: np.ndarray, sorted_full_data: np.ndarray
) -> np.ndarray:
    """
    Correctly compute rarity scores using binary search on sorted data.

    FIX Issue 2: Properly handle unsorted batch with searchsorted.
    Uses vectorized comparison instead of searchsorted to avoid sorting requirement.

    Rarity score: r(x) = 2 * |0.5 - F(x)| where F(x) is empirical CDF
    For tail focus, we use: r(x) = 1 - F(x) for lower tail

    Args:
        batch_np: Unsorted batch of samples, shape (n,)
        sorted_full_data: Sorted reference data for computing CDF, shape (m,)

    Returns:
        Rarity scores in [0, 1], shape (n,)
    """
    batch_np = batch_np.flatten()

    # Vectorized rank computation that works with unsorted batch
    # For each sample in batch, count how many values in sorted_full_data are less than it
    batch_reshaped = batch_np.reshape(-1, 1)  # (n, 1)
    sorted_reshaped = sorted_full_data.reshape(1, -1)  # (1, m)

    # Count values less than each batch sample
    ranks = (batch_reshaped > sorted_reshaped).sum(axis=1)  # (n,)

    # Compute empirical CDF
    cdf = ranks / len(sorted_full_data)

    # Rarity for lower tail (focus on crashes)
    rarity = 1.0 - cdf

    return rarity


def apply_laplace_smoothing_correct(
    histogram: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """
    Apply Laplace smoothing correctly to ensure probability constraint.

    FIX Issue 10: Correct formula that preserves sum to 1.

    Args:
        histogram: Raw histogram counts or densities
        epsilon: Smoothing parameter

    Returns:
        Smoothed probability distribution that sums to 1
    """
    smoothed = histogram + epsilon
    smoothed = smoothed / smoothed.sum()  # Correct normalization

    # Verify probability constraint
    assert np.abs(smoothed.sum() - 1.0) < 1e-6, (
        "Smoothing failed probability constraint"
    )

    return smoothed


def load_data_with_cache(
    dataset_name: str, cache_dir: str = "data_cache", cache_max_age_days: int = 7
) -> Tuple[np.ndarray, RobustScaler, np.ndarray]:
    """
    Load financial data with caching and error handling.

    FIX Issue 14: Implements caching and robust error handling for yfinance.

    Args:
        dataset_name: "Synthetic", "SPX", or "BTC"
        cache_dir: Directory for cached data
        cache_max_age_days: Maximum age of cache in days

    Returns:
        Tuple of (scaled_data, scaler, raw_data)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_data.npz")
    cache_meta_file = os.path.join(cache_dir, f"{dataset_name}_meta.json")

    # Check if valid cache exists
    use_cache = False
    if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
        try:
            with open(cache_meta_file, "r") as f:
                meta = json.load(f)
            cache_time = datetime.fromisoformat(meta["timestamp"])
            if datetime.now() - cache_time < timedelta(days=cache_max_age_days):
                use_cache = True
                print(
                    f"Loading {dataset_name} from cache (age: {(datetime.now() - cache_time).days} days)"
                )
        except Exception as e:
            print(f"Cache validation failed: {e}. Re-downloading.")

    if use_cache:
        try:
            data = np.load(cache_file)
            raw_values = data["raw"]
            scaled_values = data["scaled"]

            # Reconstruct scaler
            scaler = RobustScaler()
            scaler.center_ = data["scaler_center"]
            scaler.scale_ = data["scaler_scale"]

            return scaled_values, scaler, raw_values
        except Exception as e:
            print(f"Cache loading failed: {e}. Re-downloading.")

    # Load or generate data
    scaler = RobustScaler()

    if dataset_name == "Synthetic":
        # Mixture of Gaussians: 95% Normal, 5% Crash
        n_total = 10000
        n_body = int(0.95 * n_total)
        n_tail = n_total - n_body

        body = np.random.normal(0, 1, n_body)
        tail = np.random.normal(-5, 1.5, n_tail)  # More extreme tail

        raw_values = np.concatenate([body, tail])
        np.random.shuffle(raw_values)

    elif dataset_name == "SPX":
        raw_values = _download_yfinance_with_retry(
            "^GSPC", "2000-01-01", "2023-01-01", dataset_name
        )

    elif dataset_name == "BTC":
        raw_values = _download_yfinance_with_retry(
            "BTC-USD", "2014-01-01", "2023-01-01", dataset_name
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Scale data
    scaled_values = scaler.fit_transform(raw_values.reshape(-1, 1)).flatten()

    # Save to cache
    try:
        np.savez(
            cache_file,
            raw=raw_values,
            scaled=scaled_values,
            scaler_center=scaler.center_,
            scaler_scale=scaler.scale_,
        )
        with open(cache_meta_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "dataset": dataset_name,
                    "n_samples": len(raw_values),
                },
                f,
            )
        print(f"Cached {dataset_name} data to {cache_file}")
    except Exception as e:
        print(f"Failed to cache data: {e}")

    return scaled_values, scaler, raw_values


def _download_yfinance_with_retry(
    ticker: str, start: str, end: str, dataset_name: str, max_retries: int = 3
) -> np.ndarray:
    """
    Download data from yfinance with retry logic and fallback.

    Args:
        ticker: Yahoo Finance ticker symbol
        start: Start date string
        end: End date string
        dataset_name: Name for error messages
        max_retries: Maximum retry attempts

    Returns:
        Array of log returns
    """
    for attempt in range(max_retries):
        try:
            print(f"Downloading {ticker} (attempt {attempt + 1}/{max_retries})...")
            df = yf.download(ticker, start=start, end=end, progress=False)

            if df.empty:
                raise ValueError(f"Downloaded data is empty for {ticker}")

            # Handle different column naming conventions
            price_col = None
            if "Adj Close" in df.columns:
                price_col = "Adj Close"
            elif "Close" in df.columns:
                price_col = "Close"
            else:
                # Multi-index case
                for col in df.columns:
                    if "Adj Close" in str(col) or "Close" in str(col):
                        price_col = col
                        break

            if price_col is None:
                raise ValueError(f"Could not find price column in {df.columns}")

            # Calculate weekly log returns
            df_resampled = df[price_col].resample("W").last()
            returns = np.log(df_resampled / df_resampled.shift(1)).dropna()

            # Handle DataFrame vs Series
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]

            raw_values = returns.values.flatten()

            if len(raw_values) < 100:
                raise ValueError(f"Insufficient data: only {len(raw_values)} samples")

            print(f"Successfully downloaded {len(raw_values)} weekly returns")
            return raw_values

        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(
                    f"All download attempts failed for {dataset_name}. Using synthetic fallback."
                )
                warnings.warn(
                    f"Could not download {dataset_name}, using synthetic data instead"
                )
                # Return synthetic data as fallback
                n = 2000
                synthetic = np.random.normal(-0.001, 0.02, int(0.95 * n))
                synthetic = np.concatenate(
                    [synthetic, np.random.normal(-0.05, 0.03, int(0.05 * n))]
                )
                np.random.shuffle(synthetic)
                return synthetic


def train_val_split(
    data: np.ndarray, train_ratio: float = 0.8, preserve_order: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into train and validation sets.

    FIX Issue 11: Implements proper train/val split.
    For time series, preserves temporal order to avoid lookahead bias.

    Args:
        data: Full dataset
        train_ratio: Fraction for training
        preserve_order: If True, takes first train_ratio% chronologically

    Returns:
        Tuple of (train_data, val_data)
    """
    n = len(data)
    n_train = int(n * train_ratio)

    if preserve_order:
        # Chronological split for time series
        train_data = data[:n_train]
        val_data = data[n_train:]
    else:
        # Random split for synthetic data
        indices = np.random.permutation(n)
        train_data = data[indices[:n_train]]
        val_data = data[indices[n_train:]]

    return train_data, val_data


def verify_data_is_raw(data: np.ndarray) -> bool:
    """
    FIX Issue 30: Verify data is raw (not scaled) before metric computation.

    Args:
        data: Array to check

    Returns:
        True if data appears to be raw, False if scaled
    """
    mean_abs = np.abs(data.mean())
    std = data.std()

    # Scaled data should have mean ≈ 0 and std ≈ 1
    # Raw financial returns have |mean| < 0.01 and std in range 0.01-0.10
    is_scaled = (mean_abs < 0.1) and (0.5 < std < 1.5)

    return not is_scaled
