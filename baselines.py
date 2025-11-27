"""
Additional baseline models for comparison
Fixes Issues: 18, 25, 26
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KernelDensity
from scipy.stats import genpareto
from typing import Tuple, Dict
from tqdm import tqdm

from config import CONFIG
from models import Generator, Discriminator, prepare_dataloader
from utils import set_seed, compute_rarity_scores_correct


def historical_bootstrap_baseline(
    data: np.ndarray, n_samples: int = 10000, seed: int = 42
) -> np.ndarray:
    """
    FIX Issue 18: Simple bootstrap baseline - resample with replacement.

    Args:
        data: Historical data
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Bootstrapped samples
    """
    np.random.seed(seed)
    return np.random.choice(data, size=n_samples, replace=True)


def kde_baseline(
    data: np.ndarray, n_samples: int = 10000, bandwidth: float = 0.1, seed: int = 42
) -> np.ndarray:
    """
    FIX Issue 18: Kernel Density Estimation baseline.

    Args:
        data: Training data
        n_samples: Number of samples to generate
        bandwidth: KDE bandwidth
        seed: Random seed

    Returns:
        Samples from fitted KDE
    """
    np.random.seed(seed)

    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(data.reshape(-1, 1))

    samples = kde.sample(n_samples, random_state=seed)
    return samples.flatten()


def evt_baseline(
    data: np.ndarray, n_samples: int = 10000, threshold_q: float = 0.05, seed: int = 42
) -> np.ndarray:
    """
    FIX Issue 29: EVT baseline with proper seed control.

    Fits Generalized Pareto Distribution to tail exceedances.

    Args:
        data: Historical data
        n_samples: Number of samples to generate
        threshold_q: Quantile threshold for tail
        seed: Random seed

    Returns:
        Generated samples mixing tail (GPD) and body (bootstrap)
    """
    np.random.seed(seed)

    thresh = np.quantile(data, threshold_q)
    exceedances = thresh - data[data <= thresh]

    if len(exceedances) < 10:
        # Not enough tail data, fall back to bootstrap
        return historical_bootstrap_baseline(data, n_samples, seed)

    try:
        # Fit GPD to exceedances
        params = genpareto.fit(exceedances)

        # Generate mixture: threshold_q fraction from tail, rest from body
        n_tail = int(n_samples * threshold_q)
        n_body = n_samples - n_tail

        # Sample from GPD and convert back to original scale
        tail_samples = thresh - genpareto.rvs(*params, size=n_tail, random_state=seed)

        # Sample body from non-tail data
        body_data = data[data > thresh]
        body_samples = np.random.choice(body_data, size=n_body, replace=True)

        # Combine and shuffle
        all_samples = np.concatenate([tail_samples, body_samples])
        np.random.shuffle(all_samples)

        return all_samples

    except Exception as e:
        print(f"EVT fitting failed: {e}. Falling back to bootstrap.")
        return historical_bootstrap_baseline(data, n_samples, seed)


def train_tailgan_static(
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int,
    batch_size: int,
    alpha: float,
    lr: float,
    seed: int,
) -> Tuple[Generator, Dict]:
    """
    FIX Issue 25: Tail-GAN with static weights (no curriculum).

    This is CurriTail with S_t = 1.0 always, to show curriculum helps.

    Args:
        train_data: Training data
        val_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        alpha: Rarity weight coefficient
        lr: Learning rate
        seed: Random seed

    Returns:
        Tuple of (trained generator, history dict)
    """
    set_seed(seed)

    # Compute sorted data for rarity scores
    sorted_data = np.sort(train_data)

    device = torch.device(CONFIG.device)

    G = Generator(CONFIG.latent_dim, CONFIG.g_hidden_dims).to(device)
    D = Discriminator(CONFIG.d_hidden_dims, CONFIG.dropout_rate, use_sigmoid=True).to(
        device
    )

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))

    train_loader = prepare_dataloader(train_data, batch_size, device, shuffle=True)

    history = {"train_loss_d": [], "train_loss_g": [], "val_metrics": [], "epoch": []}

    # Static S_t = 1.0 (full rarity weighting from start)
    S_t = 1.0

    for epoch in tqdm(range(epochs), desc="Tail-GAN Static", leave=False):
        G.train()
        D.train()

        epoch_loss_d = []
        epoch_loss_g = []

        for batch_tuple in train_loader:
            real = batch_tuple[0]
            bs = real.size(0)

            # Compute rarity scores
            real_np = real.cpu().numpy().flatten()
            rarity_np = compute_rarity_scores_correct(real_np, sorted_data)
            rarity = torch.FloatTensor(rarity_np).unsqueeze(1).to(device)

            # Static weights (no curriculum)
            weights = 1.0 + alpha * S_t * rarity

            # Train Discriminator
            opt_D.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)

            real_pred = D(real)
            fake_pred = D(fake.detach())

            d_loss = (
                -(weights * torch.log(real_pred + 1e-8)).mean()
                - torch.log(1 - fake_pred + 1e-8).mean()
            )
            d_loss.backward()
            opt_D.step()

            epoch_loss_d.append(d_loss.item())

            # Train Generator
            opt_G.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)
            fake_pred = D(fake)

            g_loss = -torch.log(fake_pred + 1e-8).mean()
            g_loss.backward()
            opt_G.step()

            epoch_loss_g.append(g_loss.item())

        history["train_loss_d"].append(np.mean(epoch_loss_d))
        history["train_loss_g"].append(np.mean(epoch_loss_g))
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            G.eval()
            with torch.no_grad():
                z_val = torch.randn(len(val_data), CONFIG.latent_dim, device=device)
                gen_val = G(z_val).cpu().numpy().flatten()

            from metrics import compute_tail_metrics

            val_metrics = compute_tail_metrics(
                val_data, gen_val, CONFIG.tail_threshold_q
            )
            history["val_metrics"].append(val_metrics)

    return G, history


def train_importance_sampling_baseline(
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int,
    batch_size: int,
    tail_quantile: float,
    lr: float,
    seed: int,
) -> Tuple[Generator, Dict]:
    """
    FIX Issue 18: Importance sampling baseline - GAN with tail-focused sampling.

    Args:
        train_data: Training data
        val_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        tail_quantile: Quantile threshold for tail definition
        lr: Learning rate
        seed: Random seed

    Returns:
        Tuple of (trained generator, history dict)
    """
    set_seed(seed)

    device = torch.device(CONFIG.device)

    # Train a standard GAN with importance-sampled batches
    G = Generator(CONFIG.latent_dim, CONFIG.g_hidden_dims).to(device)
    D = Discriminator(CONFIG.d_hidden_dims, CONFIG.dropout_rate, use_sigmoid=True).to(
        device
    )

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))

    # Identify tail and body samples
    thresh = np.quantile(train_data, tail_quantile)
    tail_mask = train_data <= thresh
    tail_data = train_data[tail_mask]
    body_data = train_data[~tail_mask]

    history = {"train_loss_d": [], "train_loss_g": [], "val_metrics": [], "epoch": []}

    for epoch in tqdm(range(epochs), desc="Importance Sampling", leave=False):
        G.train()
        D.train()

        epoch_loss_d = []
        epoch_loss_g = []

        # Create batches with 50% tail oversampling
        n_batches = len(train_data) // batch_size

        for _ in range(n_batches):
            # Half tail, half body
            n_tail = batch_size // 2
            n_body = batch_size - n_tail

            tail_samples = tail_data[
                np.random.choice(len(tail_data), n_tail, replace=True)
            ]
            body_samples = body_data[
                np.random.choice(len(body_data), n_body, replace=True)
            ]

            batch = np.concatenate([tail_samples, body_samples])
            np.random.shuffle(batch)

            real = torch.FloatTensor(batch).unsqueeze(1).to(device)
            bs = real.size(0)

            # Train Discriminator
            opt_D.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)

            real_pred = D(real)
            fake_pred = D(fake.detach())

            d_loss = (
                -torch.log(real_pred + 1e-8).mean()
                - torch.log(1 - fake_pred + 1e-8).mean()
            )
            d_loss.backward()
            opt_D.step()

            epoch_loss_d.append(d_loss.item())

            # Train Generator
            opt_G.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)
            fake_pred = D(fake)

            g_loss = -torch.log(fake_pred + 1e-8).mean()
            g_loss.backward()
            opt_G.step()

            epoch_loss_g.append(g_loss.item())

        history["train_loss_d"].append(np.mean(epoch_loss_d))
        history["train_loss_g"].append(np.mean(epoch_loss_g))
        history["epoch"].append(epoch)

        if epoch % 10 == 0:
            G.eval()
            with torch.no_grad():
                z_val = torch.randn(len(val_data), CONFIG.latent_dim, device=device)
                gen_val = G(z_val).cpu().numpy().flatten()

            from metrics import compute_tail_metrics

            val_metrics = compute_tail_metrics(
                val_data, gen_val, CONFIG.tail_threshold_q
            )
            history["val_metrics"].append(val_metrics)

    return G, history
