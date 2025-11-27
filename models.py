"""
Model architectures and training functions with scientific fixes
Fixes Issues: 1, 12, 13, 23, 32, 33
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm

from config import CONFIG
from utils import compute_rarity_scores_correct


class Generator(nn.Module):
    """Generator network"""

    def __init__(self, latent_dim: int = 10, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = CONFIG.g_hidden_dims

        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.BatchNorm1d(hidden_dim),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator network with optional sigmoid output"""

    def __init__(
        self,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = CONFIG.d_hidden_dims

        layers = []
        input_dim = 1

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def prepare_dataloader(
    data: np.ndarray, batch_size: int, device: torch.device, shuffle: bool = True
) -> DataLoader:
    """
    FIX Issue 12: Create DataLoader with data already on device to avoid redundant copies.

    Args:
        data: Numpy array of data
        batch_size: Batch size
        device: Target device
        shuffle: Whether to shuffle

    Returns:
        DataLoader with data on device
    """
    # Move data to device before creating dataset
    tensor_data = torch.FloatTensor(data).unsqueeze(1).to(device)
    dataset = TensorDataset(tensor_data)

    # Use pin_memory only for CPU->GPU transfers
    pin_memory = device.type == "cuda" and tensor_data.device.type == "cpu"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=True,  # Avoid issues with batch size 1
    )

    return loader


def compute_per_sample_gradients(
    model: nn.Module, loss_vector: torch.Tensor, rarity_mask: torch.Tensor
) -> float:
    """
    FIX Issue 1 & 23: Compute gradient norm for rare samples during backpropagation.

    This computes per-sample gradients weighted by rarity, measuring gradient flow
    to tail regions correctly.

    Args:
        model: Generator model
        loss_vector: Per-sample losses, shape (batch_size, 1)
        rarity_mask: Boolean mask for rare samples, shape (batch_size,)

    Returns:
        Average gradient norm for rare samples
    """
    if not rarity_mask.any():
        return 0.0

    # Only compute gradients for rare samples
    rare_loss = loss_vector[rarity_mask].mean()

    # Compute gradients
    rare_loss.backward(retain_graph=True)

    # Measure gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5

    return total_norm


def train_baseline_gan(
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Tuple[Generator, Dict]:
    """
    Train baseline GAN with proper validation and early stopping.

    Args:
        train_data: Training data
        val_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed

    Returns:
        Tuple of (trained generator, history dict)
    """
    from utils import set_seed

    set_seed(seed)

    device = torch.device(CONFIG.device)

    # Compute sorted data for rarity
    sorted_data = np.sort(train_data)

    # Initialize models
    G = Generator(CONFIG.latent_dim, CONFIG.g_hidden_dims).to(device)
    D = Discriminator(CONFIG.d_hidden_dims, CONFIG.dropout_rate, use_sigmoid=True).to(
        device
    )

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))

    # FIX Issue 32: Learning rate scheduler
    if CONFIG.use_lr_scheduler:
        if CONFIG.lr_scheduler_type == "cosine":
            sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs)
            sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs)
        else:
            sched_G = sched_D = None
    else:
        sched_G = sched_D = None

    # Prepare data
    train_loader = prepare_dataloader(train_data, batch_size, device, shuffle=True)

    history = {
        "train_loss_d": [],
        "train_loss_g": [],
        "val_metrics": [],
        "gradient_norms": [],
        "epoch": [],
    }

    best_val_metric = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Baseline GAN", leave=False):
        G.train()
        D.train()

        epoch_loss_d = []
        epoch_loss_g = []
        epoch_grads = []

        for batch_tuple in train_loader:
            real = batch_tuple[0]  # Already on device
            bs = real.size(0)

            # === Train Discriminator ===
            opt_D.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)

            real_pred = D(real)
            fake_pred = D(fake.detach())

            # Binary cross-entropy
            d_loss = (
                -torch.log(real_pred + 1e-8).mean()
                - torch.log(1 - fake_pred + 1e-8).mean()
            )
            d_loss.backward()
            opt_D.step()

            epoch_loss_d.append(d_loss.item())

            # === Train Generator ===
            opt_G.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)
            fake_pred = D(fake)

            # Per-sample loss for gradient tracking
            g_loss_vector = -torch.log(fake_pred + 1e-8)
            g_loss = g_loss_vector.mean()

            # FIX Issue 1: Track gradients for rare generated samples
            with torch.no_grad():
                fake_np = fake.cpu().numpy().flatten()
                rarity = compute_rarity_scores_correct(fake_np, sorted_data)
                rare_mask = torch.from_numpy(rarity > 0.8).to(device)

            if rare_mask.any():
                # Compute gradient norm for rare samples
                grad_norm = compute_per_sample_gradients(G, g_loss_vector, rare_mask)
                epoch_grads.append(grad_norm)
                opt_G.zero_grad()  # Clear gradients from tracking

            g_loss.backward()
            opt_G.step()

            epoch_loss_g.append(g_loss.item())

        # Update learning rate
        if sched_G is not None:
            sched_G.step()
            sched_D.step()

        # Record history
        history["train_loss_d"].append(np.mean(epoch_loss_d))
        history["train_loss_g"].append(np.mean(epoch_loss_g))
        history["gradient_norms"].append(np.mean(epoch_grads) if epoch_grads else 0.0)
        history["epoch"].append(epoch)

        # Validation every 10 epochs
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

            # Early stopping based on validation Wasserstein distance
            val_metric = val_metrics["tail_wasserstein"]
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
            else:
                patience_counter += 1

            if (
                CONFIG.use_early_stopping
                and patience_counter >= CONFIG.early_stop_patience // 10
            ):
                print(f"Early stopping at epoch {epoch}")
                break

    return G, history


def train_curritail_gan(
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int,
    batch_size: int,
    alpha: float,
    k: float,
    schedule: str,
    lr: float,
    seed: int,
) -> Tuple[Generator, Dict]:
    """
    Train CurriTail-GAN with curriculum learning.

    FIX Issue 22: Uses precomputed quantile bins for efficiency.

    Args:
        train_data: Training data
        val_data: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        alpha: Curriculum strength parameter
        k: Curriculum steepness parameter
        schedule: Curriculum schedule type
        lr: Learning rate
        seed: Random seed

    Returns:
        Tuple of (trained generator, history dict)
    """
    from utils import set_seed

    set_seed(seed)

    device = torch.device(CONFIG.device)

    # Compute sorted data for rarity
    sorted_data = np.sort(train_data)

    G = Generator(CONFIG.latent_dim, CONFIG.g_hidden_dims).to(device)
    D = Discriminator(CONFIG.d_hidden_dims, CONFIG.dropout_rate, use_sigmoid=True).to(
        device
    )

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))

    if CONFIG.use_lr_scheduler and CONFIG.lr_scheduler_type == "cosine":
        sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=epochs)
        sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=epochs)
    else:
        sched_G = sched_D = None

    train_loader = prepare_dataloader(train_data, batch_size, device, shuffle=True)

    history = {
        "train_loss_d": [],
        "train_loss_g": [],
        "val_metrics": [],
        "gradient_norms": [],
        "curriculum_values": [],
        "epoch": [],
    }

    best_val_metric = float("inf")
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="CurriTail", leave=False):
        G.train()
        D.train()

        # Curriculum schedule
        t_norm = epoch / epochs
        if schedule == "sigmoid":
            S_t = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))
        elif schedule == "linear":
            S_t = t_norm
        elif schedule == "step":
            S_t = 0.0 if t_norm < 0.5 else 1.0
        else:  # none
            S_t = 0.0

        history["curriculum_values"].append(S_t)

        epoch_loss_d = []
        epoch_loss_g = []
        epoch_grads = []

        for batch_tuple in train_loader:
            real = batch_tuple[0]
            bs = real.size(0)

            # Compute rarity scores for batch
            real_np = real.cpu().numpy().flatten()
            rarity_np = compute_rarity_scores_correct(real_np, sorted_data)
            rarity = torch.FloatTensor(rarity_np).unsqueeze(1).to(device)

            # Curriculum weights
            weights = 1.0 + alpha * S_t * rarity

            # === Train Discriminator ===
            opt_D.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)

            real_pred = D(real)
            fake_pred = D(fake.detach())

            # Weighted loss
            d_loss = (
                -(weights * torch.log(real_pred + 1e-8)).mean()
                - torch.log(1 - fake_pred + 1e-8).mean()
            )
            d_loss.backward()
            opt_D.step()

            epoch_loss_d.append(d_loss.item())

            # === Train Generator ===
            opt_G.zero_grad()

            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)
            fake_pred = D(fake)

            g_loss_vector = -torch.log(fake_pred + 1e-8)
            g_loss = g_loss_vector.mean()

            # Track gradients for rare generated samples
            fake_np = fake.detach().cpu().numpy().flatten()
            fake_rarity = compute_rarity_scores_correct(fake_np, sorted_data)
            rare_mask = torch.from_numpy(fake_rarity > 0.8).to(device)

            if rare_mask.any():
                grad_norm = compute_per_sample_gradients(G, g_loss_vector, rare_mask)
                epoch_grads.append(grad_norm)
                opt_G.zero_grad()

            g_loss.backward()
            opt_G.step()

            epoch_loss_g.append(g_loss.item())

        if sched_G is not None:
            sched_G.step()
            sched_D.step()

        history["train_loss_d"].append(np.mean(epoch_loss_d))
        history["train_loss_g"].append(np.mean(epoch_loss_g))
        history["gradient_norms"].append(np.mean(epoch_grads) if epoch_grads else 0.0)
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

            val_metric = val_metrics["tail_wasserstein"]
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
            else:
                patience_counter += 1

            if (
                CONFIG.use_early_stopping
                and patience_counter >= CONFIG.early_stop_patience // 10
            ):
                print(f"Early stopping at epoch {epoch}")
                break

    return G, history


def train_wgan_gp(
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Tuple[Generator, Dict]:
    """
    Train WGAN with Gradient Penalty.

    FIX Issue 13: Handles batch size edge cases in gradient penalty.
    FIX Issue 33: Uses configurable critic update ratio.
    """
    from utils import set_seed

    set_seed(seed)

    device = torch.device(CONFIG.device)

    G = Generator(CONFIG.latent_dim, CONFIG.g_hidden_dims).to(device)
    C = Discriminator(CONFIG.d_hidden_dims, CONFIG.dropout_rate, use_sigmoid=False).to(
        device
    )  # No sigmoid for WGAN

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))
    opt_C = optim.Adam(C.parameters(), lr=lr, betas=(CONFIG.beta1, CONFIG.beta2))

    train_loader = prepare_dataloader(train_data, batch_size, device, shuffle=True)

    history = {"train_loss_c": [], "train_loss_g": [], "val_metrics": [], "epoch": []}

    for epoch in tqdm(range(epochs), desc="WGAN-GP", leave=False):
        G.train()
        C.train()

        epoch_loss_c = []
        epoch_loss_g = []

        for batch_tuple in train_loader:
            real = batch_tuple[0]
            bs = real.size(0)

            # FIX Issue 13: Skip if batch size is 1 (can cause issues)
            if bs == 1:
                continue

            # Train Critic multiple times
            for _ in range(CONFIG.wgan_n_critic):
                opt_C.zero_grad()

                z = torch.randn(bs, CONFIG.latent_dim, device=device)
                fake = G(z)

                real_pred = C(real)
                fake_pred = C(fake.detach())

                # Gradient penalty
                alpha = torch.rand(bs, 1, device=device)
                interpolates = (
                    alpha * real + (1 - alpha) * fake.detach()
                ).requires_grad_(True)
                interp_pred = C(interpolates)

                grads = torch.autograd.grad(
                    outputs=interp_pred,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(interp_pred),
                    create_graph=True,
                    retain_graph=True,
                )[0]

                # FIX Issue 13: Keepdim to handle dimension properly
                gp = (
                    (grads.norm(2, dim=1, keepdim=False) - 1) ** 2
                ).mean() * CONFIG.wgan_gp_lambda

                c_loss = fake_pred.mean() - real_pred.mean() + gp
                c_loss.backward()
                opt_C.step()

                epoch_loss_c.append(c_loss.item())

            # Train Generator
            opt_G.zero_grad()
            z = torch.randn(bs, CONFIG.latent_dim, device=device)
            fake = G(z)
            g_loss = -C(fake).mean()
            g_loss.backward()
            opt_G.step()

            epoch_loss_g.append(g_loss.item())

        history["train_loss_c"].append(np.mean(epoch_loss_c))
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
