"""
Configuration file for CurriTail-GAN experiments
Contains all hyperparameters, paths, and global settings
"""
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import torch

@dataclass
class ExperimentConfig:
    """Complete experiment configuration with all hyperparameters"""
    
    # Random seeds for reproducibility
    seeds: List[int] = None  # Will be set to 30 seeds
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data configuration
    datasets: List[str] = None  # Will be ["Synthetic", "SPX", "BTC"]
    train_ratio: float = 0.8  # Train/validation split
    
    # Model architecture
    latent_dim: int = 10
    g_hidden_dims: List[int] = None  # Will be [32, 64, 128]
    d_hidden_dims: List[int] = None  # Will be [128, 64, 32]
    dropout_rate: float = 0.3
    
    # Training hyperparameters
    epochs: int = 400
    batch_size: int = 64
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # CurriTail specific
    alpha: float = 3.0  # Rarity weighting strength
    curriculum_k: float = 12.0  # Sigmoid steepness
    curriculum_schedule: str = "sigmoid"  # sigmoid, linear, step, none
    
    # WGAN-GP specific
    wgan_n_critic: int = 5
    wgan_gp_lambda: float = 10.0
    
    # Metric computation
    tail_threshold_q: float = 0.01  # 1% tail
    n_histogram_bins: int = 5  # Reduced from 20 for sparse tails
    
    # Early stopping
    early_stop_patience: int = 50
    use_early_stopping: bool = True
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"  # cosine, plateau, none
    
    # Evaluation
    n_generated_samples: int = 10000
    n_bootstrap_iters: int = 1000
    confidence_level: float = 0.95
    
    # Portfolio optimization
    portfolio_target_return: float = 0.0005  # Weekly target
    portfolio_risk_free_rate: float = 0.0001  # Weekly risk-free
    
    # Output paths
    output_dir: str = "outputs"
    models_dir: str = "saved_models"
    figures_dir: str = "figures"
    results_dir: str = "results"
    cache_dir: str = "data_cache"
    
    # Reproducibility
    use_deterministic: bool = True
    cache_data: bool = True
    cache_max_age_days: int = 7
    
    # Visualization
    figure_dpi: int = 300
    figure_format: str = "png"
    seaborn_style: str = "whitegrid"
    seaborn_context: str = "paper"
    font_scale: float = 1.4
    
    def __post_init__(self):
        """Set default values for list fields"""
        if self.seeds is None:
            self.seeds = list(range(42, 72))  # 30 seeds as required
        if self.datasets is None:
            self.datasets = ["Synthetic", "SPX", "BTC"]
        if self.g_hidden_dims is None:
            self.g_hidden_dims = [32, 64, 128]
        if self.d_hidden_dims is None:
            self.d_hidden_dims = [128, 64, 32]
        
        # Create directories
        for dir_path in [self.output_dir, self.models_dir, self.figures_dir, 
                         self.results_dir, self.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set deterministic behavior if requested
        if self.use_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def save(self, filepath: str):
        """Save configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON"""
        with open(filepath, 'r') as f:
            return cls(**json.load(f))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# Global configuration instance
CONFIG = ExperimentConfig()
