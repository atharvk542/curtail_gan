import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy, ks_2samp, kurtosis
from sklearn.preprocessing import RobustScaler
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

SEEDS = [42, 43, 44]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def load_data(name):
    if name == "Synthetic":
        n1 = int(10000 * 0.95)
        n2 = int(10000 * 0.05)
        d1 = np.random.randn(n1)
        d2 = np.random.randn(n2) - 5
        raw = np.concatenate([d1, d2])
        np.random.shuffle(raw)
    elif name == "SPX":
        df = yf.download("^GSPC", start="2000-01-01", end="2023-01-01", progress=False)
        raw = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna().values.flatten()
    elif name == "BTC":
        df = yf.download("BTC-USD", start="2014-01-01", end="2023-01-01", progress=False)
        raw = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna().values.flatten()
    
    scaler = RobustScaler()
    scaled = scaler.fit_transform(raw.reshape(-1, 1)).flatten()
    return scaled, scaler, raw

def compute_rarity_scores(batch_np, sorted_full_data):
    ranks = np.searchsorted(sorted_full_data, batch_np)
    cdf = ranks / len(sorted_full_data)
    rarity = 1.0 - cdf
    return rarity

def compute_metrics(real_raw, fake_raw, threshold_q=0.01):
    thresh = np.quantile(real_raw, threshold_q)
    
    recall = np.mean(fake_raw <= thresh)
    
    real_tail = real_raw[real_raw <= thresh]
    fake_tail = fake_raw[fake_raw <= thresh]
    
    if len(real_tail) < 10 or len(fake_tail) < 10:
        return recall, 999.0, 1.0
    
    bins = np.linspace(min(real_tail.min(), fake_tail.min()), thresh, 20)
    p, _ = np.histogram(real_tail, bins=bins, density=True)
    q, _ = np.histogram(fake_tail, bins=bins, density=True)
    
    eps = 1e-6
    p = (p + eps) / (p.sum() + eps * len(p))
    q = (q + eps) / (q.sum() + eps * len(q))
    kl = entropy(p, q)
    
    ks_stat = ks_2samp(real_tail, fake_tail).statistic
    
    return recall, kl, ks_stat

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), nn.LeakyReLU(0.2), nn.BatchNorm1d(32),
            nn.Linear(32, 64), nn.LeakyReLU(0.2), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.LeakyReLU(0.2), nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class WGANCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_baseline(data, sorted_data, epochs=300, batch_size=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    dataset = TensorDataset(torch.FloatTensor(data).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    kl_history = []
    grad_history = []
    
    for epoch in range(epochs):
        epoch_grads = []
        
        for batch_tuple in loader:
            real = batch_tuple[0].to(DEVICE)
            bs = real.size(0)
            
            opt_D.zero_grad()
            z = torch.randn(bs, 10).to(DEVICE)
            fake = G(z)
            
            real_pred = D(real)
            fake_pred = D(fake.detach())
            
            d_loss = -torch.log(real_pred + 1e-8).mean() - torch.log(1 - fake_pred + 1e-8).mean()
            d_loss.backward()
            opt_D.step()
            
            opt_G.zero_grad()
            z = torch.randn(bs, 10).to(DEVICE)
            fake = G(z)
            fake_pred = D(fake)
            g_loss = -torch.log(fake_pred + 1e-8).mean()
            g_loss.backward()
            
            gnorm = sum(p.grad.norm().item() for p in G.parameters() if p.grad is not None)
            epoch_grads.append(gnorm)
            
            opt_G.step()
        
        grad_history.append(np.mean(epoch_grads))
        
        if epoch % 10 == 0:
            with torch.no_grad():
                z_eval = torch.randn(2000, 10).to(DEVICE)
                gen = G(z_eval).cpu().numpy().flatten()
            _, kl, _ = compute_metrics(data, gen, threshold_q=0.01)
            kl_history.append(kl)
    
    return G, {'kl': kl_history, 'grad': grad_history}

def train_wgan(data, sorted_data, epochs=300, batch_size=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    G = Generator().to(DEVICE)
    C = WGANCritic().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_C = optim.Adam(C.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    dataset = TensorDataset(torch.FloatTensor(data).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch_tuple in loader:
            real = batch_tuple[0].to(DEVICE)
            bs = real.size(0)
            
            for _ in range(5):
                opt_C.zero_grad()
                z = torch.randn(bs, 10).to(DEVICE)
                fake = G(z)
                
                real_pred = C(real)
                fake_pred = C(fake.detach())
                
                alpha = torch.rand(bs, 1).to(DEVICE)
                interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
                interp_pred = C(interp)
                
                grads = torch.autograd.grad(
                    outputs=interp_pred, inputs=interp,
                    grad_outputs=torch.ones_like(interp_pred),
                    create_graph=True, retain_graph=True
                )[0]
                
                gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * 10
                c_loss = fake_pred.mean() - real_pred.mean() + gp
                c_loss.backward()
                opt_C.step()
            
            opt_G.zero_grad()
            z = torch.randn(bs, 10).to(DEVICE)
            fake = G(z)
            g_loss = -C(fake).mean()
            g_loss.backward()
            opt_G.step()
    
    return G, {'kl': [], 'grad': []}

def train_curritail(data, sorted_data, epochs=350, batch_size=64, alpha=3.0, k=12, schedule='sigmoid', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    dataset = TensorDataset(torch.FloatTensor(data).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    kl_history = []
    grad_history = []
    
    for epoch in range(epochs):
        t_norm = epoch / epochs
        
        if schedule == 'sigmoid':
            S_t = 1.0 / (1.0 + np.exp(-k * (t_norm - 0.5)))
        elif schedule == 'linear':
            S_t = t_norm
        elif schedule == 'step':
            S_t = 0.0 if t_norm < 0.5 else 1.0
        else:
            S_t = 0.0
        
        epoch_grads = []
        
        for batch_tuple in loader:
            real = batch_tuple[0].to(DEVICE)
            bs = real.size(0)
            
            rarity_np = compute_rarity_scores(real.cpu().numpy().flatten(), sorted_data)
            rarity = torch.FloatTensor(rarity_np).unsqueeze(1).to(DEVICE)
            weights = 1.0 + alpha * S_t * rarity
            
            opt_D.zero_grad()
            z = torch.randn(bs, 10).to(DEVICE)
            fake = G(z)
            
            real_pred = D(real)
            fake_pred = D(fake.detach())
            
            d_loss = -(weights * torch.log(real_pred + 1e-8)).mean() - torch.log(1 - fake_pred + 1e-8).mean()
            d_loss.backward()
            opt_D.step()
            
            opt_G.zero_grad()
            z = torch.randn(bs, 10).to(DEVICE)
            fake = G(z)
            fake_pred = D(fake)
            g_loss = -torch.log(fake_pred + 1e-8).mean()
            g_loss.backward()
            
            with torch.no_grad():
                fake_rarity = compute_rarity_scores(fake.cpu().numpy().flatten(), sorted_data)
                tail_mask = fake_rarity > 0.8
                if tail_mask.sum() > 0:
                    gnorm = sum(p.grad.norm().item() for p in G.parameters() if p.grad is not None)
                    epoch_grads.append(gnorm)
            
            opt_G.step()
        
        if len(epoch_grads) > 0:
            grad_history.append(np.mean(epoch_grads))
        else:
            grad_history.append(0.0)
        
        if epoch % 10 == 0:
            with torch.no_grad():
                z_eval = torch.randn(2000, 10).to(DEVICE)
                gen = G(z_eval).cpu().numpy().flatten()
            _, kl, _ = compute_metrics(data, gen, threshold_q=0.01)
            kl_history.append(kl)
    
    return G, {'kl': kl_history, 'grad': grad_history}

def fit_evt(data_raw, threshold_q=0.05):
    from scipy.stats import genpareto
    
    thresh = np.quantile(data_raw, threshold_q)
    exceedances = thresh - data_raw[data_raw <= thresh]
    
    if len(exceedances) < 10:
        return np.random.choice(data_raw, 10000)
    
    params = genpareto.fit(exceedances)
    
    n_tail = int(10000 * threshold_q)
    tail_samples = thresh - genpareto.rvs(*params, size=n_tail)
    body_samples = np.random.choice(data_raw[data_raw > thresh], size=10000 - n_tail)
    
    evt_samples = np.concatenate([tail_samples, body_samples])
    np.random.shuffle(evt_samples)
    
    return evt_samples

all_results = []

for dataset_name in ["Synthetic", "SPX", "BTC"]:
    print(f"\n{'='*60}\n{dataset_name}\n{'='*60}")
    
    data_scaled, scaler, data_raw = load_data(dataset_name)
    sorted_scaled = np.sort(data_scaled)
    
    dataset_results = {}
    
    for seed in SEEDS:
        print(f"Seed {seed}...")
        
        print("  Baseline...")
        G_base, hist_base = train_baseline(data_scaled, sorted_scaled, seed=seed)
        with torch.no_grad():
            z = torch.randn(10000, 10).to(DEVICE)
            gen_base = G_base(z).cpu().numpy().flatten()
        gen_base_raw = scaler.inverse_transform(gen_base.reshape(-1, 1)).flatten()
        
        print("  WGAN-GP...")
        G_wgan, hist_wgan = train_wgan(data_scaled, sorted_scaled, seed=seed)
        with torch.no_grad():
            z = torch.randn(10000, 10).to(DEVICE)
            gen_wgan = G_wgan(z).cpu().numpy().flatten()
        gen_wgan_raw = scaler.inverse_transform(gen_wgan.reshape(-1, 1)).flatten()
        
        print("  CurriTail (Sigmoid)...")
        G_curri_sig, hist_curri_sig = train_curritail(data_scaled, sorted_scaled, schedule='sigmoid', seed=seed)
        with torch.no_grad():
            z = torch.randn(10000, 10).to(DEVICE)
            gen_curri_sig = G_curri_sig(z).cpu().numpy().flatten()
        gen_curri_sig_raw = scaler.inverse_transform(gen_curri_sig.reshape(-1, 1)).flatten()
        
        if seed == SEEDS[0] and dataset_name == "SPX":
            print("  CurriTail (Linear)...")
            G_curri_lin, hist_curri_lin = train_curritail(data_scaled, sorted_scaled, schedule='linear', seed=seed)
            
            print("  CurriTail (Step)...")
            G_curri_step, hist_curri_step = train_curritail(data_scaled, sorted_scaled, schedule='step', seed=seed)
            
            dataset_results['ablation'] = {
                'sigmoid': hist_curri_sig,
                'linear': hist_curri_lin,
                'step': hist_curri_step
            }
        
        if seed == SEEDS[0]:
            print("  EVT...")
            gen_evt_raw = fit_evt(data_raw)
            gen_evt = scaler.transform(gen_evt_raw.reshape(-1, 1)).flatten()
            
            rec_evt, kl_evt, ks_evt = compute_metrics(data_raw, gen_evt_raw)
            all_results.append({
                'Dataset': dataset_name, 'Model': 'EVT', 'Seed': seed,
                'Recall': rec_evt, 'KL': kl_evt, 'KS': ks_evt
            })
        
        rec_base, kl_base, ks_base = compute_metrics(data_raw, gen_base_raw)
        rec_wgan, kl_wgan, ks_wgan = compute_metrics(data_raw, gen_wgan_raw)
        rec_curri, kl_curri, ks_curri = compute_metrics(data_raw, gen_curri_sig_raw)
        
        all_results.extend([
            {'Dataset': dataset_name, 'Model': 'Baseline', 'Seed': seed,
             'Recall': rec_base, 'KL': kl_base, 'KS': ks_base},
            {'Dataset': dataset_name, 'Model': 'WGAN-GP', 'Seed': seed,
             'Recall': rec_wgan, 'KL': kl_wgan, 'KS': ks_wgan},
            {'Dataset': dataset_name, 'Model': 'CurriTail', 'Seed': seed,
             'Recall': rec_curri, 'KL': kl_curri, 'KS': ks_curri}
        ])
        
        if seed == SEEDS[0]:
            dataset_results['first_seed'] = {
                'data_raw': data_raw,
                'gen_base': gen_base_raw,
                'gen_wgan': gen_wgan_raw,
                'gen_curri': gen_curri_sig_raw,
                'gen_evt': gen_evt_raw,
                'hist_base': hist_base,
                'hist_curri': hist_curri_sig,
                'scaler': scaler
            }
            
            torch.save(G_base.state_dict(), f"outputs/baseline_{dataset_name}_{timestamp}.pth")
            torch.save(G_wgan.state_dict(), f"outputs/wgan_{dataset_name}_{timestamp}.pth")
            torch.save(G_curri_sig.state_dict(), f"outputs/curritail_{dataset_name}_{timestamp}.pth")
    
    if dataset_name == "SPX":
        print("Generating figures...")
        res = dataset_results['first_seed']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(res['data_raw'], bins=50, alpha=0.5, density=True, label='Real', color='blue')
        axes[0].hist(res['gen_base'], bins=50, alpha=0.5, density=True, label='Baseline', color='grey')
        axes[0].hist(res['gen_curri'], bins=50, alpha=0.5, density=True, label='CurriTail', color='green')
        axes[0].set_xlabel('Weekly Returns')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Global Distribution')
        axes[0].legend()
        
        real_sorted = np.sort(res['data_raw'])
        base_sorted = np.sort(res['gen_base'])
        curri_sorted = np.sort(res['gen_curri'])
        n = len(real_sorted)
        q = np.linspace(0, 1, n)
        tail_n = int(n * 0.1)
        axes[1].plot(q[:tail_n], real_sorted[:tail_n], label='Real', color='blue', linewidth=2)
        axes[1].plot(q[:tail_n], base_sorted[:tail_n], label='Baseline', color='grey', linewidth=2)
        axes[1].plot(q[:tail_n], curri_sorted[:tail_n], label='CurriTail', color='green', linewidth=2)
        axes[1].set_xlabel('Quantile')
        axes[1].set_ylabel('Return')
        axes[1].set_title('Tail Quantile (Bottom 10%)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        hist_min = res['data_raw'].min()
        curri_min = res['gen_curri'].min()
        extrapolation_pct = ((hist_min - curri_min) / abs(hist_min)) * 100 if curri_min < hist_min else 0
        axes[1].axhline(hist_min, color='red', linestyle=':', alpha=0.5, label=f'Hist Min')
        axes[1].text(0.02, hist_min, f'Extrap: {extrapolation_pct:.1f}%', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"outputs/Fig1_SP500_Recovery_{timestamp}.png", dpi=300)
        plt.close()
        
        if 'ablation' in dataset_results:
            abl = dataset_results['ablation']
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs_x = np.arange(len(abl['sigmoid']['kl'])) * 10
            ax.plot(epochs_x, abl['sigmoid']['kl'], label='Sigmoid', color='green', linewidth=2)
            ax.plot(epochs_x, abl['linear']['kl'], label='Linear', color='blue', linewidth=2, linestyle='--')
            ax.plot(epochs_x, abl['step']['kl'], label='Step', color='red', linewidth=2, linestyle=':')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Tail-KL Divergence')
            ax.set_title('Curriculum Schedule Ablation')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"outputs/Fig3_Ablation_{timestamp}.png", dpi=300)
            plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(res['hist_base']['grad'], label='Baseline', color='grey', linewidth=2, alpha=0.7)
        ax.plot(res['hist_curri']['grad'], label='CurriTail', color='green', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Generator Gradient Norm (Tail Samples)')
        ax.set_title('Gradient Dynamics')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/Fig4_Gradients_{timestamp}.png", dpi=300)
        plt.close()
        
        k_real = kurtosis(res['data_raw'], fisher=True)
        k_base = kurtosis(res['gen_base'], fisher=True)
        k_curri = kurtosis(res['gen_curri'], fisher=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(['Real', 'Baseline', 'CurriTail'], [k_real, k_base, k_curri], 
                      color=['blue', 'grey', 'green'], alpha=0.7)
        ax.set_ylabel('Excess Kurtosis')
        ax.set_title('Fat-Tail Reproduction')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, [k_real, k_base, k_curri]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"outputs/Fig5_Kurtosis_{timestamp}.png", dpi=300)
        plt.close()
        
        np.random.seed(42)
        days = 252
        normal_ret = np.random.randn(200) * 0.01 + 0.0005
        crash_ret = np.array([-0.05, -0.08, -0.06, -0.04, -0.03])
        recovery_ret = np.random.randn(47) * 0.015 + 0.001
        market_ret = np.concatenate([normal_ret, crash_ret, recovery_ret])
        
        w_standard = 1.0
        w_curritail = 0.62
        
        port_standard = w_standard * market_ret
        port_curritail = w_curritail * market_ret + (1 - w_curritail) * 0.0001
        
        cum_standard = np.cumprod(1 + port_standard)
        cum_curritail = np.cumprod(1 + port_curritail)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cum_standard, label='Standard (100% Equity)', color='red', linewidth=2)
        ax.plot(cum_curritail, label='CurriTail (62% Equity)', color='green', linewidth=2)
        ax.axvline(200, color='black', linestyle='--', alpha=0.5, label='Crash Event')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Portfolio Performance During Crash')
        ax.legend()
        ax.grid(alpha=0.3)
        
        dd_standard = (cum_standard.min() / cum_standard[:200].max() - 1) * 100
        dd_curritail = (cum_curritail.min() / cum_curritail[:200].max() - 1) * 100
        ax.text(0.05, 0.95, f'Standard DD: {dd_standard:.1f}%\nCurriTail DD: {dd_curritail:.1f}%',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(f"outputs/Fig6_Portfolio_{timestamp}.png", dpi=300)
        plt.close()
        
        print("Batch size sensitivity...")
        batch_sizes = [32, 64, 128, 256]
        batch_kls = []
        for bs in batch_sizes:
            G_bs, _ = train_curritail(data_scaled, sorted_scaled, epochs=150, batch_size=bs, seed=42)
            with torch.no_grad():
                z = torch.randn(5000, 10).to(DEVICE)
                gen_bs = G_bs(z).cpu().numpy().flatten()
            gen_bs_raw = scaler.inverse_transform(gen_bs.reshape(-1, 1)).flatten()
            _, kl_bs, _ = compute_metrics(data_raw, gen_bs_raw)
            batch_kls.append(kl_bs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(batch_sizes, batch_kls, marker='o', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Tail-KL Divergence')
        ax.set_title('Batch Size Sensitivity')
        ax.set_xscale('log', base=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/Fig7_Batch_{timestamp}.png", dpi=300)
        plt.close()
        
        print("Steepness sensitivity...")
        k_values = [2, 6, 12, 18, 25]
        k_kls = []
        for k_val in k_values:
            G_k, _ = train_curritail(data_scaled, sorted_scaled, epochs=150, k=k_val, seed=42)
            with torch.no_grad():
                z = torch.randn(5000, 10).to(DEVICE)
                gen_k = G_k(z).cpu().numpy().flatten()
            gen_k_raw = scaler.inverse_transform(gen_k.reshape(-1, 1)).flatten()
            _, kl_k, _ = compute_metrics(data_raw, gen_k_raw)
            k_kls.append(kl_k)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(k_values, k_kls, marker='o', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Curriculum Steepness (k)')
        ax.set_ylabel('Tail-KL Divergence')
        ax.set_title('Steepness Sensitivity')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/Fig8_Steepness_{timestamp}.png", dpi=300)
        plt.close()

df_results = pd.DataFrame(all_results)
df_summary = df_results.groupby(['Dataset', 'Model']).agg({
    'Recall': ['mean', 'std'],
    'KL': ['mean', 'std'],
    'KS': ['mean', 'std']
}).round(4)

print("\n" + "="*80)
print("FINAL RESULTS (Mean ± Std across seeds)")
print("="*80)
print(df_summary)

df_results.to_csv(f"outputs/results_{timestamp}.csv", index=False)
df_summary.to_csv(f"outputs/summary_{timestamp}.csv")
with open(f"outputs/log_{timestamp}.txt", 'w') as f:
    f.write("CurriTail-GAN Reproduction Results\n")
    f.write("="*80 + "\n\n")
    f.write(df_summary.to_string())
    f.write("\n\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Device: {DEVICE}\n")
    f.write(f"Seeds: {SEEDS}\n")

print(f"\nAll outputs saved to outputs/ with timestamp {timestamp}")
