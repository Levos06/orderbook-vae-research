import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_volume import LogCumVolumeDataset
from model_price import PriceVAE

def visualize_expanded_volume_results():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    LATENT_DIM = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Load Data
    full_dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Scaling (Must match training)
    train_residuals = full_dataset.residuals[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_residuals)
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    # 2. Load Model
    model = PriceVAE(input_dim=800, latent_dim=LATENT_DIM).to(DEVICE)
    model_path = os.path.join(MODELS_DIR, "vae_volume_best.pth")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 3. Generate Plot
    test_subset = Subset(full_dataset, test_idx)
    loader = DataLoader(test_subset, batch_size=3, shuffle=False)
    res_batch, m_batch, b_batch = next(iter(loader))
    
    with torch.no_grad():
        res_batch_gpu = res_batch.to(DEVICE).float()
        recon_res_scaled, _, _ = model(res_batch_gpu)
        
        orig_res = scaler.inverse_transform(res_batch.numpy())
        recon_res = scaler.inverse_transform(recon_res_scaled.cpu().numpy())
        
        m_np, b_np = m_batch.numpy().reshape(-1, 1), b_batch.numpy().reshape(-1, 1)
        x = np.arange(800).reshape(1, -1)
        trends = m_np * x + b_np
        
        # Reconstruct Curves
        orig_log_cum = trends + orig_res
        recon_log_cum = trends + recon_res
        
        # Reconstruct Volumes
        orig_log_vol = np.diff(orig_log_cum, axis=1, prepend=0)
        recon_log_vol = np.diff(recon_log_cum, axis=1, prepend=0)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        for i in range(3):
            # Col 0: Detrended Residuals (What VAE sees)
            ax0 = axes[i, 0]
            ax0.plot(orig_res[i], color='forestgreen', alpha=0.5, label='Original Residual')
            ax0.plot(recon_res[i], color='crimson', linestyle='--', label='Recon Residual')
            ax0.set_title(f'Sample {i+1} - Detrended Residuals')
            ax0.grid(True, alpha=0.3)
            ax0.legend()
            
            # Col 1: Log Cumulative Curve (Full shape)
            ax1 = axes[i, 1]
            ax1.plot(orig_log_cum[i], color='forestgreen', alpha=0.5, label='Original')
            ax1.plot(recon_log_cum[i], color='crimson', linestyle='--', label='Reconstructed')
            ax1.set_title(f'Sample {i+1} - Log Cumulative Curve')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Col 2: Log Volume (Derived)
            ax2 = axes[i, 2]
            ax2.bar(np.arange(800), orig_log_vol[i], color='forestgreen', alpha=0.3, label='Original LogVol')
            ax2.step(np.arange(800), recon_log_vol[i], color='crimson', where='mid', label='Recon LogVol')
            ax2.set_title(f'Sample {i+1} - Log Volume Reconstruction')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Axis labels
            ax0.set_ylabel('Residual Value')
            ax1.set_ylabel('Log Cumulative Vol')
            ax2.set_ylabel('Log Volume')
            if i == 2:
                for j in range(3): axes[i, j].set_xlabel('Orderbook Level')

        plt.suptitle("Volume VAE Reconstruction: Residuals -> Cumulative -> Volumes", fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out_path = os.path.join(PLOTS_DIR, "volume_vae_reconstructions_expanded.png")
        base_path = os.path.join(PLOTS_DIR, "volume_vae_reconstructions.png")
        plt.savefig(out_path, dpi=120)
        # Also overwrite the original if that's what user prefers, but better to keep both for now and notify
        plt.savefig(base_path, dpi=120)
        print(f"Expanded plot saved to {out_path} and {base_path}")

if __name__ == "__main__":
    visualize_expanded_volume_results()
