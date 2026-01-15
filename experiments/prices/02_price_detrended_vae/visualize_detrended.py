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

from dataset_detrended import DetrendedPriceDataset
from model_price import PriceVAE
from utils_detrend import get_trend

def fix_visualizations(method='ols'):
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    BATCH_SIZE = 5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Load Data
    full_dataset = DetrendedPriceDataset(DATA_PATH, method=method)
    indices = np.arange(len(full_dataset))
    _, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 2. Re-apply Scalar (same as in train_detrended.py)
    # We need the exact same scaler as used during training
    train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
    train_residuals = full_dataset.residuals[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_residuals)
    
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    # 3. Load Model
    model = PriceVAE(input_dim=800, latent_dim=16).to(DEVICE)
    model_path = os.path.join(MODELS_DIR, f"vae_price_detrend_{method}.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 4. Generate Plot
    test_subset = Subset(full_dataset, test_idx)
    loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    res_batch, m_batch, b_batch = next(iter(loader))
    
    with torch.no_grad():
        res_batch = res_batch.to(DEVICE).float()
        recon_res_scaled, _, _ = model(res_batch)
        
        # SQUEEZE m and b to (B, 1) to avoid broadcasting disaster
        m_np = m_batch.numpy().reshape(-1, 1)
        b_np = b_batch.numpy().reshape(-1, 1)
        
        orig_res = scaler.inverse_transform(res_batch.cpu().numpy())
        recon_res = scaler.inverse_transform(recon_res_scaled.cpu().numpy())
        
        # Safe trend calculation: (B, 1) * (1, 800) -> (B, 800)
        x = np.arange(800).reshape(1, -1)
        trends = m_np * x + b_np # (B, 800)
        
        orig_prices = trends + orig_res # (B, 800)
        recon_prices = trends + recon_res # (B, 800)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        for i in range(3):
            # Left: Residuals
            ax_res = axes[i, 0]
            ax_res.plot(orig_res[i], color='forestgreen', alpha=0.6, label='Original Residual')
            ax_res.plot(recon_res[i], color='crimson', label='Reconstructed Residual', linestyle='--')
            ax_res.set_title(f'Sample {i+1} Residual ({method.upper()})')
            ax_res.legend(loc='upper right')
            ax_res.grid(True, alpha=0.3)
            
            # Right: Full Price
            ax_price = axes[i, 1]
            ax_price.plot(orig_prices[i], color='forestgreen', alpha=0.6, label='Original Price')
            ax_price.plot(recon_prices[i], color='crimson', label='Reconstructed Price', linestyle='--')
            ax_price.plot(trends[i], color='black', linestyle=':', alpha=0.4, label='Trend')
            
            ax_price.set_title(f'Sample {i+1} Full Price Reconstruction')
            ax_price.legend(loc='upper left')
            ax_price.grid(True, alpha=0.3)
            
            # Label
            ax_res.set_ylabel('Value')
            ax_price.set_ylabel('Price')
            
        fig.suptitle(f'Detrended VAE: {method.upper()} Method Results', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(PLOTS_DIR, exist_ok=True)
        out_path = os.path.join(PLOTS_DIR, f"detrended_recon_{method}.png")
        plt.savefig(out_path, dpi=120)
        print(f"Fixed plot saved to {out_path}")

if __name__ == "__main__":
    fix_visualizations('ols')
    fix_visualizations('endpoints')
