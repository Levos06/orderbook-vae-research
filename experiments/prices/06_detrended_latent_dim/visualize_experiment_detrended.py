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

def visualize_experiment_results():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    LATENT_DIMS = [4, 8, 32, 64]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Prepare Data (same indices as training)
    full_dataset = DetrendedPriceDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Scaling
    train_residuals = full_dataset.residuals[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_residuals)
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=3, shuffle=False)
    res_batch, _, _ = next(iter(test_loader))
    res_batch_gpu = res_batch.to(DEVICE).float()
    
    orig_res = scaler.inverse_transform(res_batch.numpy())
    
    fig, axes = plt.subplots(len(LATENT_DIMS), 3, figsize=(18, 4 * len(LATENT_DIMS)))
    
    for i, latent_dim in enumerate(LATENT_DIMS):
        model_path = os.path.join(MODELS_DIR, f"vae_ols_dim_{latent_dim}.pth")
        if not os.path.exists(model_path):
            print(f"Skipping {latent_dim}, model not found.")
            continue
            
        model = PriceVAE(input_dim=800, latent_dim=latent_dim).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            recon_scaled, _, _ = model(res_batch_gpu)
            recon_res = scaler.inverse_transform(recon_scaled.cpu().numpy())
            
            for j in range(3):
                ax = axes[i, j]
                ax.plot(orig_res[j], color='forestgreen', alpha=0.5, label='Original')
                ax.plot(recon_res[j], color='crimson', linestyle='--', label='Recon')
                ax.set_title(f'Latent Dim {latent_dim} | Sample {j+1}')
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.set_ylabel('Residual Value')
                if i == len(LATENT_DIMS) - 1:
                    ax.set_xlabel('Orderbook Level')
                if i == 0 and j == 2:
                    ax.legend()

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "ols_dim_comparison.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    visualize_experiment_results()
