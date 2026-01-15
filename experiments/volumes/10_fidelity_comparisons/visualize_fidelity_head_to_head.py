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
from model_conv import ConvVAE

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
VOLUME_DERIV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "03_log_cumsum_derivative_loss", "models")
VOLUME_CONV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "04_conv1d_vae", "models")

def load_and_predict(model_cls, model_path, scaler_prefix, model_dir, dataset, test_idx, device):
    # Load Scaler
    mean = np.load(os.path.join(model_dir, f"{scaler_prefix}_mean.npy"))
    scale = np.load(os.path.join(model_dir, f"{scaler_prefix}_scale.npy"))
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.n_features_in_ = 800

    # Load Model
    model = model_cls(input_dim=800, latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare batch
    test_res_orig = dataset.residuals[test_idx]
    test_res_scaled = scaler.transform(test_res_orig)
    
    with torch.no_grad():
        res_gpu = torch.from_numpy(test_res_scaled.astype(np.float32)).to(device)
        recon_scaled, _, _ = model(res_gpu)
        recon_res = scaler.inverse_transform(recon_scaled.cpu().numpy())
    
    return recon_res

def finalize_head_to_head():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Dataset
    full_dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    _, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx_full = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Pick 4 random samples from test set
    np.random.seed(7) # Fixed seed for reproducible comparison
    sample_indices = np.random.choice(len(test_idx_full), 4, replace=False)
    test_idx = test_idx_full[sample_indices]

    # Get Ground Truth
    gt_res = full_dataset.residuals[test_idx]
    m = full_dataset.m[test_idx].reshape(-1, 1)
    b = full_dataset.b[test_idx].reshape(-1, 1)
    x = np.arange(800).reshape(1, -1)
    trends = m * x + b
    
    gt_log_cum = trends + gt_res
    gt_log_vol = np.diff(gt_log_cum, axis=1, prepend=0)

    # Get Predictions
    deriv_res = load_and_predict(
        PriceVAE,
        os.path.join(VOLUME_DERIV_DIR, "vae_volume_derivative_best.pth"),
        'scaler_volume_derivative',
        VOLUME_DERIV_DIR,
        full_dataset,
        test_idx,
        DEVICE,
    )
    conv_res = load_and_predict(
        ConvVAE,
        os.path.join(VOLUME_CONV_DIR, "vae_volume_conv_best.pth"),
        'scaler_volume_conv',
        VOLUME_CONV_DIR,
        full_dataset,
        test_idx,
        DEVICE,
    )

    # Plot
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    
    for i in range(4):
        # Left: Cumulative Curves
        ax_cum = axes[i, 0]
        ax_cum.plot(gt_log_cum[i], color='black', alpha=0.3, label='Ground Truth', linewidth=2)
        
        # Derivative Loss Reconstruction
        deriv_cum = trends[i] + deriv_res[i]
        ax_cum.plot(deriv_cum, color='crimson', linestyle='--', label='MLP + Derivative Loss', alpha=0.8)
        
        # Conv1D Reconstruction
        conv_cum = trends[i] + conv_res[i]
        ax_cum.plot(conv_cum, color='dodgerblue', linestyle=':', label='Conv1D VAE', alpha=0.9, linewidth=2)
        
        ax_cum.set_title(f'Sample {i+1} - Log Cumulative Curve')
        ax_cum.grid(True, alpha=0.2)
        ax_cum.legend()

        # Right: Log Volume Derivation
        ax_vol = axes[i, 1]
        ax_vol.plot(gt_log_vol[i], color='black', alpha=0.2, label='Ground Truth', drawstyle='steps-mid')
        
        # Derivative Loss derived volumes
        deriv_vol = np.diff(deriv_cum, prepend=0)
        ax_vol.plot(deriv_vol, color='crimson', label='MLP + Deriv Loss', alpha=0.7, linewidth=1.2)
        
        # Conv1D derived volumes
        conv_vol = np.diff(conv_cum, prepend=0)
        ax_vol.plot(conv_vol, color='dodgerblue', label='Conv1D VAE', alpha=0.9, linewidth=1.5)
        
        ax_vol.set_title(f'Sample {i+1} - Reconstructed Log Volume')
        ax_vol.grid(True, alpha=0.2)
        ax_vol.legend()
        
        if i == 3:
            ax_cum.set_xlabel('Orderbook Level')
            ax_vol.set_xlabel('Orderbook Level')

    plt.suptitle("Volume Fidelity: MLP+Derivative Loss vs Conv1D VAE", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "volume_fidelity_comparison_detailed.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Detailed comparison plot saved to {plot_path}")

if __name__ == "__main__":
    finalize_head_to_head()
