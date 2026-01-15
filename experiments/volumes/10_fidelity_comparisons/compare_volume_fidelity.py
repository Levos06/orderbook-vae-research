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
VOLUME_BASE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "02_log_cumsum_vae", "models")
VOLUME_DERIV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "03_log_cumsum_derivative_loss", "models")
VOLUME_CONV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "04_conv1d_vae", "models")

def load_method_results(method_name, model_class, model_path, scaler_prefix, model_dir, dataset, test_idx, device):
    # Load Scaler
    mean_path = os.path.join(model_dir, f"{scaler_prefix}_mean.npy")
    scale_path = os.path.join(model_dir, f"{scaler_prefix}_scale.npy")
    if not os.path.exists(mean_path):
        # Fallback to general scaler if specific one doesn't exist
        print(f"Warning: Scaler for {method_name} not found specifically. Using fits.")
        indices = np.arange(len(dataset))
        train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
        train_res = dataset.residuals[train_idx]
        scaler = StandardScaler()
        scaler.fit(train_res)
    else:
        scaler = StandardScaler()
        scaler.mean_ = np.load(mean_path)
        scaler.scale_ = np.load(scale_path)
        scaler.n_features_in_ = 800

    # Load Model
    input_dim = 800
    latent_dim = 32
    if model_class == ConvVAE:
        model = ConvVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    else:
        model = PriceVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Error: Model {model_path} not found.")
        return None
    
    # Scale residuals for model
    test_res_orig = dataset.residuals[test_idx]
    test_res_scaled = scaler.transform(test_res_orig)
    
    with torch.no_grad():
        res_batch_gpu = torch.from_numpy(test_res_scaled[:3].astype(np.float32)).to(device)
        recon_res_scaled, _, _ = model(res_batch_gpu)
        recon_res = scaler.inverse_transform(recon_res_scaled.cpu().numpy())
    
    return recon_res

def compare_fidelity():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Prepare Base Dataset
    full_dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    _, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Get ground truth for comparison
    test_res_orig = full_dataset.residuals[test_idx[:3]]
    m_batch = full_dataset.m[test_idx[:3]].reshape(-1, 1)
    b_batch = full_dataset.b[test_idx[:3]].reshape(-1, 1)
    x = np.arange(800).reshape(1, -1)
    trends = m_batch * x + b_batch
    orig_log_cum = trends + test_res_orig
    orig_log_vol = np.diff(orig_log_cum, axis=1, prepend=0)

    # 2. Get results for each method
    methods = [
        ('Base MLP', PriceVAE, os.path.join(VOLUME_BASE_DIR, "vae_volume_best.pth"), 'scaler_volume', VOLUME_BASE_DIR),
        ('MLP + Deriv Loss', PriceVAE, os.path.join(VOLUME_DERIV_DIR, "vae_volume_derivative_best.pth"), 'scaler_volume_derivative', VOLUME_DERIV_DIR),
        ('Conv1D VAE', ConvVAE, os.path.join(VOLUME_CONV_DIR, "vae_volume_conv_best.pth"), 'scaler_volume_conv', VOLUME_CONV_DIR)
    ]
    
    recon_results = {}
    for name, m_cls, m_path, s_pre, m_dir in methods:
        res = load_method_results(name, m_cls, m_path, s_pre, m_dir, full_dataset, test_idx, DEVICE)
        if res is not None:
            recon_results[name] = res

    # 3. Plot Comparison
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    
    for i in range(3):
        # Column 1: Detrended Residuals (What VAE sees)
        ax0 = axes[i, 0]
        ax0.plot(test_res_orig[i], 'k-', alpha=0.3, label='Ground Truth')
        for name, res in recon_results.items():
            ax0.plot(res[i], label=name, alpha=0.8)
        ax0.set_title(f'Sample {i+1} - Residuals Comparison')
        ax0.legend()
        ax0.grid(True, alpha=0.2)
        
        # Column 2: Reconstructed Log Volume (The goal)
        ax1 = axes[i, 1]
        ax1.plot(orig_log_vol[i], 'k-', alpha=0.2, label='Ground Truth')
        for name, res in recon_results.items():
            recon_log_cum = trends[i] + res[i]
            recon_log_vol = np.diff(recon_log_cum, prepend=0)
            ax1.plot(recon_log_vol, label=name, alpha=0.9)
        ax1.set_title(f'Sample {i+1} - Log Volume Reconstruction')
        ax1.legend()
        ax1.grid(True, alpha=0.2)

        # Column 3: Error in Vol reconstruction (Relative/Absolute)
        ax2 = axes[i, 2]
        for name, res in recon_results.items():
            recon_log_cum = trends[i] + res[i]
            recon_log_vol = np.diff(recon_log_cum, prepend=0)
            error = np.abs(recon_log_vol - orig_log_vol[i])
            ax2.plot(error, label=f'{name} Error', alpha=0.7)
        ax2.set_title(f'Sample {i+1} - Reconstruction Error (abs)')
        ax2.legend()
        ax2.grid(True, alpha=0.2)

    plt.suptitle("Volume Fidelity Comparison: MLP vs Deriv Loss vs Conv1D", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "volume_fidelity_comparison.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    compare_fidelity()
