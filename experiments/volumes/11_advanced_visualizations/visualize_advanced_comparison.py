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
from model_conv_complex import ConvVAEComplex
from model_ae import StandardAE

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
VOLUME_DERIV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "03_log_cumsum_derivative_loss", "models")
VOLUME_CONV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "04_conv1d_vae", "models")
VOLUME_CONV_COMPLEX_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "05_conv1d_complex", "models")
AE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "06_autoencoder", "models")
HYBRID_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "07_hybrid", "models")

def load_scaler(model_dir, prefix):
    mean = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scale = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.n_features_in_ = 800
    return scaler

def get_predictions(model_name, dataset, test_idx, device):
    # Determine configuration
    if model_name == 'mlp_deriv':
        m_cls, m_path, s_pre = PriceVAE, os.path.join(VOLUME_DERIV_DIR, "vae_volume_derivative_best.pth"), 'scaler_volume_derivative'
        s_dir = VOLUME_DERIV_DIR
    elif model_name == 'conv_simple':
        m_cls, m_path, s_pre = ConvVAE, os.path.join(VOLUME_CONV_DIR, "vae_volume_conv_best.pth"), 'scaler_volume_conv'
        s_dir = VOLUME_CONV_DIR
    elif model_name == 'conv_complex':
        m_cls, m_path, s_pre = ConvVAEComplex, os.path.join(VOLUME_CONV_COMPLEX_DIR, "vae_volume_conv_complex_best.pth"), 'scaler_volume_conv_complex'
        s_dir = VOLUME_CONV_COMPLEX_DIR
    elif model_name == 'hybrid':
        # Hybrid requires two models
        ae_model = StandardAE(input_dim=800, latent_dim=32).to(device)
        ae_model.load_state_dict(torch.load(os.path.join(AE_DIR, "ae_volume_best.pth"), map_location=device))
        ae_model.eval()
        scaler_ae = load_scaler(AE_DIR, 'scaler_ae')
        
        vae_model = ConvVAE(input_dim=800, latent_dim=32).to(device)
        vae_model.load_state_dict(torch.load(os.path.join(HYBRID_DIR, "vae_volume_hybrid_best.pth"), map_location=device))
        vae_model.eval()
        scaler_hybrid = load_scaler(HYBRID_DIR, 'scaler_hybrid')
        
        # Prediction logic
        res_orig = dataset.residuals[test_idx].astype(np.float32)
        with torch.no_grad():
            res_scaled_ae = scaler_ae.transform(res_orig)
            tensor_ae = torch.from_numpy(res_scaled_ae).to(device)
            ae_recon_scaled = ae_model(tensor_ae).cpu().numpy()
            
            # Hybrid Residual = scaled_orig - ae_recon
            hybrid_res_input = res_scaled_ae - ae_recon_scaled
            hybrid_res_scaled = scaler_hybrid.transform(hybrid_res_input)
            tensor_vae = torch.from_numpy(hybrid_res_scaled.astype(np.float32)).to(device)
            vae_recon_scaled, _, _ = vae_model(tensor_vae)
            vae_recon = scaler_hybrid.inverse_transform(vae_recon_scaled.cpu().numpy())
            
            # Total Recon (scaled) = ae_recon + vae_recon
            total_recon_scaled = ae_recon_scaled + vae_recon
            return scaler_ae.inverse_transform(total_recon_scaled)
    else:
        return None

    # Standard model prediction
    model = m_cls(input_dim=800, latent_dim=32).to(device)
    model.load_state_dict(torch.load(m_path, map_location=device))
    model.eval()
    scaler = load_scaler(s_dir, s_pre)
    
    res_orig = dataset.residuals[test_idx].astype(np.float32)
    with torch.no_grad():
        res_scaled = scaler.transform(res_orig)
        tensor = torch.from_numpy(res_scaled).to(device)
        recon_scaled, _, _ = model(tensor)
        return scaler.inverse_transform(recon_scaled.cpu().numpy())

def compare_advanced():
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
    
    # Pick 4 random samples
    np.random.seed(42)
    test_indices = np.random.choice(len(test_idx_full), 4, replace=False)
    test_idx = test_idx_full[test_indices]
    
    # Ground Truth
    m = full_dataset.m[test_idx].reshape(-1, 1)
    b = full_dataset.b[test_idx].reshape(-1, 1)
    x = np.arange(800).reshape(1, -1)
    trends = m * x + b
    gt_log_cum = trends + full_dataset.residuals[test_idx]
    gt_log_vol = np.diff(gt_log_cum, axis=1, prepend=0)

    # Models to compare
    model_names = ['mlp_deriv', 'conv_simple', 'conv_complex', 'hybrid']
    colors = ['crimson', 'dodgerblue', 'forestgreen', 'darkorange']
    labels = ['MLP + Deriv Loss', 'Conv1D Simple', 'Conv1D Complex', 'Hybrid AE+VAE']
    
    all_recons = {}
    for m_name in model_names:
        all_recons[m_name] = get_predictions(m_name, full_dataset, test_idx, DEVICE)

    # Plot
    fig, axes = plt.subplots(4, 2, figsize=(20, 22))
    for i in range(4):
        # Left: Cumulative
        ax_cum = axes[i, 0]
        ax_cum.plot(gt_log_cum[i], 'k-', alpha=0.3, label='Ground Truth', linewidth=2)
        for m_name, color, label in zip(model_names, colors, labels):
            recon_cum = trends[i] + all_recons[m_name][i]
            ax_cum.plot(recon_cum, color=color, linestyle='--', label=label, alpha=0.8)
        ax_cum.set_title(f'Sample {i+1} - Cumulative Liquidity')
        ax_cum.legend()
        ax_cum.grid(True, alpha=0.2)

        # Right: Log Volume
        ax_vol = axes[i, 1]
        ax_vol.plot(gt_log_vol[i], 'k-', alpha=0.2, label='Ground Truth')
        for m_name, color, label in zip(model_names, colors, labels):
            recon_cum = trends[i] + all_recons[m_name][i]
            recon_vol = np.diff(recon_cum, prepend=0)
            ax_vol.plot(recon_vol, color=color, label=label, alpha=0.7)
        ax_vol.set_title(f'Sample {i+1} - Log Volume')
        ax_vol.legend()
        ax_vol.grid(True, alpha=0.2)

    plt.suptitle("Advanced Volume Reconstruction: MLP vs Conv vs Complex vs Hybrid", fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "volume_advanced_comparison.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Advanced comparison plot saved to {plot_path}")

if __name__ == "__main__":
    compare_advanced()
