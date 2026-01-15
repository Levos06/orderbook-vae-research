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
from meta_model import MetaMLP

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
VOLUME_DERIV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "03_log_cumsum_derivative_loss", "models")
VOLUME_CONV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "04_conv1d_vae", "models")
VOLUME_CONV_COMPLEX_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "05_conv1d_complex", "models")
AE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "06_autoencoder", "models")
HYBRID_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "07_hybrid", "models")
META_DIR = os.path.join(BASE_DIR, "models")

def load_scaler(model_dir, prefix, n_features=800):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler.n_features_in_ = n_features
    return scaler

def get_base_predictions(model_name, dataset, test_idx, device):
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
        ae_model = StandardAE(input_dim=800, latent_dim=32).to(device)
        ae_model.load_state_dict(torch.load(os.path.join(AE_DIR, "ae_volume_best.pth"), map_location=device))
        ae_model.eval()
        scaler_ae = load_scaler(AE_DIR, 'scaler_ae')
        vae_model = ConvVAE(input_dim=800, latent_dim=32).to(device)
        vae_model.load_state_dict(torch.load(os.path.join(HYBRID_DIR, "vae_volume_hybrid_best.pth"), map_location=device))
        vae_model.eval()
        scaler_hybrid = load_scaler(HYBRID_DIR, 'scaler_hybrid')
        res_orig = dataset.residuals[test_idx].astype(np.float32)
        with torch.no_grad():
            res_scaled_ae = scaler_ae.transform(res_orig)
            tensor_ae = torch.from_numpy(res_scaled_ae).to(device)
            ae_recon_scaled = ae_model(tensor_ae).cpu().numpy()
            hybrid_res_input = res_scaled_ae - ae_recon_scaled
            hybrid_res_scaled = scaler_hybrid.transform(hybrid_res_input).astype(np.float32)
            tensor_vae = torch.from_numpy(hybrid_res_scaled).to(device)
            vae_recon_scaled, _, _ = vae_model(tensor_vae)
            vae_recon = scaler_hybrid.inverse_transform(vae_recon_scaled.cpu().numpy())
            total_recon_scaled = ae_recon_scaled + vae_recon
            return scaler_ae.inverse_transform(total_recon_scaled)
    else: return None

    # Base models
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

def visualize_meta():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    indices = np.arange(len(dataset))
    _, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx_full = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 1 sample comparison
    np.random.seed(13) # New seed for variety
    sample_idx = test_idx_full[0:1]
    
    # Get base model predictions
    r1 = get_base_predictions('mlp_deriv', dataset, sample_idx, DEVICE)
    r2 = get_base_predictions('conv_simple', dataset, sample_idx, DEVICE)
    r3 = get_base_predictions('conv_complex', dataset, sample_idx, DEVICE)
    r4 = get_base_predictions('hybrid', dataset, sample_idx, DEVICE)
    
    # Get Meta Prediction
    meta_model = MetaMLP().to(DEVICE)
    meta_model.load_state_dict(torch.load(os.path.join(META_DIR, "meta_ensemble_best.pth"), map_location=DEVICE))
    meta_model.eval()
    
    scaler_meta_x = load_scaler(META_DIR, 'scaler_meta_x', n_features=3200)
    scaler_meta_y = load_scaler(META_DIR, 'scaler_meta_y', n_features=800)
    
    ensemble_input = np.concatenate([r1, r2, r3, r4], axis=1).astype(np.float32)
    with torch.no_grad():
        bx = torch.from_numpy(scaler_meta_x.transform(ensemble_input)).to(DEVICE)
        meta_out_scaled = meta_model(bx).cpu().numpy()
        meta_recon = scaler_meta_y.inverse_transform(meta_out_scaled)[0]

    # Ground Truth Plotting data
    m = dataset.m[sample_idx].reshape(-1, 1)
    b = dataset.b[sample_idx].reshape(-1, 1)
    x_coords = np.arange(800).reshape(1, -1)
    trend = m * x_coords + b
    gt_res = dataset.residuals[sample_idx][0]
    gt_log_cum = trend[0] + gt_res
    gt_log_vol = np.diff(gt_log_cum, prepend=0)

    # Reconstructions for plotting
    meta_full_cum = trend[0] + meta_recon
    meta_log_vol = np.diff(meta_full_cum, prepend=0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Meta Residuals vs GT
    axes[0].plot(gt_res, color='black', alpha=0.3, label='Ground Truth Residual', linewidth=2)
    axes[0].plot(meta_recon, color='purple', linestyle='--', label='Meta-MLP Ensemble', alpha=0.9)
    axes[0].set_title('Meta-Ensemble: Detrended Residuals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    
    # Right: Meta Log Vol vs GT
    axes[1].plot(gt_log_vol, color='black', alpha=0.2, label='Ground Truth Log Vol')
    axes[1].plot(meta_log_vol, color='purple', label='Meta-MLP Log Vol', alpha=0.8)
    axes[1].set_title('Meta-Ensemble: Reconstructed Log Volume')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)
    
    plt.suptitle("Volume Fidelity: Meta-MLP Ensemble Performance", fontsize=20, fontweight='bold')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "volume_meta_comparison.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Meta comparison plot saved to {plot_path}")

if __name__ == "__main__":
    visualize_meta()
