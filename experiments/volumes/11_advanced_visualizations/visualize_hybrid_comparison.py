import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_volume import LogCumVolumeDataset
from model_ae import StandardAE
from model_conv import ConvVAE
from model_conv_complex import ConvVAEComplex

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
AE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "06_autoencoder", "models")
HYBRID_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "07_hybrid", "models")
HYBRID_COMPLEX_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "08_hybrid_complex", "models")

def load_scaler(model_dir, prefix, n_features=800):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler.n_features_in_ = n_features
    return scaler

def get_hybrid_prediction(model_type, dataset, test_idx, device):
    # AE is the same for both
    ae_model = StandardAE(input_dim=800, latent_dim=32).to(device)
    ae_model.load_state_dict(torch.load(os.path.join(AE_DIR, "ae_volume_best.pth"), map_location=device))
    ae_model.eval()
    scaler_ae = load_scaler(AE_DIR, 'scaler_ae')
    
    if model_type == 'simple':
        vae_model = ConvVAE(input_dim=800, latent_dim=32).to(device)
        vae_model.load_state_dict(torch.load(os.path.join(HYBRID_DIR, "vae_volume_hybrid_best.pth"), map_location=device))
        s_pre = 'scaler_hybrid'
        s_dir = HYBRID_DIR
    else: # complex
        vae_model = ConvVAEComplex(input_dim=800, latent_dim=32).to(device)
        vae_model.load_state_dict(torch.load(os.path.join(HYBRID_COMPLEX_DIR, "vae_volume_hybrid_complex_best.pth"), map_location=device))
        s_pre = 'scaler_hybrid_complex'
        s_dir = HYBRID_COMPLEX_DIR
    
    vae_model.eval()
    scaler_hybrid = load_scaler(s_dir, s_pre)
    
    res_orig = dataset.residuals[test_idx].astype(np.float32)
    with torch.no_grad():
        res_scaled_ae = scaler_ae.transform(res_orig).astype(np.float32)
        tensor_ae = torch.from_numpy(res_scaled_ae).to(device)
        ae_recon_scaled = ae_model(tensor_ae).cpu().numpy()
        
        hybrid_res_input = res_scaled_ae - ae_recon_scaled
        hybrid_res_scaled = scaler_hybrid.transform(hybrid_res_input).astype(np.float32)
        tensor_vae = torch.from_numpy(hybrid_res_scaled).to(device)
        vae_recon_scaled, _, _ = vae_model(tensor_vae)
        vae_recon = scaler_hybrid.inverse_transform(vae_recon_scaled.cpu().numpy())
        
        total_recon_scaled = ae_recon_scaled + vae_recon
        return scaler_ae.inverse_transform(total_recon_scaled)

def visualize_hybrid_comparison():
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
    
    sample_idx = test_idx_full[2:3] # different sample
    
    recon_simple = get_hybrid_prediction('simple', dataset, sample_idx, DEVICE)[0]
    recon_complex = get_hybrid_prediction('complex', dataset, sample_idx, DEVICE)[0]
    
    # Ground Truth
    m = dataset.m[sample_idx].reshape(-1, 1)
    b = dataset.b[sample_idx].reshape(-1, 1)
    x = np.arange(800).reshape(1, -1)
    trend = m * x + b
    gt_res = dataset.residuals[sample_idx][0]
    gt_log_cum = trend[0] + gt_res
    gt_log_vol = np.diff(gt_log_cum, prepend=0)

    # Reconstructions full
    simple_cum = trend[0] + recon_simple
    simple_vol = np.diff(simple_cum, prepend=0)
    complex_cum = trend[0] + recon_complex
    complex_vol = np.diff(complex_cum, prepend=0)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Row 1: Simple Hybrid
    axes[0, 0].plot(gt_res, color='black', alpha=0.3, label='Ground Truth')
    axes[0, 0].plot(recon_simple, color='dodgerblue', linestyle='--', label='Hybrid Simple')
    axes[0, 0].set_title('Residuals: Hybrid Simple')
    axes[0, 0].legend()
    
    axes[0, 1].plot(gt_log_vol, color='black', alpha=0.2, label='Ground Truth')
    axes[0, 1].plot(simple_vol, color='dodgerblue', label='Hybrid Simple Vol')
    axes[0, 1].set_title('Log Volume: Hybrid Simple')
    axes[0, 1].legend()

    # Row 2: Complex Hybrid
    axes[1, 0].plot(gt_res, color='black', alpha=0.3, label='Ground Truth')
    axes[1, 0].plot(recon_complex, color='forestgreen', linestyle='--', label='Hybrid Complex')
    axes[1, 0].set_title('Residuals: Hybrid Complex')
    axes[1, 0].legend()
    
    axes[1, 1].plot(gt_log_vol, color='black', alpha=0.2, label='Ground Truth')
    axes[1, 1].plot(complex_vol, color='forestgreen', label='Hybrid Complex Vol')
    axes[1, 1].set_title('Log Volume: Hybrid Complex')
    axes[1, 1].legend()

    plt.suptitle("Hybrid AE+VAE Comparison: Simple vs Complex VAE", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "volume_hybrid_complex_comparison.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Hybrid comparison plot saved to {plot_path}")

if __name__ == "__main__":
    visualize_hybrid_comparison()
