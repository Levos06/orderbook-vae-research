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

from dataset_detrended import DetrendedPriceDataset
from model_price import PriceVAE
from model_price_conv import PriceConvVAE
from model_price_conv_ultra import PriceConvVAEUltra

BASE_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
ULTRA_DIR = os.path.join(BASE_DIR, "models")
CONV_DIR = os.path.join(ROOT_DIR, "experiments", "prices", "03_price_conv", "models")
DETREND_DIR = os.path.join(ROOT_DIR, "experiments", "prices", "02_price_detrended_vae", "models")

def load_scaler(model_dir, prefix):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler.n_features_in_ = 800
    return scaler

def get_predictions(model_type, dataset, indices, device):
    res_orig = dataset.residuals[indices].astype(np.float32)
    
    if model_type == 'mlp':
        m_cls, m_path, s_pre = PriceVAE, os.path.join(DETREND_DIR, "vae_price_detrend_ols.pth"), 'scaler_detrend_ols'
        dim = 16
        s_dir = DETREND_DIR
    elif model_type == 'conv':
        m_cls, m_path, s_pre = PriceConvVAE, os.path.join(CONV_DIR, "vae_price_conv_best.pth"), 'scaler_price_conv'
        dim = 16
        s_dir = CONV_DIR
    else: # ultra
        m_cls, m_path, s_pre = PriceConvVAEUltra, os.path.join(ULTRA_DIR, "vae_price_conv_ultra_best.pth"), 'scaler_price_conv_ultra'
        dim = 32
        s_dir = ULTRA_DIR
    
    model = m_cls(input_dim=800, latent_dim=dim).to(device)
    model.load_state_dict(torch.load(m_path, map_location=device))
    model.eval()
    
    scaler = load_scaler(s_dir, s_pre)
    with torch.no_grad():
        res_scaled = scaler.transform(res_orig).astype(np.float32)
        tensor = torch.from_numpy(res_scaled).to(device)
        recon_scaled, _, _ = model(tensor)
        return scaler.inverse_transform(recon_scaled.cpu().numpy())

def compare_price_ultra():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    dataset = DetrendedPriceDataset(DATA_PATH, method='ols')
    indices = np.arange(len(dataset))
    _, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    _, test_idx_full = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Pick 2 samples for high-fidelity zoom
    test_idx = test_idx_full[10:12]
    
    mlp_res = get_predictions('mlp', dataset, test_idx, DEVICE)
    conv_res = get_predictions('conv', dataset, test_idx, DEVICE)
    ultra_res = get_predictions('ultra', dataset, test_idx, DEVICE)
    
    # Ground Truth
    m = dataset.m[test_idx].reshape(-1, 1)
    b = dataset.b[test_idx].reshape(-1, 1)
    x = np.arange(800).reshape(1, -1)
    trends = m * x + b
    gt_res = dataset.residuals[test_idx]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    for i in range(2):
        # Full Residuals
        axes[i, 0].plot(gt_res[i], 'k-', alpha=0.3, label='Ground Truth', linewidth=2)
        axes[i, 0].plot(mlp_res[i], 'r--', label='MLP (Dim 16)', alpha=0.5)
        axes[i, 0].plot(conv_res[i], 'b-', label='Conv (Dim 16)', alpha=0.6)
        axes[i, 0].plot(ultra_res[i], 'forestgreen', label='Ultra-Conv (Dim 32)', alpha=0.9, linewidth=1.5)
        axes[i, 0].set_title(f'Sample {i+1} - Full Residuals')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.2)
        
        # Zoomed View (Micro-patterns)
        zoom_range = slice(300, 500)
        axes[i, 1].plot(np.arange(300, 500), gt_res[i, zoom_range], 'k-', alpha=0.3, label='Ground Truth', linewidth=3)
        axes[i, 1].plot(np.arange(300, 500), mlp_res[i, zoom_range], 'r--', label='MLP zoom', alpha=0.5)
        axes[i, 1].plot(np.arange(300, 500), conv_res[i, zoom_range], 'b-', label='Conv zoom', alpha=0.6)
        axes[i, 1].plot(np.arange(300, 500), ultra_res[i, zoom_range], 'forestgreen', label='Ultra-Conv zoom', alpha=0.9, linewidth=2)
        axes[i, 1].set_title(f'Sample {i+1} - Zoomed Micro-Patterns (Levels 300-500)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.2)
        
    plt.suptitle("Price Fidelity: Ultra-Complex VAE vs Standard Models", fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, "price_ultra_fidelity.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Ultra comparison plot saved to {plot_path}")

if __name__ == "__main__":
    compare_price_ultra()
