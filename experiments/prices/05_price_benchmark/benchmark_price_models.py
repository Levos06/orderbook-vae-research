import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRICE_MLP_DIR = os.path.join(ROOT_DIR, "experiments", "prices", "01_price_mlp", "models")
DETREND_DIR = os.path.join(ROOT_DIR, "experiments", "prices", "02_price_detrended_vae", "models")
CONV_DIR = os.path.join(ROOT_DIR, "experiments", "prices", "03_price_conv", "models")
ULTRA_DIR = os.path.join(ROOT_DIR, "experiments", "prices", "04_price_conv_ultra", "models")

def load_scaler(model_dir, prefix, n_features=800):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler.n_features_in_ = n_features
    return scaler

def get_predictions(model_id, data_raw, device):
    # data_raw: (N, 800) original price lines
    
    if model_id == 'base_mlp':
        # This one was trained on raw normalized prices
        m_path, s_pre = os.path.join(PRICE_MLP_DIR, "vae_price_best.pth"), 'scaler_price'
        model = PriceVAE(input_dim=800, latent_dim=16).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device))
        model.eval()
        scaler = load_scaler(PRICE_MLP_DIR, s_pre)
        with torch.no_grad():
            x_scaled = scaler.transform(data_raw).astype(np.float32)
            recon_scaled, _, _ = model(torch.from_numpy(x_scaled).to(device))
            return scaler.inverse_transform(recon_scaled.cpu().numpy())

    # The rest are detrended OLS models
    x_coords = np.arange(800).reshape(1, -1)
    # Calculate OLS for the batch
    X_mat = np.vander(np.arange(800), 2) # (800, 2) [x, 1]
    # Solve (X^T X)^-1 X^T y
    # y is (N, 800) -> weights (N, 2)
    # weights = pinv(X) @ y^T -> (2, 800) @ (800, N) = (2, N)
    weights = np.linalg.pinv(X_mat) @ data_raw.T
    m = weights[0].reshape(-1, 1)
    b = weights[1].reshape(-1, 1)
    trends = m * x_coords + b
    residuals_raw = data_raw - trends

    if model_id == 'mlp_ols':
        m_cls, m_path, s_pre, d = PriceVAE, os.path.join(DETREND_DIR, "vae_price_detrend_ols.pth"), 'scaler_detrend_ols', 16
        s_dir = DETREND_DIR
    elif model_id == 'conv':
        m_cls, m_path, s_pre, d = PriceConvVAE, os.path.join(CONV_DIR, "vae_price_conv_best.pth"), 'scaler_price_conv', 16
        s_dir = CONV_DIR
    elif model_id == 'ultra':
        m_cls, m_path, s_pre, d = PriceConvVAEUltra, os.path.join(ULTRA_DIR, "vae_price_conv_ultra_best.pth"), 'scaler_price_conv_ultra', 32
        s_dir = ULTRA_DIR
        
    model = m_cls(input_dim=800, latent_dim=d).to(device)
    model.load_state_dict(torch.load(m_path, map_location=device))
    model.eval()
    scaler = load_scaler(s_dir, s_pre)
    
    with torch.no_grad():
        res_scaled = scaler.transform(residuals_raw).astype(np.float32)
        recon_scaled, _, _ = model(torch.from_numpy(res_scaled).to(device))
        recon_res = scaler.inverse_transform(recon_scaled.cpu().numpy())
        return trends + recon_res

def run_benchmark():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Load and Extract Price Line
    raw_data = np.load(DATA_PATH).astype(np.float32)
    bids_prices = raw_data[:, 0, :, 0]
    bids_inverted = bids_prices[:, ::-1]
    asks_prices = raw_data[:, 1, :, 0]
    data_raw = np.concatenate([bids_inverted, asks_prices], axis=1) # (N, 800)
    
    # Isolate OOS: Last 1000 samples
    oos_raw = data_raw[-1000:]
    
    print(f"Benchmarking Price Models on {len(oos_raw)} OOS samples...")
    
    models = [
        ('base_mlp', 'Base MLP'),
        ('mlp_ols', 'MLP + OLS Detrend'),
        ('conv', 'Conv1D (4L)'),
        ('ultra', 'Ultra-Conv (5L)')
    ]
    
    results = []
    
    for mid, mname in models:
        print(f"Evaluating {mname}...")
        recon_full = get_predictions(mid, oos_raw, DEVICE)
        
        # Metrics on full price line
        rmse = np.sqrt(np.mean((oos_raw - recon_full)**2))
        mae = np.mean(np.abs(oos_raw - recon_full))
        max_err = np.max(np.abs(oos_raw - recon_full))
        
        # R2 score (average across samples)
        r2_vals = [r2_score(oos_raw[i], recon_full[i]) for i in range(len(oos_raw))]
        avg_r2 = np.mean(r2_vals)
        
        results.append({
            'Model': mname,
            'RMSE': f"{rmse:.6f}",
            'MAE': f"{mae:.6f}",
            'R²': f"{avg_r2:.6f}",
            'Max Error': f"{max_err:.6f}"
        })

    df = pd.DataFrame(results)
    print("\n--- RIGOROUS PRICE BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_markdown(os.path.join(RESULTS_DIR, "price_benchmark_table.md"), index=False)
    print(f"\nLeaderboard saved to {os.path.join(RESULTS_DIR, 'price_benchmark_table.md')}")

if __name__ == "__main__":
    run_benchmark()
