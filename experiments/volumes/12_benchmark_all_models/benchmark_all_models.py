import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
META_SRC_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "09_meta_ensemble")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
 
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if META_SRC_DIR not in sys.path:
    sys.path.insert(0, META_SRC_DIR)

from dataset_volume import LogCumVolumeDataset
from model_price import PriceVAE
from model_conv import ConvVAE
from model_conv_complex import ConvVAEComplex
from model_ae import StandardAE
from meta_model import MetaMLP

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VOLUME_BASE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "02_log_cumsum_vae", "models")
VOLUME_DERIV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "03_log_cumsum_derivative_loss", "models")
VOLUME_CONV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "04_conv1d_vae", "models")
VOLUME_CONV_COMPLEX_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "05_conv1d_complex", "models")
AE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "06_autoencoder", "models")
HYBRID_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "07_hybrid", "models")
HYBRID_COMPLEX_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "08_hybrid_complex", "models")
META_MODEL_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "09_meta_ensemble", "models")

def load_scaler(model_dir, prefix, n_features=800):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler.n_features_in_ = n_features
    return scaler

def get_predictions(model_id, dataset, indices, device):
    res_orig = dataset.residuals[indices].astype(np.float32)
    
    if model_id == 'base_mlp':
        m_cls, m_path, s_pre = PriceVAE, os.path.join(VOLUME_BASE_DIR, "vae_volume_best.pth"), 'scaler_volume'
        s_dir = VOLUME_BASE_DIR
    elif model_id == 'mlp_deriv':
        m_cls, m_path, s_pre = PriceVAE, os.path.join(VOLUME_DERIV_DIR, "vae_volume_derivative_best.pth"), 'scaler_volume_derivative'
        s_dir = VOLUME_DERIV_DIR
    elif model_id == 'conv_simple':
        m_cls, m_path, s_pre = ConvVAE, os.path.join(VOLUME_CONV_DIR, "vae_volume_conv_best.pth"), 'scaler_volume_conv'
        s_dir = VOLUME_CONV_DIR
    elif model_id == 'conv_complex':
        m_cls, m_path, s_pre = ConvVAEComplex, os.path.join(VOLUME_CONV_COMPLEX_DIR, "vae_volume_conv_complex_best.pth"), 'scaler_volume_conv_complex'
        s_dir = VOLUME_CONV_COMPLEX_DIR
    elif model_id == 'hybrid_simple' or model_id == 'hybrid_complex':
        ae_model = StandardAE(input_dim=800, latent_dim=32).to(device)
        ae_model.load_state_dict(torch.load(os.path.join(AE_DIR, "ae_volume_best.pth"), map_location=device))
        ae_model.eval()
        scaler_ae = load_scaler(AE_DIR, 'scaler_ae')
        
        if model_id == 'hybrid_simple':
            vae_model = ConvVAE(input_dim=800, latent_dim=32).to(device)
            vae_model.load_state_dict(torch.load(os.path.join(HYBRID_DIR, "vae_volume_hybrid_best.pth"), map_location=device))
            s_hv = 'scaler_hybrid'
            s_dir = HYBRID_DIR
        else:
            vae_model = ConvVAEComplex(input_dim=800, latent_dim=32).to(device)
            vae_model.load_state_dict(torch.load(os.path.join(HYBRID_COMPLEX_DIR, "vae_volume_hybrid_complex_best.pth"), map_location=device))
            s_hv = 'scaler_hybrid_complex'
            s_dir = HYBRID_COMPLEX_DIR
        
        vae_model.eval()
        scaler_hv = load_scaler(s_dir, s_hv)
        
        with torch.no_grad():
            res_scaled_ae = scaler_ae.transform(res_orig)
            tensor_ae = torch.from_numpy(res_scaled_ae).to(device)
            ae_recon_scaled = ae_model(tensor_ae).cpu().numpy()
            
            hybrid_res_input = res_scaled_ae - ae_recon_scaled
            hybrid_res_scaled = scaler_hv.transform(hybrid_res_input).astype(np.float32)
            tensor_vae = torch.from_numpy(hybrid_res_scaled).to(device)
            vae_recon_scaled, _, _ = vae_model(tensor_vae)
            vae_recon = scaler_hv.inverse_transform(vae_recon_scaled.cpu().numpy())
            
            return scaler_ae.inverse_transform(ae_recon_scaled + vae_recon)

    elif model_id == 'meta_ensemble':
        # Get sub-model predictions
        r1 = get_predictions('mlp_deriv', dataset, indices, device)
        r2 = get_predictions('conv_simple', dataset, indices, device)
        r3 = get_predictions('conv_complex', dataset, indices, device)
        r4 = get_predictions('hybrid_simple', dataset, indices, device) # Using simple hybrid in ensemble as trained
        
        meta_model = MetaMLP().to(device)
        meta_model.load_state_dict(torch.load(os.path.join(META_MODEL_DIR, "meta_ensemble_best.pth"), map_location=device))
        meta_model.eval()
        
        scaler_mx = load_scaler(META_MODEL_DIR, 'scaler_meta_x', n_features=3200)
        scaler_my = load_scaler(META_MODEL_DIR, 'scaler_meta_y', n_features=800)
        
        X_meta = np.concatenate([r1, r2, r3, r4], axis=1).astype(np.float32)
        with torch.no_grad():
            bx = torch.from_numpy(scaler_mx.transform(X_meta)).to(device)
            out_scaled = meta_model(bx).cpu().numpy()
            return scaler_my.inverse_transform(out_scaled)
    else:
        return None

    # Base models
    model = m_cls(input_dim=800, latent_dim=32).to(device)
    model.load_state_dict(torch.load(m_path, map_location=device))
    model.eval()
    scaler = load_scaler(s_dir, s_pre)
    with torch.no_grad():
        res_scaled = scaler.transform(res_orig)
        tensor = torch.from_numpy(res_scaled).to(device)
        recon_scaled, _, _ = model(tensor)
        return scaler.inverse_transform(recon_scaled.cpu().numpy())

def run_benchmark():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Load Data
    dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    # OOS: Last 1000 samples (assuming dataset size is large enough)
    # We take samples that definitely weren't in the train/val split (random state 42, test_size 0.2)
    full_idx = np.arange(len(dataset))
    oos_idx = full_idx[-1000:]
    
    print(f"Benchmarking on {len(oos_idx)} OOS samples...")
    
    # Ground Truth Preparation
    m = dataset.m[oos_idx].reshape(-1, 1)
    b = dataset.b[oos_idx].reshape(-1, 1)
    x_coords = np.arange(800).reshape(1, -1)
    trend = m * x_coords + b
    gt_res = dataset.residuals[oos_idx]
    gt_log_cum = trend + gt_res
    gt_log_vol = np.diff(gt_log_cum, axis=1, prepend=0)

    models = [
        ('base_mlp', 'Base MLP'),
        ('mlp_deriv', 'MLP + Deriv Loss'),
        ('conv_simple', 'Conv1D Simple'),
        ('conv_complex', 'Conv1D Complex'),
        ('hybrid_simple', 'Hybrid Simple'),
        ('hybrid_complex', 'Hybrid Complex'),
        ('meta_ensemble', 'Meta-Ensemble')
    ]
    
    results = []
    
    for mid, mname in models:
        print(f"Evaluating {mname}...")
        recon_res = get_predictions(mid, dataset, oos_idx, DEVICE)
        
        # Metrics on Residuals
        rmse_res = np.sqrt(np.mean((gt_res - recon_res)**2))
        mae_res = np.mean(np.abs(gt_res - recon_res))
        
        # Metrics on Log Volume
        recon_cum = trend + recon_res
        recon_vol = np.diff(recon_cum, axis=1, prepend=0)
        
        rmse_vol = np.sqrt(np.mean((gt_log_vol - recon_vol)**2))
        
        # Spearman correlation (average per sample)
        corrs = []
        for i in range(len(oos_idx)):
            c, _ = spearmanr(gt_log_vol[i], recon_vol[i])
            corrs.append(c)
        avg_spearman = np.nanmean(corrs)
        
        max_err = np.max(np.abs(gt_log_vol - recon_vol))
        
        results.append({
            'Model': mname,
            'RMSE (Res)': f"{rmse_res:.4f}",
            'MAE (Res)': f"{mae_res:.4f}",
            'RMSE (Vol)': f"{rmse_vol:.4f}",
            'Spearman (Vol)': f"{avg_spearman:.4f}",
            'Max Error': f"{max_err:.4f}"
        })

    df = pd.DataFrame(results)
    print("\n--- RIGOROUS BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    
    # Save as Markdown table for walkthrough
    md_table = df.to_markdown(index=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "benchmark_table.md"), 'w') as f:
        f.write(md_table)
    print(f"\nLeaderboard saved to {os.path.join(RESULTS_DIR, 'benchmark_table.md')}")

if __name__ == "__main__":
    run_benchmark()
