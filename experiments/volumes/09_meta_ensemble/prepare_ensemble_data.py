import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

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
DATA_DIR = os.path.join(BASE_DIR, "data")
VOLUME_DERIV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "03_log_cumsum_derivative_loss", "models")
VOLUME_CONV_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "04_conv1d_vae", "models")
VOLUME_CONV_COMPLEX_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "05_conv1d_complex", "models")
AE_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "06_autoencoder", "models")
HYBRID_DIR = os.path.join(ROOT_DIR, "experiments", "volumes", "07_hybrid", "models")

def load_scaler(model_dir, prefix):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, f"{prefix}_mean.npy"))
    scaler.scale_ = np.load(os.path.join(model_dir, f"{prefix}_scale.npy"))
    scaler.n_features_in_ = 800
    return scaler

def get_model_reconstructions(model_id, dataset, device):
    print(f"Generating predictions for {model_id}...")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Setup model and scaler
    if model_id == 'mlp_deriv':
        m_cls, m_path, s_pre = PriceVAE, os.path.join(VOLUME_DERIV_DIR, "vae_volume_derivative_best.pth"), 'scaler_volume_derivative'
        s_dir = VOLUME_DERIV_DIR
    elif model_id == 'conv_simple':
        m_cls, m_path, s_pre = ConvVAE, os.path.join(VOLUME_CONV_DIR, "vae_volume_conv_best.pth"), 'scaler_volume_conv'
        s_dir = VOLUME_CONV_DIR
    elif model_id == 'conv_complex':
        m_cls, m_path, s_pre = ConvVAEComplex, os.path.join(VOLUME_CONV_COMPLEX_DIR, "vae_volume_conv_complex_best.pth"), 'scaler_volume_conv_complex'
        s_dir = VOLUME_CONV_COMPLEX_DIR
    elif model_id == 'hybrid':
        # Hybrid is special
        ae_model = StandardAE(input_dim=800, latent_dim=32).to(device)
        ae_model.load_state_dict(torch.load(os.path.join(AE_DIR, "ae_volume_best.pth"), map_location=device))
        ae_model.eval()
        scaler_ae = load_scaler(AE_DIR, 'scaler_ae')
        
        vae_model = ConvVAE(input_dim=800, latent_dim=32).to(device)
        vae_model.load_state_dict(torch.load(os.path.join(HYBRID_DIR, "vae_volume_hybrid_best.pth"), map_location=device))
        vae_model.eval()
        scaler_hybrid = load_scaler(HYBRID_DIR, 'scaler_hybrid')
        
        all_recons = []
        with torch.no_grad():
            for batch_res, _, _ in loader:
                batch_res_np = batch_res.numpy()
                res_scaled_ae = scaler_ae.transform(batch_res_np).astype(np.float32)
                tensor_ae = torch.from_numpy(res_scaled_ae).to(device)
                ae_recon_scaled = ae_model(tensor_ae).cpu().numpy()
                
                hybrid_res_input = res_scaled_ae - ae_recon_scaled
                hybrid_res_scaled = scaler_hybrid.transform(hybrid_res_input).astype(np.float32)
                tensor_vae = torch.from_numpy(hybrid_res_scaled).to(device)
                vae_recon_scaled, _, _ = vae_model(tensor_vae)
                vae_recon = scaler_hybrid.inverse_transform(vae_recon_scaled.cpu().numpy())
                
                total_recon_scaled = ae_recon_scaled + vae_recon
                total_recon = scaler_ae.inverse_transform(total_recon_scaled)
                all_recons.append(total_recon)
        return np.concatenate(all_recons, axis=0)

    # Base case standard models
    model = m_cls(input_dim=800, latent_dim=32).to(device)
    model.load_state_dict(torch.load(m_path, map_location=device))
    model.eval()
    scaler = load_scaler(s_dir, s_pre)
    
    all_recons = []
    with torch.no_grad():
        for batch_res, _, _ in loader:
            batch_res_np = batch_res.numpy()
            res_scaled = scaler.transform(batch_res_np).astype(np.float32)
            tensor = torch.from_numpy(res_scaled).to(device)
            recon_scaled, _, _ = model(tensor)
            recon = scaler.inverse_transform(recon_scaled.cpu().numpy())
            all_recons.append(recon)
    return np.concatenate(all_recons, axis=0)

def prepare_data():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    
    # 1. Collect all predictions
    r1 = get_model_reconstructions('mlp_deriv', dataset, DEVICE)
    r2 = get_model_reconstructions('conv_simple', dataset, DEVICE)
    r3 = get_model_reconstructions('conv_complex', dataset, DEVICE)
    r4 = get_model_reconstructions('hybrid', dataset, DEVICE)
    
    # 2. Stack inputs: (N, 4 * 800)
    ensemble_inputs = np.concatenate([r1, r2, r3, r4], axis=1)
    ensemble_targets = dataset.residuals
    
    # 3. Save
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "ensemble_inputs.npy"), ensemble_inputs)
    np.save(os.path.join(DATA_DIR, "ensemble_targets.npy"), ensemble_targets)
    print(f"Ensemble data saved to {DATA_DIR}")

if __name__ == "__main__":
    prepare_data()
