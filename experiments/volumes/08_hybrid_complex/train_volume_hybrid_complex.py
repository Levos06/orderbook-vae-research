import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_volume import LogCumVolumeDataset
from model_ae import StandardAE
from model_conv_complex import ConvVAEComplex
from model_price import loss_function

def train_volume_hybrid_complex():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    AE_MODELS_DIR = os.path.join(
        ROOT_DIR,
        "experiments",
        "volumes",
        "06_autoencoder",
        "models",
    )
    BATCH_SIZE = 64 # Smaller batch for complex model
    LATENT_DIM_AE = 32
    LATENT_DIM_VAE = 32
    LEARNING_RATE = 5e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training Hybrid COMPLEX VAE on residues of AE on {DEVICE}")
    
    # 1. Load Data
    full_dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, _ = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 2. Load AE and its Scaler
    ae_model = StandardAE(input_dim=800, latent_dim=LATENT_DIM_AE).to(DEVICE)
    ae_model.load_state_dict(torch.load(os.path.join(AE_MODELS_DIR, "ae_volume_best.pth"), map_location=DEVICE))
    ae_model.eval()
    
    scaler_ae = StandardScaler()
    scaler_ae.mean_ = np.load(os.path.join(AE_MODELS_DIR, "scaler_ae_mean.npy"))
    scaler_ae.scale_ = np.load(os.path.join(AE_MODELS_DIR, "scaler_ae_scale.npy"))
    scaler_ae.n_features_in_ = 800
    
    # 3. Precompute Hybrid residuals
    with torch.no_grad():
        all_res_scaled = scaler_ae.transform(full_dataset.residuals).astype(np.float32)
        all_tensor = torch.from_numpy(all_res_scaled).to(DEVICE)
        
        ae_recon_list = []
        for i in range(0, len(all_tensor), 1000):
            batch = all_tensor[i:i+1000]
            ae_recon_list.append(ae_model(batch).cpu().numpy())
        ae_recon = np.concatenate(ae_recon_list, axis=0)
        hybrid_residuals = all_res_scaled - ae_recon
    
    # 4. Scale Hybrid Residuals
    scaler_hybrid = StandardScaler()
    scaler_hybrid.fit(hybrid_residuals[train_idx])
    hybrid_residuals_scaled = scaler_hybrid.transform(hybrid_residuals)
    
    class HybridDataset(torch.utils.data.Dataset):
        def __init__(self, residuals): self.residuals = residuals
        def __len__(self): return len(self.residuals)
        def __getitem__(self, idx): return torch.from_numpy(self.residuals[idx]).float()

    train_loader = DataLoader(HybridDataset(hybrid_residuals_scaled[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HybridDataset(hybrid_residuals_scaled[val_idx]), batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Model: Complex VAE
    model = ConvVAEComplex(input_dim=800, latent_dim=LATENT_DIM_VAE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_res in train_loader:
            batch_res = batch_res.to(DEVICE).float()
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch_res)
            loss, _, _ = loss_function(recon_batch, batch_res, mu, log_var)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_res in val_loader:
                batch_res = batch_res.to(DEVICE).float()
                recon_batch, mu, log_var = model(batch_res)
                loss, _, _ = loss_function(recon_batch, batch_res, mu, log_var)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/100] Hybrid Complex VAE Val Loss: {avg_val_loss:.6f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "vae_volume_hybrid_complex_best.pth"))
            np.save(os.path.join(MODELS_DIR, "scaler_hybrid_complex_mean.npy"), scaler_hybrid.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_hybrid_complex_scale.npy"), scaler_hybrid.scale_)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break
                
    print("Hybrid Complex VAE training complete.")

if __name__ == "__main__":
    train_volume_hybrid_complex()
