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
from model_ae import StandardAE, ae_loss_function

def train_volume_ae():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    BATCH_SIZE = 128
    LATENT_DIM = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training Standard AE on {DEVICE}")
    
    full_dataset = LogCumVolumeDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, _ = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(full_dataset.residuals[train_idx])
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    model = StandardAE(input_dim=800, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_res, _, _ in train_loader:
            batch_res = batch_res.to(DEVICE).float()
            optimizer.zero_grad()
            recon_batch = model(batch_res)
            loss = ae_loss_function(recon_batch, batch_res)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_res, _, _ in val_loader:
                batch_res = batch_res.to(DEVICE).float()
                recon_batch = model(batch_res)
                loss = ae_loss_function(recon_batch, batch_res)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/100] AE Val Loss: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "ae_volume_best.pth"))
            np.save(os.path.join(MODELS_DIR, "scaler_ae_mean.npy"), scaler.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_ae_scale.npy"), scaler.scale_)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break
                
    print("Standard AE training complete.")

if __name__ == "__main__":
    train_volume_ae()
