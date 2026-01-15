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

from dataset_detrended import DetrendedPriceDataset
from model_price_conv import PriceConvVAE
from model_price import loss_function

def train_price_conv():
    # Configuration
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    BATCH_SIZE = 128
    LATENT_DIM = 16 # Same as MLP for fair comparison
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training Convolutional Price VAE on {DEVICE}")
    
    # 1. Load Data (OLS method)
    full_dataset = DetrendedPriceDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 2. Scaling
    train_residuals = full_dataset.residuals[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_residuals)
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model
    model = PriceConvVAE(input_dim=800, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_res, _, _ in train_loader:
            batch_res = batch_res.to(DEVICE).float()
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch_res)
            loss, mse, kld = loss_function(recon_batch, batch_res, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_res, _, _ in val_loader:
                batch_res = batch_res.to(DEVICE).float()
                recon_batch, mu, log_var = model(batch_res)
                loss, mse, kld = loss_function(recon_batch, batch_res, mu, log_var)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/100] Val Loss: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "vae_price_conv_best.pth"))
            np.save(os.path.join(MODELS_DIR, "scaler_price_conv_mean.npy"), scaler.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_price_conv_scale.npy"), scaler.scale_)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping.")
                break
                
    print("Price Conv1D training complete.")

if __name__ == "__main__":
    train_price_conv()
