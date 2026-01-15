import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_detrended import DetrendedPriceDataset
from model_price import PriceVAE, loss_function
from utils_detrend import get_trend

def train_detrended(method='ols'):
    # Configuration
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    BATCH_SIZE = 128
    LATENT_DIM = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Training with detrend method: {method.upper()}")
    
    # 1. Load Data
    full_dataset = DetrendedPriceDataset(DATA_PATH, method=method)
    data_size = len(full_dataset)
    indices = np.arange(data_size)
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 2. Scaling Residuals
    # Scale based on training residuals
    train_residuals = full_dataset.residuals[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_residuals)
    
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    input_dim = 800
    model = PriceVAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 4. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_res, _, _ in train_loader:
            batch_res = batch_res.to(DEVICE).float()
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch_res)
            loss, _, _ = loss_function(recon_batch, batch_res, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_res, _, _ in val_loader:
                batch_res = batch_res.to(DEVICE).float()
                recon_batch, mu, log_var = model(batch_res)
                loss, _, _ = loss_function(recon_batch, batch_res, mu, log_var)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/{EPOCHS}] Train Loss: {train_loss/len(train_loader.dataset):10.4f} | Val Loss: {avg_val_loss:10.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_path = os.path.join(MODELS_DIR, f"vae_price_detrend_{method}.pth")
            torch.save(model.state_dict(), model_path)
            np.save(os.path.join(MODELS_DIR, f"scaler_detrend_{method}_mean.npy"), scaler.mean_)
            np.save(os.path.join(MODELS_DIR, f"scaler_detrend_{method}_scale.npy"), scaler.scale_)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 5. Visualization
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"vae_price_detrend_{method}.pth")))
    model.eval()
    with torch.no_grad():
        test_subset = Subset(full_dataset, test_idx)
        res_batch, m_batch, b_batch = next(iter(DataLoader(test_subset, batch_size=5, shuffle=False)))
        
        res_batch = res_batch.to(DEVICE).float()
        recon_res_scaled, _, _ = model(res_batch)
        
        orig_res = scaler.inverse_transform(res_batch.cpu().numpy())
        recon_res = scaler.inverse_transform(recon_res_scaled.cpu().numpy())
        
        m_batch, b_batch = m_batch.numpy().reshape(-1, 1), b_batch.numpy().reshape(-1, 1)
        # Safe trend calculation: (B, 1) * (1, 800) -> (B, 800)
        x_indices = np.arange(800).reshape(1, -1)
        trends = m_batch * x_indices + b_batch 
        
        orig_prices = trends + orig_res
        recon_prices = trends + recon_res
        
        # New Plotting Logic
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        for i in range(3):
            # Left: Residuals
            ax_res = axes[i, 0]
            ax_res.plot(orig_res[i], color='forestgreen', alpha=0.6, label='Original Residual', linewidth=1.5)
            ax_res.plot(recon_res[i], color='crimson', label='Reconstructed Residual', linestyle='--', linewidth=1.2)
            ax_res.set_title(f'Sample {i+1} Residual (Method: {method.upper()})', fontweight='bold')
            ax_res.legend(loc='upper right', fontsize='small')
            ax_res.grid(True, linestyle=':', alpha=0.6)
            
            # Right: Full Price Reconstruction
            ax_price = axes[i, 1]
            ax_price.plot(orig_prices[i], color='forestgreen', alpha=0.6, label='Original Price', linewidth=1.5)
            ax_price.plot(recon_prices[i], color='crimson', label='Reconstructed Price', linestyle='--', linewidth=1.2)
            ax_price.plot(trends[i], color='black', linestyle=':', alpha=0.4, label='Trend Line')
            
            # Zoom in on the price to show the details if they are too small
            price_range = np.max(orig_prices[i]) - np.min(orig_prices[i])
            ax_price.set_ylim(np.min(orig_prices[i]) - 0.05 * price_range, np.max(orig_prices[i]) + 0.05 * price_range)
            
            ax_price.set_title(f'Sample {i+1} Full Price Reconstruction', fontweight='bold')
            ax_price.legend(loc='upper left', fontsize='small')
            ax_price.grid(True, linestyle=':', alpha=0.6)
            
            # Label Axes
            ax_res.set_ylabel('Residual Value')
            ax_price.set_ylabel('Price Level')
            if i == 2:
                ax_res.set_xlabel('Orderbook Level (Bid 400 -> Ask 400)')
                ax_price.set_xlabel('Orderbook Level (Bid 400 -> Ask 400)')
            
        fig.suptitle(f'Detrended VAE Reconstruction Performance ({method.upper()})', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(PLOTS_DIR, f"detrended_recon_{method}.png")
        plt.savefig(plot_path, dpi=120)
        print(f"Improved plot saved to {plot_path}")

if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else 'ols'
    train_detrended(method)
