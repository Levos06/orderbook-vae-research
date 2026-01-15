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

def run_experiment_ols():
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    LATENT_DIMS = [4, 8, 32, 64]
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Starting OLS Detrended Experiment on {DEVICE}")
    
    # 1. Prepare Data
    full_dataset = DetrendedPriceDataset(DATA_PATH, method='ols')
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Scaling
    train_residuals = full_dataset.residuals[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_residuals)
    full_dataset.residuals = scaler.transform(full_dataset.residuals)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    results = {}
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for latent_dim in LATENT_DIMS:
        print(f"\n--- Training with Latent Dim: {latent_dim} ---")
        model = PriceVAE(input_dim=800, latent_dim=latent_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch_res, _, _ in train_loader:
                batch_res = batch_res.to(DEVICE).float()
                optimizer.zero_grad()
                recon_batch, mu, log_var = model(batch_res)
                loss, _, _ = loss_function(recon_batch, batch_res, mu, log_var)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_res, _, _ in val_loader:
                    batch_res = batch_res.to(DEVICE).float()
                    recon_batch, mu, log_var = model(batch_res)
                    loss, _, _ = loss_function(recon_batch, batch_res, mu, log_var)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            if epoch % 20 == 0:
                print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
                
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"vae_ols_dim_{latent_dim}.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stop at {epoch}")
                    break
        
        results[latent_dim] = best_val_loss

    # Visual Comparison
    test_subset = Subset(full_dataset, test_idx)
    res_batch, m_batch, b_batch = next(iter(DataLoader(test_subset, batch_size=3, shuffle=False)))
    res_batch_gpu = res_batch.to(DEVICE).float()
    
    fig, axes = plt.subplots(len(LATENT_DIMS), 3, figsize=(18, 4 * len(LATENT_DIMS)))
    
    # Precompute original residuals for plotting
    orig_res = scaler.inverse_transform(res_batch.numpy())
    
    for i, latent_dim in enumerate(LATENT_DIMS):
        # Re-initialize model with correct latent_dim
        model = PriceVAE(input_dim=800, latent_dim=latent_dim).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"vae_ols_dim_{latent_dim}.pth"), map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            recon_scaled, _, _ = model(res_batch_gpu)
            recon_res = scaler.inverse_transform(recon_scaled.cpu().numpy())
            
            for j in range(3):
                ax = axes[i, j]
                ax.plot(orig_res[j], color='forestgreen', alpha=0.5, label='Original')
                ax.plot(recon_res[j], color='crimson', linestyle='--', label='Recon')
                ax.set_title(f'Dim {latent_dim} | Sample {j+1}')
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.set_ylabel(f'Latent Dim {latent_dim}')
                if i == len(LATENT_DIMS) - 1:
                    ax.set_xlabel('Orderbook Level')
                if i == 0 and j == 2:
                    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "ols_dim_comparison.png")
    plt.savefig(plot_path)
    print(f"\nExperiment complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    run_experiment_ols()
