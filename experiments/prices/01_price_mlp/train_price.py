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

from dataset_price import PriceDataset
from model_price import PriceVAE, loss_function

def train_price():
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
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    full_dataset = PriceDataset(DATA_PATH)
    data_size = len(full_dataset)
    indices = np.arange(data_size)
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 2. Scaling
    # For prices, we can use a global scaler or subtract the mid-price.
    # Standard scaler on the whole 800-vector is easier.
    train_data_raw = full_dataset.data[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_data_raw)
    
    full_dataset.data = scaler.transform(full_dataset.data)
    
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
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        train_mse = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss, mse, kld = loss_function(recon_batch, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse += mse.item()
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_mse = train_mse / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                recon_batch, mu, log_var = model(batch)
                loss, mse, kld = loss_function(recon_batch, batch, mu, log_var)
                val_loss += loss.item()
                val_mse += mse.item()
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mse = val_mse / len(val_loader.dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_mse'].append(avg_val_mse)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:3d}/{EPOCHS}] Train Loss: {avg_train_loss:10.4f} | Val Loss: {avg_val_loss:10.4f} | Val MSE: {avg_val_mse:10.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "vae_price_best.pth"))
            np.save(os.path.join(MODELS_DIR, "scaler_price_mean.npy"), scaler.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_price_scale.npy"), scaler.scale_)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # 5. Visualizations
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Price VAE Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train')
    plt.plot(history['val_mse'], label='Val')
    plt.title('Price VAE MSE')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "price_training_curves.png"))
    
    # Reconstruction Visualization
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "vae_price_best.pth")))
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader)).to(DEVICE)
        recon, _, _ = model(batch)
        
        orig = scaler.inverse_transform(batch.cpu().numpy())
        recon_data = scaler.inverse_transform(recon.cpu().numpy())
        
        plt.figure(figsize=(15, 10))
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(orig[i], 'g-', alpha=0.5, label='Original')
            plt.plot(recon_data[i], 'r--', label='Reconstructed')
            plt.title(f'Sample {i+1} Price Line (Ascending)')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "price_reconstructions.png"))
    
    print("Price-only training complete.")

if __name__ == "__main__":
    train_price()
