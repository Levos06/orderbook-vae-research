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

from dataset import OrderbookDataset
from model import VAE, loss_function

def train():
    # Configuration
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
    BATCH_SIZE = 128
    LATENT_DIM = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    full_dataset = OrderbookDataset(DATA_PATH)
    data_size = len(full_dataset)
    indices = np.arange(data_size)
    
    # Split indices: 80% train, 10% val, 10% test
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 2. Scaling (Fit only on train data)
    # We need to flatten the whole train set to calculate mean/std
    train_data_raw = full_dataset.data[train_idx]
    scaler = StandardScaler()
    scaler.fit(train_data_raw)
    
    # Update dataset with scaled data
    full_dataset.data = scaler.transform(full_dataset.data)
    
    # Create DataLoaders
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model
    input_dim = full_dataset.data.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM).to(DEVICE)
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
        
        # Validation
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
        
        print(f"[{epoch:3d}/{EPOCHS}] Train Loss: {avg_train_loss:10.4f} | Val Loss: {avg_val_loss:10.4f} | Val MSE: {avg_val_mse:10.4f}")
        
        # Early Stopping & Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_path = os.path.join(MODELS_DIR, "vae_best_model.pth")
            torch.save(model.state_dict(), model_path)
            # Save scaler params too for later use
            np.save(os.path.join(MODELS_DIR, "scaler_mean.npy"), scaler.mean_)
            np.save(os.path.join(MODELS_DIR, "scaler_scale.npy"), scaler.scale_)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # 5. Final Evaluation on Test Set
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "vae_best_model.pth")))
    model.eval()
    test_mse = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            recon_batch, mu, log_var = model(batch)
            _, mse, _ = loss_function(recon_batch, batch, mu, log_var)
            test_mse += mse.item()
    print(f"Test MSE: {test_mse / len(test_loader.dataset):.4f}")
    
    # 6. Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"))
    
    # Visualization of reconstructions
    with torch.no_grad():
        # Get a sample from test set
        sample_batch = next(iter(test_loader)).to(DEVICE)
        reconstructed, _, _ = model(sample_batch)
        
        # Unscale for better looking plot
        orig_sample = scaler.inverse_transform(sample_batch.cpu().numpy())
        recon_sample = scaler.inverse_transform(reconstructed.cpu().numpy())
        
        # Reshape to (N, 2, 400, 2)
        orig_sample = orig_sample.reshape(-1, 2, 400, 2)
        recon_sample = recon_sample.reshape(-1, 2, 400, 2)
        
        plt.figure(figsize=(15, 8))
        # Plot 3 samples: Original vs Reconstructed
        for i in range(3):
            # Bids
            plt.subplot(3, 2, i*2 + 1)
            plt.step(orig_sample[i, 0, :, 0], orig_sample[i, 0, :, 1], color='green', label='Original Bid')
            plt.step(recon_sample[i, 0, :, 0], recon_sample[i, 0, :, 1], color='red', linestyle='--', label='Recon Bid')
            plt.title(f'Sample {i+1} Bids')
            plt.legend()
            
            # Asks
            plt.subplot(3, 2, i*2 + 2)
            plt.step(orig_sample[i, 1, :, 0], orig_sample[i, 1, :, 1], color='blue', label='Original Ask')
            plt.step(recon_sample[i, 1, :, 0], recon_sample[i, 1, :, 1], color='orange', linestyle='--', label='Recon Ask')
            plt.title(f'Sample {i+1} Asks')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "reconstructions.png"))
    
    print("Training finished. Results saved.")

if __name__ == "__main__":
    train()
