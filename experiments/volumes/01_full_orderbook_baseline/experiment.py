import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import OrderbookDataset
from model import VAE, loss_function

def setup_logger(log_file):
    logger = logging.getLogger("VAE_Experiment")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

def run_experiment():
    # Configuration
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_file = os.path.join(RESULTS_DIR, "experiment.log")
    if os.path.exists(log_file):
        os.remove(log_file)
        
    logger = setup_logger(log_file)
    
    latent_dims = list(range(16, 481, 4))
    results = {
        'latent_dim': [],
        'best_val_loss': [],
        'best_val_mse': []
    }
    
    logger.info(f"Starting experiment on {DEVICE}")
    logger.info(f"Latent dimensions to test: {latent_dims}")
    
    # 1. Load and Prep Data
    full_dataset = OrderbookDataset(DATA_PATH)
    indices = np.arange(len(full_dataset))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(full_dataset.data[train_idx])
    full_dataset.data = scaler.transform(full_dataset.data)
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = full_dataset.data.shape[1]
    
    # 2. Loop through dimensions
    for dim in latent_dims:
        logger.info(f"\n===== TESTING LATENT DIM: {dim} =====")
        model = VAE(input_dim=input_dim, latent_dim=dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_dim_val_loss = float('inf')
        best_dim_val_mse = float('inf')
        epochs_no_improve = 0
        
        start_time = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon_batch, mu, log_var = model(batch)
                loss, mse, kld = loss_function(recon_batch, batch, mu, log_var)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            
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
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Dim {dim} | Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.4f}")
            
            if avg_val_loss < best_dim_val_loss:
                best_dim_val_loss = avg_val_loss
                best_dim_val_mse = avg_val_mse
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"vae_dim_{dim}.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    logger.info(f"Dim {dim} | Early stopping at epoch {epoch}. Best Val Loss: {best_dim_val_loss:.4f}")
                    break
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Finished Dim {dim} | Best Val Loss: {best_dim_val_loss:.4f} | Best Val MSE: {best_dim_val_mse:.4f} | Time: {duration:.2f}s")
        
        results['latent_dim'].append(dim)
        results['best_val_loss'].append(best_dim_val_loss)
        results['best_val_mse'].append(best_dim_val_mse)
        
    # 3. Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(results['latent_dim'], results['best_val_loss'], marker='o', label='Total Loss (MSE + KLD)')
    plt.plot(results['latent_dim'], results['best_val_mse'], marker='s', label='MSE (Reconstruction)')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Validation Metric')
    plt.title('Performance vs Latent Dimension')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "latent_dim_comparison.png"))
    
    # Save results summary
    with open(os.path.join(RESULTS_DIR, "experiment_results.txt"), 'w') as f:
        f.write("LatentDim\tValLoss\tValMSE\n")
        for i in range(len(results['latent_dim'])):
            f.write(f"{results['latent_dim'][i]}\t{results['best_val_loss'][i]:.4f}\t{results['best_val_mse'][i]:.4f}\n")
            
    logger.info("\nExperiment complete. Plot saved as 'latent_dim_comparison.png'.")

if __name__ == "__main__":
    run_experiment()
