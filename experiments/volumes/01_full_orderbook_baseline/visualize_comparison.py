import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import VAE
from dataset import OrderbookDataset

def visualize_comparison(dims=[48, 120, 192, 256, 360, 432]):
    DATA_PATH = os.path.join(
        ROOT_DIR,
        "@data",
        "orderbook_data.npy",
    )
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Loading data and scaler...")
    # Load dataset
    dataset = OrderbookDataset(DATA_PATH)
    
    # Load scaler params
    mean = np.load(os.path.join(MODELS_DIR, "scaler_mean.npy"))
    scale = np.load(os.path.join(MODELS_DIR, "scaler_scale.npy"))
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.var_ = scale ** 2 # scaler needs var_ internally
    scaler.n_samples_seen_ = 1 # dummy
    
    # Scale data
    scaled_data = scaler.transform(dataset.data)
    
    # Pick a random sample index from what would be the "test set" 
    # (re-using the same seed logic as in train.py if possible, but any index works)
    np.random.seed(42)
    sample_idx = np.random.randint(int(len(scaled_data) * 0.9), len(scaled_data))
    
    sample_tensor = torch.from_numpy(scaled_data[sample_idx:sample_idx+1]).to(DEVICE)
    input_dim = scaled_data.shape[1]
    
    # Extract original for plotting (unscaled)
    original_unscaled = dataset.data[sample_idx].reshape(2, 400, 2)
    
    plt.figure(figsize=(20, 15))
    
    for i, dim in enumerate(dims):
        model_path = os.path.join(MODELS_DIR, f"vae_dim_{dim}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping.")
            continue
            
        print(f"Reconstructing with Dim {dim}...")
        model = VAE(input_dim=input_dim, latent_dim=dim).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            recon, _, _ = model(sample_tensor)
            recon_unscaled = scaler.inverse_transform(recon.cpu().numpy()).reshape(2, 400, 2)
            
        # Plotting
        # Bids
        plt.subplot(len(dims), 2, i*2 + 1)
        plt.step(original_unscaled[0, :, 0], original_unscaled[0, :, 1], color='green', alpha=0.5, label='Original')
        plt.step(recon_unscaled[0, :, 0], recon_unscaled[0, :, 1], color='red', linestyle='--', label=f'Recon (Dim {dim})')
        plt.title(f'Bids (Dim {dim})')
        if i == 0: plt.legend()
        plt.yscale('log') # Log scale for amounts is often better for orderbooks
        
        # Asks
        plt.subplot(len(dims), 2, i*2 + 2)
        plt.step(original_unscaled[1, :, 0], original_unscaled[1, :, 1], color='blue', alpha=0.5, label='Original')
        plt.step(recon_unscaled[1, :, 0], recon_unscaled[1, :, 1], color='orange', linestyle='--', label=f'Recon (Dim {dim})')
        plt.title(f'Asks (Dim {dim})')
        if i == 0: plt.legend()
        plt.yscale('log')
        
    plt.tight_layout()
    output_filename = os.path.join(PLOTS_DIR, "comparison_grid.png")
    plt.savefig(output_filename)
    print(f"Saved comparison to {output_filename}")

if __name__ == "__main__":
    visualize_comparison()
