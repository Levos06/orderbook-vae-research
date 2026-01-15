import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
from utils_detrend import detrend_prices # Reusing OLS logic

def visualize_processed_volumes(data_path=None, num_samples=3):
    if data_path is None:
        data_path = os.path.join(
            ROOT_DIR,
            "experiments",
            "volumes",
            "00_data_prep_orderbook",
            "data",
            "orderbook_data.npy",
        )
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # 1. Load data: (N, 2, 400, 2)
    data = np.load(data_path)
    N = data.shape[0]
    indices = np.random.choice(N, num_samples, replace=False)
    
    # 2. Extract Volumes in sequence: Bid 400 -> Bid 1 -> Ask 1 -> Ask 400
    bids_vol = data[:, 0, :, 1] # (N, 400) - best is index 0
    bids_vol_inverted = bids_vol[:, ::-1] # (N, 400) - Bid 400 is first
    asks_vol = data[:, 1, :, 1] # (N, 400) - best is index 0
    
    vol_sequence = np.concatenate([bids_vol_inverted, asks_vol], axis=1) # (N, 800)
    
    # 3. Cumulative Sum
    cum_vol = np.cumsum(vol_sequence, axis=1) # (N, 800)
    
    # 4. OLS Detrending
    # We reuse detrend_prices because the math for fitting a line to 800 points is identical
    residuals, m, b = detrend_prices(cum_vol, method='ols')
    
    # 5. Plotting
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Column 1: Raw Volumes
        ax1 = axes[i, 0]
        ax1.bar(np.arange(800), vol_sequence[idx], color='blue', alpha=0.4, width=1.0)
        ax1.set_title(f'Sample {idx} - Raw Volumes')
        ax1.set_ylabel('Volume')
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Cumulative Curve
        ax2 = axes[i, 1]
        ax2.plot(cum_vol[idx], color='black', linewidth=1.5, label='CumSum')
        # Show the trend line
        x = np.arange(800)
        trend_line = m[idx] * x + b[idx]
        ax2.plot(trend_line, color='red', linestyle='--', alpha=0.8, label='OLS Trend')
        ax2.set_title(f'Cumulative Volume Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Residuals (The "Signal" for VAE)
        ax3 = axes[i, 2]
        ax3.plot(residuals[idx], color='purple', linewidth=1.5)
        ax3.set_title(f'Detrended Residuals')
        # Normalize visually to emphasize the "shape"
        ax3.axhline(0, color='black', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        if i == num_samples - 1:
            ax1.set_xlabel('Level (Bid 400 -> Ask 400)')
            ax2.set_xlabel('Level (Bid 400 -> Ask 400)')
            ax3.set_xlabel('Level (Bid 400 -> Ask 400)')

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "processed_volumes_ols.png")
    plt.savefig(out_path, dpi=120)
    print(f"Processed volume samples saved to {out_path}")

if __name__ == "__main__":
    visualize_processed_volumes()
