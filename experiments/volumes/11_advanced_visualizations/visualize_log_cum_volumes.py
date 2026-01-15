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
from utils_detrend import detrend_prices

def visualize_log_cum_volumes(data_path=None, num_samples=3):
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

    # 1. Load data
    data = np.load(data_path)
    N = data.shape[0]
    indices = np.random.choice(N, num_samples, replace=False)
    
    # 2. Extract Volumes in sequence: Bid 400 -> Ask 400
    bids_vol = data[:, 0, :, 1]
    bids_vol_inverted = bids_vol[:, ::-1]
    asks_vol = data[:, 1, :, 1]
    vol_sequence = np.concatenate([bids_vol_inverted, asks_vol], axis=1) # (N, 800)
    
    # 3. Log Transform
    log_vol = np.log1p(vol_sequence)
    
    # 4. Cumulative Sum
    log_cum_vol = np.cumsum(log_vol, axis=1)
    
    # 5. OLS Detrending
    residuals, m, b = detrend_prices(log_cum_vol, method='ols')
    
    # 6. Plotting
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Column 1: Log Volumes (Raw vs Log)
        ax1 = axes[i, 0]
        ax1.plot(log_vol[idx], color='blue', alpha=0.7)
        ax1.set_title(f'Sample {idx} - Log(1+Vol)')
        ax1.set_ylabel('Log Volume')
        ax1.grid(True, alpha=0.3)
        
        # Column 2: Log Cumulative Curve
        ax2 = axes[i, 1]
        ax2.plot(log_cum_vol[idx], color='black', linewidth=1.5, label='LogCumSum')
        x = np.arange(800)
        trend_line = m[idx] * x + b[idx]
        ax2.plot(trend_line, color='red', linestyle='--', alpha=0.8, label='OLS Trend')
        ax2.set_title(f'Log-Cumulative Volume Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Column 3: Residuals
        ax3 = axes[i, 2]
        ax3.plot(residuals[idx], color='darkorange', linewidth=1.5)
        ax3.set_title(f'Detrended Residuals (Log Scale)')
        ax3.axhline(0, color='black', alpha=0.3)
        ax3.grid(True, alpha=0.3)
        
        if i == num_samples - 1:
            ax1.set_xlabel('Level (Bid 400 -> Ask 400)')
            ax2.set_xlabel('Level (Bid 400 -> Ask 400)')
            ax3.set_xlabel('Level (Bid 400 -> Ask 400)')

    plt.suptitle("Pipeline: Log(1+Vol) -> Cumulative Sum -> OLS Detrend", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "log_cum_volumes_ols.png")
    plt.savefig(out_path, dpi=120)
    print(f"Log-Cumulative volume samples saved to {out_path}")

if __name__ == "__main__":
    visualize_log_cum_volumes()
