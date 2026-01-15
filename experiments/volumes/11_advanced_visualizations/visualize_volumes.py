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

def visualize_volumes(data_path=None, num_samples=5):
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

    # Load data: (N, 2, 400, 2)
    # side 0=Bid, side 1=Ask
    # level 0-399 (0 is best price/volume)
    # feature 0=Price, 1=Amount (Volume)
    data = np.load(data_path)
    N = data.shape[0]
    
    # Pick random samples
    indices = np.random.choice(N, num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i, idx in enumerate(indices):
        sample = data[idx]
        
        # Bids: Best is level 0 (highest price)
        # Asks: Best is level 0 (lowest price)
        # We want to plot them relative to the "center" (spread)
        
        bid_vols = sample[0, :, 1]
        ask_vols = sample[1, :, 1]
        
        # Create a symmetric x-axis: levels 400 down to 1 (Bids), 1 up to 400 (Asks)
        x_bids = np.arange(-400, 0)
        x_asks = np.arange(1, 401)
        
        ax = axes[i]
        # Bids in Green (negative X for visualization)
        ax.fill_between(x_bids, bid_vols[::-1], step="mid", color='forestgreen', alpha=0.6, label='Bids')
        # Asks in Red
        ax.fill_between(x_asks, ask_vols, step="mid", color='crimson', alpha=0.6, label='Asks')
        
        ax.set_title(f'Orderbook Volumes - Sample {idx}', fontweight='bold')
        ax.set_ylabel('Volume')
        ax.set_xlabel('Levels from Spread (Bids < 0 < Asks)')
        ax.grid(True, linestyle=':', alpha=0.5)
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "volume_samples.png")
    plt.savefig(out_path, dpi=120)
    print(f"Volume samples saved to {out_path}")

if __name__ == "__main__":
    visualize_volumes()
