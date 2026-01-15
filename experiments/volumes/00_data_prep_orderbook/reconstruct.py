import os
import sys
import pandas as pd
import numpy as np
import tqdm
import random

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "@data")
PARQUET_PATH = os.path.join(DATA_DIR, "BTC-USDT-L2orderbook-400lv-2025-08-28.parquet")
OUTPUT_NPY = os.path.join(DATA_DIR, "orderbook_data.npy")

def reconstruct_and_sample(
    input_file,
    output_file,
    target_states=150000,
    sample_size=50000,
    levels=400
):
    print(f"Reading {input_file}...")
    # Read the file
    df = pd.read_parquet(input_file)
    
    print(f"Data shape: {df.shape}")
    
    # We will maintain two dictionaries for the current state: bids and asks
    # Key: Price, Value: Amount
    current_bids = {}
    current_asks = {}
    
    states = []
    
    # Group by timestamp to process atomic updates
    grouped = df.groupby('timestamp', sort=False)
    
    pbar = tqdm.tqdm(total=target_states, desc="Reconstructing states")
    
    for timestamp, group in grouped:
        if len(states) >= target_states:
            break
            
        is_snapshot =  'snapshot' in group['action'].values
        
        if is_snapshot:
            current_bids = {}
            current_asks = {}
            
        for _, row in group.iterrows():
            side = row['side']
            price = row['price']
            amount = row['amount']
            
            target_dict = current_bids if side == 'bid' else current_asks
            
            # Assuming amount 0 means delete. 
            # In some L2 feeds only explicit deletes remove, but often 0 qty is the standard.
            if amount == 0:
                if price in target_dict:
                    del target_dict[price]
            else:
                target_dict[price] = amount
                
        # Sort keys
        # Bids: Descending
        sorted_bids = sorted(current_bids.items(), key=lambda x: x[0], reverse=True)[:levels]
        # Asks: Ascending
        sorted_asks = sorted(current_asks.items(), key=lambda x: x[0])[:levels]
        
        # We save if we have at least some data.
        if len(sorted_bids) > 0 and len(sorted_asks) > 0:
             # Structure: [2, levels, 2] -> (Side, Level, [Price, Amount])
             state = np.zeros((2, levels, 2), dtype=np.float32)
             
             for i, (p, a) in enumerate(sorted_bids):
                 state[0, i, 0] = p
                 state[0, i, 1] = a
                 
             for i, (p, a) in enumerate(sorted_asks):
                 state[1, i, 0] = p
                 state[1, i, 1] = a
                 
             states.append(state)
             pbar.update(1)
             
    pbar.close()
    
    print(f"Collected {len(states)} states.")
    
    if len(states) < sample_size:
        print(f"Warning: requested {sample_size} samples but only have {len(states)}. returning all.")
        final_data = np.array(states)
    else:
        print(f"Sampling {sample_size} random states...")
        final_data = np.array(random.sample(states, sample_size))
        
    print(f"Final data shape: {final_data.shape}")
    np.save(output_file, final_data)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    reconstruct_and_sample(
        input_file=PARQUET_PATH,
        output_file=OUTPUT_NPY,
        target_states=150000,
        sample_size=50000,
        levels=400
    )
