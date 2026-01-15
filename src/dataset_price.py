import torch
from torch.utils.data import Dataset
import numpy as np

class PriceDataset(Dataset):
    def __init__(self, data_path, transform=None):
        raw_data = np.load(data_path).astype(np.float32)
        # raw_data shape: (N, 2, 400, 2)
        # side 0=Bid, side 1=Ask
        # level 0-399 (0 is best price)
        # feature 0=Price, 1=Amount
        
        # Extraction
        # Bid 400...Bid 1 (increasing price)
        bids_prices = raw_data[:, 0, :, 0] # (N, 400) - bids[0] is highest, bids[399] is lowest
        bids_inverted = bids_prices[:, ::-1] # Now lowest bid first
        
        # Ask 1...Ask 400 (increasing price)
        asks_prices = raw_data[:, 1, :, 0] # (N, 400) - asks[0] is lowest, asks[399] is highest
        
        # Concatenate: [Bids_low_to_high, Asks_low_to_high]
        # This forms a single ascending price line
        self.data = np.concatenate([bids_inverted, asks_prices], axis=1) # (N, 800)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.from_numpy(sample)
