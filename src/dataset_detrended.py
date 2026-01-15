import torch
from torch.utils.data import Dataset
import numpy as np
from utils_detrend import detrend_prices

class DetrendedPriceDataset(Dataset):
    def __init__(self, data_path, method='ols'):
        raw_data = np.load(data_path).astype(np.float32)
        
        # Original sequence: Bid 400 -> ... -> Bid 1 -> Ask 1 -> ... -> Ask 400 (Ascending)
        bids_prices = raw_data[:, 0, :, 0]
        bids_inverted = bids_prices[:, ::-1]
        asks_prices = raw_data[:, 1, :, 0]
        
        self.original_prices = np.concatenate([bids_inverted, asks_prices], axis=1) # (N, 800)
        
        # Detrend
        self.residuals, self.m, self.b = detrend_prices(self.original_prices, method=method)
        
    def __len__(self):
        return len(self.residuals)

    def __getitem__(self, idx):
        return torch.from_numpy(self.residuals[idx]).float(), \
               torch.from_numpy(self.m[idx:idx+1]).float(), \
               torch.from_numpy(self.b[idx:idx+1]).float()
