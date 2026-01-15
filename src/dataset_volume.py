import torch
from torch.utils.data import Dataset
import numpy as np
from utils_detrend import detrend_prices

class LogCumVolumeDataset(Dataset):
    def __init__(self, data_path, method='ols'):
        raw_data = np.load(data_path).astype(np.float32)
        
        # Sequence: Bid 400 -> ... -> Bid 1 -> Ask 1 -> ... -> Ask 400
        bids_vol = raw_data[:, 0, :, 1]
        bids_inverted = bids_vol[:, ::-1]
        asks_vol = raw_data[:, 1, :, 1]
        
        self.raw_vols = np.concatenate([bids_inverted, asks_vol], axis=1) # (N, 800)
        
        # 1. Log transform
        self.log_vols = np.log1p(self.raw_vols)
        
        # 2. Cumulative sum
        self.log_cum_vols = np.cumsum(self.log_vols, axis=1)
        
        # 3. Detrend
        self.residuals, self.m, self.b = detrend_prices(self.log_cum_vols, method=method)
        
    def __len__(self):
        return len(self.residuals)

    def __getitem__(self, idx):
        # We need the trend params to reconstruct the curve later
        return torch.from_numpy(self.residuals[idx]).float(), \
               torch.from_numpy(self.m[idx:idx+1]).float(), \
               torch.from_numpy(self.b[idx:idx+1]).float()
