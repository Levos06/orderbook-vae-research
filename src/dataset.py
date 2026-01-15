import torch
from torch.utils.data import Dataset
import numpy as np

class OrderbookDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(data_path).astype(np.float32)
        # Check shape: Expected (N, 2, 400, 2)
        # Flatten to (N, 1600) for simple VAE or keep as is for ConvVAE
        # Proposed plan was linear VAE, so we can flatten in __getitem__ or here.
        
        # Let's flatten here to make things easier for the scaler
        self.original_shape = self.data.shape[1:]
        self.data = self.data.reshape(self.data.shape[0], -1)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.from_numpy(sample)
