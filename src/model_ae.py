import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardAE(nn.Module):
    def __init__(self, input_dim=800, latent_dim=32):
        super(StandardAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def ae_loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')
