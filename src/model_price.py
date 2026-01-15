import torch
import torch.nn as nn
import torch.nn.functional as F

class PriceVAE(nn.Module):
    def __init__(self, input_dim=800, latent_dim=16, hidden_dims=[512, 256, 128]):
        super(PriceVAE, self).__init__()
        
        # Encoder
        modules = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            last_dim = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        
        reversed_dims = hidden_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(reversed_dims[i], reversed_dims[i+1]),
                    nn.BatchNorm1d(reversed_dims[i+1]),
                    nn.LeakyReLU(),
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)
        
    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, beta=1.0):
    # MSE Loss (reconstruction)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return MSE + beta * KLD, MSE, KLD
