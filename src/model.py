import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=1600, hidden_dims=[1024, 512, 256], latent_dim=20):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU())
            in_dim = h_dim
            
        self.encoder_body = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        reversed_hidden_dims = hidden_dims[::-1]
        
        for h_dim in reversed_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU())
            in_dim = h_dim
            
        self.decoder_body = nn.Sequential(*decoder_layers)
        
        self.final_layer = nn.Linear(hidden_dims[0], input_dim)
        
    def encode(self, x):
        result = self.encoder_body(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        result = self.decoder_body(z)
        result = self.final_layer(result)
        return result
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

def loss_function(recon_x, x, mu, log_var, beta=1.0):
    # MSE Loss
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return MSE + beta * KLD, MSE, KLD
