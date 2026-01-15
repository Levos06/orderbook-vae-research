import torch
import torch.nn as nn

class PriceConvVAEUltra(nn.Module):
    def __init__(self, input_dim=800, latent_dim=32):
        super(PriceConvVAEUltra, self).__init__()
        self.input_dim = input_dim
        
        # Encoder: (Batch, 1, 800)
        # Deepening to 5 layers, focusing on small kernels for micro-structure
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),   # (B, 32, 400)
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 200)
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), # (B, 128, 100)
            nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1), # (B, 256, 50)
            nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1), # (B, 512, 25)
            nn.BatchNorm1d(512), nn.LeakyReLU(),
        )
        
        self.fc_mu = nn.Linear(512 * 25, latent_dim)
        self.fc_var = nn.Linear(512 * 25, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 25)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1), # (B, 256, 50)
            nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # (B, 128, 100)
            nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 200)
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # (B, 32, 400)
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),    # (B, 1, 800)
        )
        
    def encode(self, x):
        x = x.unsqueeze(1)
        conv_out = self.encoder_conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        mu = self.fc_mu(conv_out)
        log_var = self.fc_var(conv_out)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(result.size(0), 512, 25)
        result = self.decoder_conv(result)
        return result.squeeze(1)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
