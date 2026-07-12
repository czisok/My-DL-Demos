
import torch
from torch import nn
# ============================================================
# Part 1: 自动编码器 (Auto-Encoder)
# ============================================================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=728, latent_dim=32):
        super(AutoEncoder, self).__init__()

        # Encoder: 784 -> 512 -> 256 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder: latent_dim -> 256 -> 512 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),  # 输出范围 [0, 1]，与输入像素值范围一致
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z