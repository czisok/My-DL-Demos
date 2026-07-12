
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================
# Part 2: 变分自编码器 (Variational Auto-Encoder)
# ============================================================
class VAE(nn.Module):
    def __init__(self, input_dim=724, latent_dim=32):
        super(VAE, self).__init__()

        # Encoder: 输出 mu 和 log_var
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)       # 均值 μ
        self.fc_logvar = nn.Linear(256, latent_dim)   # 对数方差 log(σ²)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, input_dim)

    def encode(self, x):
        """编码器：输入 x，输出分布参数 mu 和 log_var"""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧：
        z = mu + sigma * epsilon
        其中 epsilon ~ N(0, I)
        sigma = exp(0.5 * log_var)
        """
        std = torch.exp(0.5 * log_var)  # σ = exp(log(σ²)/2)
        eps = torch.randn_like(std)      # ε ~ N(0, I)
        z = mu + std * eps
        return z

    def decode(self, z):
        """解码器：从潜在变量 z 重建输入"""
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        x_recon = torch.sigmoid(self.fc5(h))
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss_function(x_recon, x, mu, log_var):
    """
    VAE 损失函数 = 重建损失 + KL 散度

    重建损失：BCE(x, x_recon)
    KL 散度：-0.5 * sum(1 + log(σ²) - μ² - σ²)
    """
    # 重建损失 (Binary Cross-Entropy)
    # 像素归一化到[0, 1]之间后，建议使用bce损失
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 散度: D_KL(q(z|x) || p(z))
    # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return (recon_loss + kl_loss) / x.size(0), recon_loss / x.size(0), kl_loss / x.size(0)
