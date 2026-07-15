import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time
import cv2
import einops
import numpy as np
from data_utils import get_mnist_dataloader
import time


class ResidualBlock(nn.Module):
    """
    残差卷积: x -> ReLU -> Conv -> ReLU -> Conv + x -> out
    """
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class VQVAEModelV2(nn.Module):
    """
        简易版 VQVAE, 用于 MNIST 数据集
    """
    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            ResidualBlock(dim),
        )
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim),
            ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
        )
        self.n_downsample = 2

    def forward(self, x):
        # encode
        ze = self.encoder(x)
        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        # make C to the second dim
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        embedding = self.vq_embedding.weight.data

        # ze: [N, C, H, W]
        # embedding [K, C]
        N, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(N, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)

def reconstruct(model, x, device, dataset_type='MNIST'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'work_dirs/vqvae_reconstruct_{dataset_type}.jpg', x_cat)


def sample_imgs(vqvae, gen_model, img_shape, n_sample=81, device='cuda', dataset_type='MNIST'):
    vqvae = vqvae.to(device)
    vqvae.eval()
    gen_model = gen_model.to(device)
    gen_model.eval()

    C, H, W = img_shape
    H, W = vqvae.get_latent_HW((C, H, W))
    input_shape = (n_sample, H, W)
    x = torch.zeros(input_shape).to(device).to(torch.long)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = gen_model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1)
                x[:, i, j] = pixel[:, 0]

    imgs = vqvae.decode(x)

    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(
        imgs, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=int(n_sample**0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'work_dirs/vqvae_sample_{dataset_type}.jpg', imgs)


# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     mnist_cfg1 = dict(dataset_type='MNIST',
#                       img_shape=(1, 28, 28),
#                       dim=32,
#                       n_embedding=32,
#                       batch_size=128,
#                       n_epochs=20,
#                       l_w_embedding=1,
#                       l_w_commitment=0.25,
#                       lr=2e-4,
#                       n_epochs_2=50,
#                       batch_size_2=256,
#                       pixelcnn_n_blocks=15,
#                       pixelcnn_dim=128,
#                       pixelcnn_linear_dim=32,
#                       vqvae_path='./model_mnist.pth',
#                       gen_model_path='./gen_model_mnist.pth')
#     train_dataloader, _ = get_mnist_dataloader(128)
    
#     img_shape = mnist_cfg1['img_shape']
#     vqvae = VQVAE(img_shape[0], mnist_cfg1['dim'], mnist_cfg1['n_embedding'])
#     # 1. Train VQVAE
#     train_vqvae(vqvae, train_dataloader, device)

#     # 2. Test VQVAE by visualizaing reconstruction result
#     vqvae.load_state_dict(torch.load(mnist_cfg1['vqvae_path']))
    
#     img = next(train_dataloader).to(device)
#     reconstruct(vqvae, img, device, mnist_cfg1['dataset_type'])
