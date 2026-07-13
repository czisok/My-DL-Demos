import torch
import torch.nn.functional as F
import numpy as np
from VQVAE.model import VQVAEModel
from VQVAE.model_2 import VQVAEModelV2
import torch.optim as optim
from data_utils import get_cifar10_dataloader, get_mnist_dataloader, DATA_ROOT_PATH, init_data_root
import time
import torch.nn as nn
import einops
import cv2
import numpy as np

import argparse

import json

def parse_args():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="这是一个参数解析示例")

    # 2. 添加参数（必填/可选、类型、说明、默认值）
    parser.add_argument("data_root", type=str, default="./", help="data root path")

    # 3. 解析参数
    args = parser.parse_args()
    return args

def restet_global_set(args):
    with open("global_set.json", "r") as f:
        config = json.load(f)
    config['data_root'] = args.data_root
    with open("global_set.json", "w") as f:
        json.dump(config, f, indent=4)

def train_vqvae(model, train_loader, epoch, optimizer, device, data_variance):
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(epoch):
        s_time = time.time()
        for idx, (data, _) in enumerate(train_loader):
            # (data, _) = next(iter(train_loader))  # 由于样本量太大，此处仅训练第一个batch
            data = data.to(device)
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = model(data)  # vq 损失
            recon_error = F.mse_loss(data_recon, data) / data_variance  # 重构损失
            loss = recon_error + vq_loss  # 总损失
            loss.backward()
            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
        e_time = time.time()
        print('epoch %d, recon_error %.5f, perplexity: %.5f, elapsed %.2f s' % (
                (i+1),
                np.mean(train_res_recon_error),
                np.mean(train_res_perplexity),
                e_time - s_time)
            )

        if (i+1) % 100 == 0:
            print('epoch %d, recon_error %.5f, perplexity: %.5f' % (
                (i+1),
                np.mean(train_res_recon_error[-100:]),  # 最近100个batch的重构损失平均值
                np.mean(train_res_perplexity[-100:]))   # 最近100个batch的perplexity平均值
            )
    return train_res_recon_error, train_res_perplexity


def train_vqvae_2(model, dataloader, device="cuda", optimizer=None, n_epochs=100, l_w_embedding=1, l_w_commitment=0.25):
    print("batch size:", batch_size)
    model.to(device)
    model.train()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, ze, zq = model(x)
            l_reconstruct = F.mse_loss(x, x_hat)
            l_embedding = F.mse_loss(ze.detach(), zq)
            l_commitment = F.mse_loss(ze, zq.detach())
            loss = l_reconstruct + l_w_embedding * l_embedding + l_w_commitment * l_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        # torch.save(model.state_dict(), ckpt_path)
        print(f"epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s")
    print("Done")


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
    cv2.imwrite(f'./vqvae_reconstruct_{dataset_type}.jpg', x_cat)


if __name__ == '__main__':
    args = parse_args()
    init_data_root(args.data_root)
    # =========================================================================
    # 超参数设置
    # =========================================================================
    batch_size = 256
    epoch = 5  # 15000
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("Running on %s" % device)
    print("Training VQVAE")
    print("=" * 60)

    model = VQVAEModel(1, 1, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)

    # model_2 = VQVAEModelV2(1, 32, 32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    train_loader, _, = get_mnist_dataloader(batch_size)
    train_vqvae(model, train_loader, epoch, optimizer, device, 1.0)
    
    # train_vqvae_2(model_2, train_loader, device, optimizer, epoch)
    # img, _ = next(iter(train_loader))
    # img = img.to(device)
    # print("img.shape:", img.shape)
    # reconstruct(model_2, img, device, dataset_type='MNIST')

    # train_loader, test_loader, data_variance = get_cifar10_dataloader(batch_size, data_path='/Users/bytedance/Downloads/dataset_for_dl/cifar-10')
    # train_vqvae(model, train_loader, epoch, optimizer, device, data_variance)
