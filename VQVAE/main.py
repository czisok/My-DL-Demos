from data_utils import get_mnist_dataloader
import torch
import torch.nn.functional as F
import numpy as np
from VQVAE.model import VQVAEModel
from VQVAE.model_simple import VQVAEModelV2
import torch.optim as optim
from data_utils import get_dataloader
import time
import torch.nn as nn
import einops
import cv2
import numpy as np
import argparse
import json
import os
from scipy.signal import savgol_filter
from my_utils import plt, show_image, show_image_by_array
from torchvision.utils import make_grid


cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
def parse_args():
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="这是一个参数解析示例")

    # 2. 添加参数（必填/可选、类型、说明、默认值）
    parser.add_argument("--data_root", type=str, default="./", help="data root path")
    parser.add_argument("--data_type", type=str, default="mnist", choices=["mnist", "cifar10"],   help="data type")
    parser.add_argument(
        "--model_version", 
        type=str, 
        choices=["v1", "v2"],  
        default="v1",
        help="v1, v2-simple model"
    )

    # 3. 解析参数
    args = parser.parse_args()
    return args

# def restet_global_set(args):
#     with open("global_set.json", "r") as f:
#         config = json.load(f)
#     config['data_root'] = args.data_root
#     with open("global_set.json", "w") as f:
#         json.dump(config, f, indent=4)

def train_vqvae(model, train_loader, epoch, optimizer, device, data_variance, dataset_type="MNIST"):
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(epoch):
        s_time = time.time()
        if dataset_type == "cifar10":
            (data, _) = next(iter(train_loader))  # 由于样本量太大，此处仅训练第一个batch
            data = data.to(device)
            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = model(data)  # vq 损失
            recon_error = F.mse_loss(data_recon, data) / data_variance  # 重构损失
            loss = recon_error + vq_loss  # 总损失
            loss.backward()
            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
        else:
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

def plot_train_loss(train_res_recon_error, train_res_perplexity, dataset_type="MNIST"):
    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1,2,2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')
    plt.savefig(f"{cur_dir}/training_loss_comparison_{dataset_type}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {cur_dir}/training_loss_comparison_{dataset_type}.png")
    
def view_restruct_v1(model, data_loader, dataset_type="MNIST"):
    model.eval()
    (valid_originals, _) = next(iter(data_loader))
    valid_originals = valid_originals.to(device)  # [N, C, H, W]
    
    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)  # [N, C, H, W]
    print("valid_reconstructions shape:", valid_reconstructions.shape)
    print("valid_quantize shape:", valid_quantize.shape)
    print("valid_originals shape:", valid_originals.shape)
    
    # make_grid: 合并多个图片为一个图片,input: [N, C, H, W]
    # output: [C, H', W']
    # H': (n1, n2) * H
    # W': (n1, n2) * W
    reconstructions_img = make_grid(valid_reconstructions)+0.5  # [C, H', W']
    reconstructions_img = reconstructions_img.permute(1, 2, 0).cpu() # [H', W, C]
    show_image_by_array(reconstructions_img, title="reconstruction", save_path=f"{cur_dir}/vqvae_reconstruct_{dataset_type}.png")
    
    original_img = make_grid(valid_originals)+0.5  # [C, H', W']
    original_img = original_img.permute(1, 2, 0).cpu() # [H', W, C]
    show_image_by_array(original_img, title="original", save_path=f"{cur_dir}/vqvae_original_{dataset_type}.png")
    plt.close()


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


def reconstruct(model, x, device, dataset_type='mnist'):
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
    cv2.imwrite(f'{cur_dir}/vqvae_reconstruct_{dataset_type}.jpg', x_cat)


if __name__ == '__main__':
    args = parse_args()
    # =========================================================================
    # 超参数设置
    # =========================================================================
    batch_size = 256
    epoch = 1000  # 15000
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_version = args.model_version
    data_type = args.data_type.lower().strip().replace("-", "").replace("_", "")  # cifar-10 -> cifar10
    input_channels = 1 if data_type == "mnist" else 3
    output_channels = 1 if data_type == "mnist" else 3
    
    print("\n" + "=" * 60)
    print("Running on %s" % device)
    print("Training VQVAE")
    print(f"model_version: {model_version}")
    print("=" * 60)
    
    
    
    if model_version == 'v1':
        train_loader, test_loader, data_variance = get_dataloader(batch_size, data_root=args.data_root, data_type=data_type)
        model = VQVAEModel(input_channels, output_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
        train_res_recon_error, train_res_perplexity = train_vqvae(model, train_loader, epoch, optimizer, device, data_variance, dataset_type=data_type)
        plot_train_loss(train_res_recon_error, train_res_perplexity, dataset_type=data_type)
        view_restruct_v1(model, test_loader, dataset_type=data_type)
    else:
        train_loader, _, = get_dataloader(batch_size, data_root=args.data_root, data_type=data_type)
        model = VQVAEModelV2(input_channels, 32, 32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
        train_vqvae_2(model, train_loader, device, optimizer, epoch)
        
        img, _ = next(iter(train_loader))
        img = img.to(device)
        print("img.shape:", img.shape)
        reconstruct(model, img, device, dataset_type=data_type)

    
    
    

    # train_loader, test_loader, data_variance = get_cifar10_dataloader(batch_size, data_path='/Users/bytedance/Downloads/dataset_for_dl/cifar-10')
    # train_vqvae(model, train_loader, epoch, optimizer, device, data_variance)
