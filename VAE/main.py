import torch
from VAE.model import VAE, vae_loss_function
from my_utils import plt
from data_utils import get_mnist_dataloader
import numpy as np
import torch.optim as optim
import os


def train_vae(model, train_loader, input_dim, optimizer, epoch, device):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)

        optimizer.zero_grad()
        x_recon, mu, log_var = model(data)

        loss, recon_loss, kl_loss = vae_loss_function(
            x_recon, data, mu, log_var)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(train_loader)

    avg_total_loss = total_loss / n
    avg_recon_loss = total_recon / n
    avg_kl_loss = total_kl / n
    print(
        f"[VAE] Epoch {epoch:3d} | Loss: {avg_total_loss:.4f} | Recon: {avg_recon_loss:.4f} | KL: {avg_kl_loss:.4f}")

    return avg_total_loss, avg_recon_loss, avg_kl_loss


def test_vae(model, test_loader, input_dim, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, input_dim).to(device)
            x_recon, mu, log_var = model(data)
            loss, _, _ = vae_loss_function(x_recon, data, mu, log_var)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"[VAE] Test Loss: {avg_loss:.4f}")
    return avg_loss


# ============================================================
# 可视化函数
# ============================================================
def visualize_reconstruction(model, test_loader, input_dim, model_name, device, dir="./", n=10):
    """可视化原始图像和重建图像的对比"""
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:n].to(device)

    with torch.no_grad():
        data_flat = data.view(-1, input_dim)
        if model_name == "AE":
            recon, _ = model(data_flat)
        else:
            recon, _, _ = model(data_flat)

    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    for i in range(n):
        # 原始图像
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        # 重建图像
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.suptitle(f"{model_name} - Reconstruction Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{dir}/{model_name}_reconstruction.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dir}/{model_name}_reconstruction.png")


def visualize_latent_space(model, test_loader, input_dim, model_name, device, dir="./"):
    """可视化潜在空间（取前两个维度）"""
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for data, label in test_loader:  # 遍历所有测试集数据
            data = data.view(-1, input_dim).to(device)
            if model_name == "AE":
                _, z = model(data)
            else:
                mu, _ = model.encode(data)
                z = mu  # 使用均值作为表示
            latents.append(z.cpu())
            labels.append(label)

    latents = torch.cat(latents, dim=0).numpy()  # [1000, 32]
    labels = torch.cat(labels, dim=0).numpy()  # [1000, ]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(f'{model_name} - Latent Space Visualization (first 2 dims)')
    plt.savefig(f"{dir}/{model_name}_latent_space.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dir}/{model_name}_latent_space.png")


def generate_from_vae(model, latent_dim, device, dir="./", n=10):
    """从 VAE 的潜在空间随机采样生成新图像"""
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样
        z = torch.randn(n, latent_dim).to(device)
        generated = model.decode(z)

    fig, axes = plt.subplots(1, n, figsize=(15, 1.5))
    for i in range(n):
        axes[i].imshow(generated[i].cpu().view(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.suptitle("VAE - Generated Samples (from random z ~ N(0,I))", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{dir}/VAE_generated_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dir}/VAE_generated_samples.png")


def visualize_vae_manifold(model, latent_dim, device, dir="./", n=20, latent_range=3):
    """
    在 2D 潜在空间上均匀采样，生成数字流形图
    （仅适用于 latent_dim=2 的情况，这里取前两维演示）
    """
    model.eval()

    # 在 [-latent_range, latent_range] 之间均匀采样
    grid_x = np.linspace(-latent_range, latent_range, n)
    grid_y = np.linspace(-latent_range, latent_range, n)

    canvas = np.zeros((28 * n, 28 * n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim).to(device)
                z[0, 0] = xi
                z[0, 1] = yi
                generated = model.decode(z)
                canvas[i*28:(i+1)*28, j*28:(j+1) * 28] = generated.cpu().view(28, 28).numpy()

    plt.figure(figsize=(12, 12))
    plt.imshow(canvas, cmap='gray')
    plt.title("VAE - Latent Space Manifold (varying dim 0 & 1)")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(f"{dir}/VAE_manifold.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dir}/VAE_manifold.png")


if __name__ == "__main__":
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    LATENT_DIM = 20  # 潜在空间维度
    INPUT_DIM = 784  # 28x28
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    print("=" * 60)
    print("Running on %s" % DEVICE)
    print("run vae training")
    print("=" * 60)

    train_loader, test_loader = get_mnist_dataloader(BATCH_SIZE)
    # ==================== 训练 VAE ====================
    print("\n" + "=" * 60)
    print("Training Variational Auto-Encoder (VAE)")
    print("=" * 60)

    vae_model = VAE(INPUT_DIM, LATENT_DIM).to(DEVICE)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

    vae_losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_vae(vae_model, train_loader, INPUT_DIM, vae_optimizer, epoch, DEVICE)
        vae_losses.append(loss)

    test_vae(vae_model, test_loader, INPUT_DIM, DEVICE)

    # VAE 可视化
    visualize_reconstruction(vae_model, test_loader, INPUT_DIM, "VAE", DEVICE, script_dir)
    visualize_latent_space(vae_model, test_loader, INPUT_DIM, "VAE", DEVICE, script_dir)
    generate_from_vae(vae_model, LATENT_DIM, DEVICE, script_dir, n=10)
    visualize_vae_manifold(vae_model, LATENT_DIM, DEVICE, script_dir, n=20, latent_range=3)

    # ==================== 训练曲线对比 ====================
    plt.figure(figsize=(10, 5))
    # plt.plot(range(1, EPOCHS+1), ae_losses, 'b-o', label='AE Loss (MSE)', markersize=4)
    plt.plot(range(1, EPOCHS+1), vae_losses, 'r-s', label='VAE Loss (ELBO)', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison: AE vs VAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{script_dir}/training_loss_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {script_dir}/training_loss_comparison.png")

    print("\n" + "=" * 60)
    print("All done! Check the generated PNG files for visualizations.")
    print("=" * 60)
