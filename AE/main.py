
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from my_utils import plt
from data_utils import get_mnist_dataloader
from AE.model import AutoEncoder

def train_ae(model, train_loader, input_dim, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)

        optimizer.zero_grad()
        x_recon, _ = model(data)

        # 重建损失：MSE 或 BCE
        loss = F.mse_loss(x_recon, data, reduction='sum') / data.size(0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[AE] Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}")
    return avg_loss


def test_ae(model, test_loader, input_dim, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, input_dim).to(device)
            x_recon, _ = model(data)
            loss = F.mse_loss(x_recon, data, reduction='sum') / data.size(0)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"[AE] Test Loss: {avg_loss:.4f}")
    return avg_loss


# ============================================================
# 可视化函数
# ============================================================
def visualize_reconstruction(model, test_loader, input_dim, model_name, device, n=10):
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
    plt.savefig(f"{model_name}_reconstruction.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_reconstruction.png")


def visualize_latent_space(model, test_loader, input_dim, model_name, device):
    """可视化潜在空间（取前两个维度）"""
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.view(-1, input_dim).to(device)
            if model_name == "AE":
                _, z = model(data)
            else:
                mu, _ = model.encode(data)
                z = mu  # 使用均值作为表示
            latents.append(z.cpu())
            labels.append(label)

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(f'{model_name} - Latent Space Visualization (first 2 dims)')
    plt.savefig(f"{model_name}_latent_space.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_name}_latent_space.png")

if __name__ == '__main__':
    # =========================================================================
    # 超惨设置
    # =========================================================================
    BATCH_SIZE = 128
    EPOCHS = 3
    LEARNING_RATE = 1e-3
    LATENT_DIM = 32  # 潜在空间维度
    INPUT_DIM = 784  # 28x28
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print("Training Auto-Encoder (AE)")
    print("Running on %s" % DEVICE)
    print("=" * 60)
    
    ae_model = AutoEncoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=LEARNING_RATE)
    train_loader, test_loader = get_mnist_dataloader(BATCH_SIZE)

    ae_losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_ae(ae_model, train_loader, INPUT_DIM, ae_optimizer, epoch, DEVICE)
        ae_losses.append(loss)

    test_ae(ae_model, test_loader, INPUT_DIM, DEVICE)

    # AE 可视化
    visualize_reconstruction(ae_model, test_loader, "AE")
    visualize_latent_space(ae_model, test_loader, "AE")
