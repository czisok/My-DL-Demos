import torch
import torch.nn.functional as F
import numpy as np
from VQVAE.model import VQVAEModel
import torch.optim as optim
from data_utils import get_cifar10_dataloader
def train_vqvae(model, train_loader, epoch, optimizer, device, data_variance):
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(epoch):
        (data, _) = next(iter(train_loader))  # 由于样本量太大，此处仅训练第一个batch
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
    return train_res_recon_error, train_res_perplexity

if __name__ == '__main__':
    # =========================================================================
    # 超参数设置
    # =========================================================================
    batch_size = 256
    epoch = 1000 #15000
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
    
    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    train_loader, test_loader, data_variance = get_cifar10_dataloader(batch_size)
    train_vqvae(model, train_loader, epoch, optimizer, device, data_variance)
    