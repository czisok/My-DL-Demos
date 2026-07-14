import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
        向量量化器: 用于将输入向量量化为离散值，通过最小化与码本中最近邻的距离来实现。并返回离散值对应的码本emb
        commitment_cost: 用于平衡量化误差和码本更新的权重
        返回量化损失： || sg(z_e(x)) - z_q(x) ||^2 + \beta|| z_e((x) - sg(z_q(x)) ||^2
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        """
        此处传过来的inputs是encoder建模之后的结果，维度是[C, H, W], C与码本emb同维度(C=H), 把原来大尺寸的图像，缩小为h,w.
        量化以后，一个图片就被压缩(量化)为一个h,w的离散值。
        """
        # convert inputs from BCHW -> BHWC
        # contiguous -> 内存连续存储
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape  # [B, H, W, C]
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  # [, H]
        # Calculate distances
        """
        下面是求距离的常用方法(以下代码涉及张量广播): 
        aa = torch.sum(flat_input**2, dim=1, keepdim=True) : [B, H] -> [B, 1]
        bb = torch.sum(self._embedding.weight**2, dim=1)   : [N, H] -> [N]
        aa + bb: 1. bb扩维到 [1, N], 2. aa广播为[B, N] (第0维进行广播), bb广播为[B, N] (第1维进行广播)
        cc = 2 * torch.matmul(flat_input, self._embedding.weight.t()) : [B, H] x [N, H].t() -> [B, N]
        distances = aa + bb - cc -> [B, N] : B个H维的input跟N个H维embedding的距离 [b, n] 表示第b个input跟第n个embedding的距离
        """
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )
        # Encoding
        encoding_indices = torch.argmin(
            distances, dim=1).unsqueeze(1)  # [B*H*W, 1]
        # print(encoding_indices)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )  # [B*h*W, N_e]  N_e表示码本的大小(512)
        encodings.scatter_(
            1, encoding_indices, 1
        )  # # [B*h*W, N_e] , 也就是每行对应encoding_indices的列是1
        # Quantize and unflatten
        embs = torch.matmul(
            encodings, self._embedding.weight
        )  # [B*h*W, N_e] x [N_e, H] -> [B*h*W, H]
        # [B, h, w, H] 或 称为 [B, h, w, C], 即z_q
        quantized = embs.view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss  # z 和 z_q 损失

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(
            encodings, dim=0
        )  # [B*h*w, C] -> [C], 即计算码本每个index的分布频次

        """
    
        PPL : 衡量码本索引使用的均匀程度，反映码本利用率。熵达到最大值

        PPL = PPL = e^{-\sum_{i=1}^{k}{p(k)\times \log{p(k)}}} k是码本大小
        
        PPL ≈ K( 码本最优 , 利用率均衡 ):  \(\log_2 K\)，所有码元被均匀使用，每个索引出现概率接近 \(1/K\)。
        PPL ≪ K( 严重码本坍塌 Codebook Collapse)
        PPL = 1( 极端坍塌 )
        """
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )  # 困惑度

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()  # 正态分布初始化
        self._commitment_cost = commitment_cost

        self.register_buffer(
            "_ema_cluster_size", torch.zeros(num_embeddings)
        )  # 每个码字的EMA计数
        self._ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim)
        )  # EMA累积和 [N_e, H]
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  # [B*h*w, C]

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )  # [B, h, w, H] 或 称为 [B, h, w, C], 即z_q

        # Use EMA to update the embedding vectors
        if self.training:
            # EMA更新：每个码字被使用的次数（聚类大小 )
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size,
            # Laplace平滑：
            #    如果单纯只加epsilong,则总计数就变成n+K*epsilon, 分母变大，向量e_k回变小
            #    计数归一化，先计算每个idx的计数占总计数n的百分比（包含espilon )，然后再乘n变换回计数值
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(
                encodings.t(), flat_input
            )  # [B*h*W, N_e].t() x [B*h*w, C] -> [N_e, C] (512, 64)

            # EMA累积和 [N_e, H]
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    """
        残差卷积: (x -> relu -> conv -> relu -> conv) + x
    Args:
        in_channels (int): 输入通道数
        num_hiddens (int): 隐藏通道数
        num_residual_hiddens (int): 残差通道数
    """

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    """
        残差栈: 由多个残差卷积层组成
    """

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    """
        编码器: 由多个残差卷积层组成, x->conv1->relu->conv2->relu->conv3->residual_stack->z_e
    """

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    """
        解码器: 由多个残差卷积层组成, z_e->conv_trans1->relu->conv_trans2->relu->conv_trans3->x
    """

    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens//2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VQVAEModel(nn.Module):
    def __init__(self, input_dim, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(VQVAEModel, self).__init__()

        self._encoder = Encoder(input_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity
