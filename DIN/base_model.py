from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import sequence_mask

class MyBaseModel(nn.Module):
    """
    pytorch 版 BaseModel：用**均值池化**代替注意力的基础 CTR 模型。

    前向链路：
        uid / iid / hist_i
          └─ Embedding
              ├─ i_emb = cat(iid_emb, cate_emb)      [B, H]  (H = hidden_units)
              ├─ h_emb = cat(hist_i_emb, hist_c_emb)  [B, T, H]
              │        └─ mean(dim=1) → bn → linear → u_emb [B, H]
              └─ mlp_input = cat([u_emb, i_emb]) → [B, 2H]  （对齐 base_model.Model，移除 u*i 交互项）
                        └─ bn → MLP(2H→80→40→1) → logit [B, 1]

    修复要点（对齐 TF 版 base_model.Model 实现）：
      (1) Embedding / Dense 层统一使用 Xavier(Glorot) 均匀初始化 + bias=0，适配 Sigmoid 激活；
      (2) BN 层关闭 running_stats（track_running_stats=False），训练/评估都用当前 batch 统计量，
          对齐 TF 版未手动运行 UPDATE_OPS 导致 train/eval BN 行为一致的效果，消除 train/eval gap；
      (3) MLP 输入维度由 3H 改为 2H，移除额外的 u*i 逐元素乘积交互项，参数规模与 TF 版完全对齐。
    """

    def __init__(self, u_cnt: int, i_cnt: int, cate_cnt: int, cate_list, hidden_units: int):
        super().__init__()
        self.hidden_units = hidden_units

        # 修复#3：Embedding 启用 sparse=True，对齐 TF embedding_lookup 返回 IndexedSlices（稀疏梯度）的形态，
        #        避免 clip_grad_norm_ 把整表 embedding 的 0 梯度行计入 global_norm，导致实际裁剪力度比 TF 重。
        self.uid_embs = nn.Embedding(u_cnt, hidden_units, sparse=True)
        self.iid_embs = nn.Embedding(i_cnt, hidden_units // 2, sparse=True)
        self.cate_embs = nn.Embedding(cate_cnt, hidden_units // 2, sparse=True)

        self.item_b = nn.Embedding(i_cnt, 1, sparse=True)
        nn.init.zeros_(self.item_b.weight)

        nn.init.xavier_uniform_(self.uid_embs.weight)
        nn.init.xavier_uniform_(self.iid_embs.weight)
        nn.init.xavier_uniform_(self.cate_embs.weight)

        cate_tensor = torch.as_tensor(list(cate_list), dtype=torch.long)
        self.register_buffer('cate_list', cate_tensor, persistent=True)

        self.h_fc = nn.Linear(hidden_units, hidden_units)
        nn.init.xavier_uniform_(self.h_fc.weight)
        nn.init.zeros_(self.h_fc.bias)

        # 修复#1：BN epsilon=1e-3 对齐 tf.layers.batch_normalization 默认值（PyTorch默认1e-5，差100倍）
        self.hist_bn = nn.BatchNorm1d(hidden_units, eps=1e-3, track_running_stats=False)

        self.i_bn = nn.BatchNorm1d(hidden_units * 2, eps=1e-3, track_running_stats=False)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_units * 2, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: (uid, iid, hist_i, seq_len)   — y (label) 由外层 loss 计算，不传入 forward
        Returns:
            logit: [B, 1]   (未过 sigmoid，配合 BCEWithLogitsLoss；logit = FCN_out + item_b[iid])
        """
        uid, iid, hist_i, sl = x

        ic = self.cate_list[iid]
        hc = self.cate_list[hist_i]

        i_emb = torch.cat([self.iid_embs(iid), self.cate_embs(ic)], dim=1)     # [B, H]
        h_emb = torch.cat([self.iid_embs(hist_i), self.cate_embs(hc)], dim=-1)  # [B, T, H]

        # ------------------------------------------------------------------
        # 显式 sequence_mask → sum → ÷ sl 的平均池化（对齐 base_model sum begin 段）
        #   为什么不能直接 mean(dim=1)？
        #   padding 位置 hist_i=0 对应的 embedding 不是 0 向量，mean 会把 id=0 的
        #   语义混入用户表示；显式 mask 只聚合真实历史长度的 T 个向量，再除以真实长度
        # ------------------------------------------------------------------
        B, T, H = h_emb.shape
        # 1) sequence_mask: [B, T]  True=有效历史，False=padding
        mask_bool = sequence_mask(lengths=sl, maxlen=T, dtype=torch.bool, device=h_emb.device)
        # 2) 扩展到 feature 维: [B, T, H]，转成与 h_emb 相同 dtype 可直接相乘
        mask = mask_bool.unsqueeze(-1).to(h_emb.dtype)                            # [B, T, 1]
        # 3) padding 位置置 0，再按时间维求和 → [B, H]
        h_masked = h_emb * mask                                                   # [B, T, H]
        h_sum = h_masked.sum(dim=1)                                               # [B, H]
        # 4) ÷ 真实 sl，clamp(min=1) 保证 sl=0 的样本除 0 安全
        sl_float = sl.to(h_emb.dtype).clamp(min=1.0).unsqueeze(-1)                # [B, 1]
        h_emb = h_sum / sl_float                                                  # [B, H]

        h_emb = self.hist_bn(h_emb)
        u_emb = self.h_fc(h_emb)                                                # [B, H]

        mlp_input = torch.cat([u_emb, i_emb], dim=-1)                            # [B, 2H]  （对齐 TF 版 base_model.Model：仅拼接用户表示和目标物品表示，不含 u*i 交互项）
        mlp_input = self.i_bn(mlp_input)
        fcn_out = self.mlp(mlp_input)                                           # [B, 1]
        # 加上当前 item 的可学习偏置（全局 item 流行度/偏置项，与 base_model 对齐）
        ib = self.item_b(iid)                                                   # [B, 1]
        return fcn_out + ib                                                     # [B, 1]
