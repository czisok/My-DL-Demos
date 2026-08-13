from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import sequence_mask


class MyDINModel(nn.Module):
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

    def forward(self, u, i, hist_i, sl):
        pass
