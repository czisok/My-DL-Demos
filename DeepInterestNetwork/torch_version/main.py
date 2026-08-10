"""
DIN PyTorch 简化训练脚本 (MyBaseModel: 均值池化 + MLP)
=========================================================

功能：
  1) 加载 Amazon electronics 数据集 → train_set / test_set / cate_list / (user,item,cate)_count
  2) DINTrainDataset / DINTestDataset + collate_fn + build_train/test_dataloader
  3) MyBaseModel：Embedding → 历史均值池化 → BN + Linear(128) → concat([u,i,u*i]) → MLP(384→80→40→1)
  4) 训练一个 epoch，并输出 avg_loss + train AUC（训练 AUC 监控）
"""

from __future__ import annotations

import os
import pickle
import random
import math
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

best_auc = 0.0
# ====================================================================
# 1. Dataset 定义
# ====================================================================

class DINTrainDataset(Dataset):
    """
    训练集 Dataset。

    train_set 每条: (user_id, hist_item_list, target_item_id, label)
    __getitem__ 返回: (user_id, hist_item_list, target_item_id, label)
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int, int]:
        u, hist, item, label = self.samples[idx]
        return int(u), list(hist), int(item), int(label)


class DINTestDataset(Dataset):
    """
    评估/测试集 Dataset。

    test_set 每条: (user_id, hist_item_list, (pos_item_id, neg_item_id))
    __getitem__ 返回: (user_id, hist_item_list, pos_item_id, neg_item_id)
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, List[int], int, int]:
        u, hist, (pos, neg) = self.samples[idx]
        return int(u), list(hist), int(pos), int(neg)


# ====================================================================
# 2. collate_fn：对 batch 内变长 hist_list 做右侧 0 padding
# ====================================================================

def _hist_to_padded_tensor(hist_list_list: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """把变长的 list[list[int]] 转成 (hist_i [B, max_sl], sl [B]) 的 pair，train/test 共用。"""
    B = len(hist_list_list)
    sl_list = [len(h) for h in hist_list_list]
    max_sl = max(sl_list) if B > 0 else 0
    hist_np = np.zeros((B, max_sl), dtype=np.int64)
    for k, h in enumerate(hist_list_list):
        for l, v in enumerate(h):
            hist_np[k, l] = v
    return torch.from_numpy(hist_np), torch.tensor(sl_list, dtype=torch.int64)


def train_collate_fn(
    batch: List[Tuple[int, List[int], int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    训练 batch collate。
    返回 (u, i, y, hist_i, sl)：
        u:       [B]   int64
        i:       [B]   int64
        y:       [B]   float32
        hist_i:  [B, max_sl] int64  (右侧 0 padding)
        sl:      [B]   int64
    """
    u_list, hist_list_list, i_list, y_list = zip(*batch)
    hist_i, sl = _hist_to_padded_tensor(hist_list_list)
    u = torch.tensor(list(u_list), dtype=torch.int64)
    i = torch.tensor(list(i_list), dtype=torch.int64)
    y = torch.tensor(list(y_list), dtype=torch.float32)
    return u, i, y, hist_i, sl


def test_collate_fn(
    batch: List[Tuple[int, List[int], int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    评估 batch collate。
    返回 (u, i, j, hist_i, sl)。
    """
    u_list, hist_list_list, i_list, j_list = zip(*batch)
    hist_i, sl = _hist_to_padded_tensor(hist_list_list)
    u = torch.tensor(list(u_list), dtype=torch.int64)
    i = torch.tensor(list(i_list), dtype=torch.int64)
    j = torch.tensor(list(j_list), dtype=torch.int64)
    return u, i, j, hist_i, sl


# ====================================================================
# 3. DataLoader 工厂函数
# ====================================================================

def build_train_dataloader(
    train_set,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 1234,
) -> DataLoader:
    ds = DINTrainDataset(train_set)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        drop_last=drop_last,
        generator=g,
        pin_memory=False,
    )


def build_test_dataloader(
    test_set,
    batch_size: int = 512,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 1234,
) -> DataLoader:
    ds = DINTestDataset(test_set)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=test_collate_fn,
        drop_last=drop_last,
        generator=g,
        pin_memory=False,
    )


# ====================================================================
# 4. 数据加载封装
# ====================================================================

DATASET_PATH = (
    '/home/zhangbo.999/jupyter_workspace/dataset/amazon_review_data/'
    'amazon_2014/din_raw_data/electronics_dataset.pkl'
)


def load_dataset(path: str = DATASET_PATH):
    """
    加载 DIN 预处理后的 pickle 数据集。
    返回: (train_set, test_set, cate_list, (user_count, item_count, cate_count))
    """
    with open(path, 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)
    return train_set, test_set, cate_list, (user_count, item_count, cate_count)


def sequence_mask(
    lengths: torch.Tensor,
    maxlen: int | None = None,
    dtype: torch.dtype = torch.bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    等价于 TensorFlow 里的 tf.sequence_mask。

    Args:
        lengths: [B]  int64  每条样本的有效长度（>= 0）
        maxlen:  int/None  如果为 None 则取 lengths.max()；否则按给定长度构造 mask
        dtype:   输出 dtype，默认 bool （True=有效位置，False=padding 位置）
                 也可以传 torch.float32 得到 (1.0/0.0) 的数值 mask，可直接乘到张量上
        device:  输出 device；传 None 时使用 lengths 的 device

    Returns:
        mask:  [B, maxlen]
            mask[b, t] = True (或 1.0) 当且仅当 t < lengths[b]
            mask[b, t] = False(或 0.0) 当 t >= lengths[b]

    Examples:
        >>> sl = torch.tensor([2, 3, 1], dtype=torch.long)
        >>> sequence_mask(sl, maxlen=5)
        tensor([[ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [ True, False, False, False, False]])
    """
    if device is None:
        device = lengths.device
    if maxlen is None:
        maxlen = int(lengths.max().cpu().item()) if lengths.numel() > 0 else 0

    # arange([0,1,2,...,maxlen-1]) 扩成 [1, maxlen]，lengths 扩成 [B,1] → 逐元素比较
    idx = torch.arange(maxlen, device=device, dtype=lengths.dtype).unsqueeze(0)   # [1, M]
    l = lengths.to(device=device).unsqueeze(1)                                     # [B, 1]
    mask_bool = idx < l                                                             # [B, M]

    if dtype == torch.bool:
        return mask_bool
    return mask_bool.to(dtype)


# ====================================================================
# 5. 模型定义 (MyBaseModel: 均值池化版的基础 DIN 模型，不含 Attention)
# ====================================================================

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




# ====================================================================
# 6. 训练 & AUC 计算工具
# ====================================================================

def clip_by_global_norm_tf_(params, clip_norm: float, norm_type: float = 2.0) -> float:
    """TF 等价的 clip_by_global_norm：同时支持 dense 与 sparse_coo 梯度，对齐 TF IndexedSlices 的聚合方式。

    - dense 梯度：对整个张量展平计算 L2 范数（与 TF dense 相同）。
    - sparse 梯度（sparse_coo）：仅对 coalesce 后的 values() 计算 L2 范数，与 TF IndexedSlices
      只对 indices 对应的 dense 切片聚合 global_norm 的行为严格一致。
    - 缩放规则：scale = clip_norm / max(global_norm, clip_norm)；global_norm <= clip_norm 时不做任何修改。

    Args:
        params: Iterable[torch.nn.Parameter]，通常传 model.parameters()
        clip_norm: float，裁剪阈值（与 tf.clip_by_global_norm(t_list, clip_norm) 一致）
        norm_type: float，范数阶数，默认 2（L2 global norm）

    Returns:
        global_norm: float，裁剪前的总体范数（用于日志打印 debug）
    """
    assert norm_type == 2.0, f'当前仅实现了 L2 global norm，得到 norm_type={norm_type}'
    if clip_norm <= 0:
        return 0.0

    total_sq = 0.0
    param_grad_pairs: List = []

    # ---------- Phase 1：聚合 global_norm ----------
    for p in params:
        g = p.grad
        if g is None:
            continue
        if g.layout == torch.sparse_coo:
            # 对齐 TF IndexedSlices：只对 values()（被 lookup 到的行对应的梯度切片）参与范数
            gc = g.coalesce()
            v = gc.values().detach()
            norm_v = torch.linalg.vector_norm(v, ord=2).item()
            total_sq += norm_v * norm_v
            param_grad_pairs.append((p, 'sparse', None))
        else:
            gd = g.detach()
            norm_d = torch.linalg.vector_norm(gd.reshape(-1), ord=2).item()
            total_sq += norm_d * norm_d
            param_grad_pairs.append((p, 'dense', None))

    global_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
    if global_norm <= clip_norm:
        return float(global_norm)

    scale = clip_norm / global_norm

    # ---------- Phase 2：按比例原地缩放 .grad ----------
    # 注意：sparse param 的 grad 不要直接 inplace mul，先 detach values 再写回避免 autograd 问题
    with torch.no_grad():
        for p, kind, _ in param_grad_pairs:
            g = p.grad
            if g is None:
                continue
            if kind == 'sparse':
                gc = g.coalesce()
                new_values = gc.values().mul_(scale)
                new_grad = torch.sparse_coo_tensor(
                    indices=gc.indices(),
                    values=new_values,
                    size=gc.size(),
                    dtype=gc.dtype,
                    device=gc.device,
                ).coalesce()
                p.grad = new_grad
            else:
                g.mul_(scale)

    return float(global_norm)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """安全计算 AUC：当只有 1 个类别时返回 NaN，避免 sklearn UndefinedMetricWarning。"""
    n_pos = int(np.sum(y_true > 0.5))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    print_every: int = 500,
) -> Tuple[float, float, int, int]:
    """
    训练一个 epoch。

    Args:
        model:       MyBaseModel
        loader:      build_train_dataloader 返回的训练 DataLoader
        optimizer:   优化器（本脚本用 Adam）
        device:      torch.device
        print_every: 每多少个 batch 打印一次中间训练指标

    Returns:
        (avg_loss, train_auc, total_samples, n_batches)
          avg_loss    : 每个样本的平均损失（reduction=mean 后再按 batch_size 加权）
          train_auc   : 训练集 AUC（基于每个 batch 累积的 y_true / y_score）
          total_samples / n_batches
    """
    model.train()

    total_loss_sum = 0.0       # Σ (loss_i * B_i)   (加权后总损失，用于算 avg_loss)
    n_batches = 0
    total_samples = 0

    # 用于 epoch 结束时算 AUC：累积全部样本的 y_true 和 sigmoid(logit)
    all_y_true: List[np.ndarray] = []
    all_y_score: List[np.ndarray] = []

    t0 = time.time()
    for step_idx, batch in enumerate(loader, start=1):
        u, i, y, hist_i, sl = [t.to(device, non_blocking=False) for t in batch]
        B_local = int(y.numel())  # y.numel() 是指y的数量，也是batch_size

        optimizer.zero_grad(set_to_none=True)
        # forward 只传 4 元组 (u, i, hist_i, sl)；y 留在外层算 loss
        logit = model((u, i, hist_i, sl)).squeeze(-1)                    # [B]
        loss = F.binary_cross_entropy_with_logits(
            input=logit, target=y, reduction='mean',
        )
        loss.backward()
        optimizer.step()

        # 指标累积
        loss_val = float(loss.detach().cpu().item())
        total_loss_sum += loss_val * B_local
        total_samples += B_local
        n_batches += 1

        with torch.no_grad():
            score = torch.sigmoid(logit).cpu().numpy()
            y_np = y.cpu().numpy()
        all_y_true.append(y_np)
        all_y_score.append(score)

        # 每 N 步打印中间指标
        if print_every > 0 and (step_idx % print_every == 0):
            elapsed = time.time() - t0
            samples_per_sec = (print_every * B_local) / max(elapsed, 1e-6)
            partial_y = np.concatenate(all_y_true[-print_every:])
            partial_s = np.concatenate(all_y_score[-print_every:])
            partial_auc = _safe_roc_auc(partial_y, partial_s)
            print(
                f'  [train] step={step_idx:<6d}  '
                f'loss={loss_val:.4f}  '
                f'partial_auc={partial_auc:.4f}  '
                f'speed={samples_per_sec:.1f} samples/s'
            )
            t0 = time.time()

    # 汇总 epoch 级指标
    avg_loss = total_loss_sum / max(total_samples, 1)
    y_all = np.concatenate(all_y_true) if all_y_true else np.array([], dtype=np.float32)
    s_all = np.concatenate(all_y_score) if all_y_score else np.array([], dtype=np.float32)
    train_auc = _safe_roc_auc(y_all, s_all)
    return avg_loss, train_auc, total_samples, n_batches

def _auc_arr(score):
    score_p = score[:, 0]  # score of positive item
    score_n = score[:, 1]  # score of negative item
    # print "============== p ============="
    # print score_p
    # print "============== n ============="
    # print score_n
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])  # noclick, click, score
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    """
        fp2: count of noclick
        tp2: count of click
        fp2 - fp1: 本轮 负样本增量
        tp2 + tp1: 上一轮正样本总量 + 本轮正样本总量 ？ 啥意思？
    """
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc 有正样本且也有负样本
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _eval(model, loader, device):
    model.eval()
    auc_sum = 0.0
    total_samples = 0
    score_arr = []
    eval_loss_sum = 0.0
    eval_loss_samples = 0

    with torch.no_grad():
        for x in loader:
            uid, iid_pos, iid_neg, hist_i, sl = x
            uid = uid.to(device)
            iid_pos = iid_pos.to(device)
            iid_neg = iid_neg.to(device)
            hist_i = hist_i.to(device)
            sl = sl.to(device)

            B_local = int(uid.numel())
            # 修复#2：把 pos/neg 合并成 [2B] 样本一次性 forward，保证 BN（hist_bn / i_bn）、
            #        Embedding lookup 的归一化统计量对 pos/neg 完全对称，与 TF 一次 sess.run
            #        同时算 i/j 的实现严格等价，消除两次 forward 造成的 u_emb 数值不对称。
            uid_2b    = torch.cat([uid,     uid],     dim=0)   # [2B]
            iid_2b    = torch.cat([iid_pos, iid_neg], dim=0)   # [2B]
            hist_i_2b = torch.cat([hist_i,  hist_i],  dim=0)   # [2B, T]
            sl_2b     = torch.cat([sl,      sl],      dim=0)   # [2B]
            logit_2b = model((uid_2b, iid_2b, hist_i_2b, sl_2b)).squeeze(-1)   # [2B]
            pos_logit = logit_2b[:B_local]
            neg_logit = logit_2b[B_local:]

            # eval loss：pos 作为正样本（y=1）、neg 作为负样本（y=0），按 sum 累计再全局平均
            pos_y = torch.ones(B_local, dtype=pos_logit.dtype, device=device)
            neg_y = torch.zeros(B_local, dtype=neg_logit.dtype, device=device)
            loss_sum_pos = F.binary_cross_entropy_with_logits(pos_logit, pos_y, reduction='sum')
            loss_sum_neg = F.binary_cross_entropy_with_logits(neg_logit, neg_y, reduction='sum')
            eval_loss_sum += float(loss_sum_pos.detach().cpu().item()) + float(loss_sum_neg.detach().cpu().item())
            eval_loss_samples += B_local * 2

            pos_score = torch.sigmoid(pos_logit)
            neg_score = torch.sigmoid(neg_logit)
            p_and_n = torch.concat([pos_score.unsqueeze(-1), neg_score.unsqueeze(-1)], dim=1)   # [B, 2]

            s = pos_logit - neg_logit
            u_auc = torch.mean((s > 0).float()).cpu()
            auc_sum += float(u_auc) * B_local
            total_samples += B_local
            score_arr += _auc_arr(p_and_n)

    test_gauc = auc_sum / total_samples if total_samples > 0 else 0.0
    Auc = calc_auc(score_arr)
    eval_loss_avg = eval_loss_sum / max(eval_loss_samples, 1)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc

    return test_gauc, Auc, eval_loss_avg
        
        
# ====================================================================
# 7. 入口：加载数据 → 构建模型 → 训练一个 epoch
# ====================================================================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, global_step: int, path: str):
    """保存模型 + 优化器 + 训练进度 checkpoint（等价于原 tf.train.Saver.save）。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
        },
        path,
    )


def main(
    num_epochs: int = 50,
    train_batch_size: int = 32,
    test_batch_size: int = 512,
    base_lr: float = 1.0,            # 对齐 base_model: GradientDescent lr=1.0
    eval_every_steps: int = 1000,    # 对齐 base_model: global_step % 1000 == 0 评估
    lr_decay_step: int = 336000,     # 对齐 base_model: 第 336k step 时 lr *= 0.1
    save_path: str = 'save_path/ckpt.pt',
    seed: int = 1234,
    print_every: int = 500,
    grad_clip_norm: float = 5.0,     # 对齐 base_model: clip_by_global_norm(5)
):
    """
    多 epoch 训练主函数（对齐 base_model/train.py 的调度逻辑）。

    调度逻辑对比 base_model：
      • num_epochs = 50 (默认)
      • 训练前跑一次基线 _eval
      • 每次 global_step % 1000 == 0 → _eval，输出 Epoch / Global_step / Train_loss / Eval_GAUC / Eval_AUC
      • global_step == lr_decay_step → lr = base_lr * 0.1
      • 若 test_gauc > best_auc → torch.save ckpt
      • 每 epoch 结束输出 summary（samples/batches/elapsed/avg_loss/train_auc/best_gauc）
    """
    set_seed(seed)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    print(f'Loading dataset from: {DATASET_PATH}')
    if not os.path.exists(DATASET_PATH):
        print(f'[WARN] 数据集路径不存在：{DATASET_PATH}', file=sys.stderr)
        print('       请修改 DATASET_PATH 常量或通过环境变量传入正确路径。', file=sys.stderr)
        sys.exit(1)

    train_set, test_set, cate_list, (user_count, item_count, cate_count) = load_dataset()
    print(
        f'Loaded: train={len(train_set):,}  test={len(test_set):,}  '
        f'users={user_count:,}  items={item_count:,}  cates={cate_count:,}'
    )

    # 修复#4：DataLoader 不做内部 shuffle，由外层用 Python random.shuffle(train_set) 控制，
    #        对齐 base_model/train.py：random.seed(1234) 全局初始化一次 → 每个 epoch random.shuffle(train_set)
    #        → DataInput 按顺序切片生成 batch。这样 SGD 的样本出现序列与 TF 严格一致，避免 RNG 系统差异导致漂移。
    train_loader = build_train_dataloader(train_set, batch_size=train_batch_size, shuffle=False, seed=seed)
    test_loader = build_test_dataloader(test_set, batch_size=test_batch_size, shuffle=False, seed=seed)

    # ------------------------------------------------------------------
    # 模型 & 优化器（对齐 base_model: SGD + clip_by_global_norm(5)）
    # ------------------------------------------------------------------
    hidden_units = 128
    model = MyBaseModel(
        u_cnt=user_count,
        i_cnt=item_count,
        cate_cnt=cate_count,
        cate_list=cate_list,
        hidden_units=hidden_units,
    ).to(device)

    # 注意：如果之前用 Adam，这里改为 SGD 是为了对齐 base_model；保留 Adam 注释方便切换
    optimizer = optim.SGD(model.parameters(), lr=base_lr)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {total_params:,}')
    print(f'Optimizer : {type(optimizer).__name__}  base_lr={base_lr}  clip_norm={grad_clip_norm}')

    # ------------------------------------------------------------------
    # 全局训练状态
    # ------------------------------------------------------------------
    global best_auc
    best_auc = 0.0
    global_step = 0                # 对齐 base_model 的 global_step.eval()
    global_epoch_step = 0          # 对齐 base_model 的 global_epoch_step.eval()
    current_lr = base_lr

    # ------------------------------------------------------------------
    # 训练/评估曲线 history（两套独立 x 轴：train_steps 与 eval_steps 解耦，避免凑长度破坏语义）
    # ------------------------------------------------------------------
    train_steps: List[int] = []
    train_loss_history: List[float] = []
    train_auc_history: List[float] = []

    eval_steps: List[int] = []
    eval_loss_history: List[float] = []
    eval_auc_history: List[float] = []
    eval_gauc_history: List[float] = []

    # ------------------------------------------------------------------
    # 基线评估（训练前跑一次，看随机权重的基准）
    # ------------------------------------------------------------------
    print('=' * 70)
    t0 = time.time()
    test_gauc, test_auc, test_loss_avg = _eval(model, test_loader, device)
    elapsed = time.time() - t0
    eval_steps.append(int(global_step))
    eval_loss_history.append(float(test_loss_avg))
    eval_gauc_history.append(float(test_gauc))
    if test_auc is not None:
        eval_auc_history.append(float(test_auc))
    else:
        eval_auc_history.append(float('nan'))
    print(
        f'[Baseline Eval]  Global_step={global_step:<8d}  '
        f'Eval_Loss={test_loss_avg:.6f}  '
        f'Eval_GAUC={test_gauc:.4f}  Eval_AUC={"" if test_auc is None else f"{test_auc:.4f}"}  '
        f'Time={elapsed:,.1f}s'
    )
    print('=' * 70)

    # ------------------------------------------------------------------
    # 多 Epoch 训练循环
    # ------------------------------------------------------------------
    start_time = time.time()

    for epoch in range(num_epochs):
        # 修复#4：每个 epoch 用 Python random 原地打乱 train_set，再重建 shuffle=False 的 DataLoader
        #        → batch 顺序、batch 内样本组合与 TF DataInput 的切片逻辑严格等价。
        #        注意：random.shuffle 使用全局 state，由 set_seed(seed) 入口处唯一初始化，避免用 torch.Generator
        #        的独立 RNG 导致与 base_model 的 50 轮 shuffle 序列不一致。
        random.shuffle(train_set)
        train_loader = build_train_dataloader(train_set, batch_size=train_batch_size, shuffle=False, seed=seed)

        print(f'\n--- Epoch {global_epoch_step} / {num_epochs}  START ---')
        epoch_t0 = time.time()

        # 本 epoch 级指标
        epoch_loss_sum = 0.0        # Σ(loss_i × B_i)，按样本加权
        epoch_samples = 0           # 本 epoch 训练样本数
        epoch_batches = 0           # 本 epoch batch 数
        epoch_y_true: List[np.ndarray] = []
        epoch_y_score: List[np.ndarray] = []
        # 每 eval_every_steps 重置一次的 loss 窗口（对应 base_model 里 loss_sum/1000 打印）
        window_loss_sum = 0.0
        window_samples = 0

        model.train()

        for step_idx, batch in enumerate(train_loader, start=1):
            u, i, y, hist_i, sl = [t.to(device, non_blocking=False) for t in batch]
            B_local = int(y.numel())

            optimizer.zero_grad(set_to_none=True)
            logit = model((u, i, hist_i, sl)).squeeze(-1)                       # [B]
            loss = F.binary_cross_entropy_with_logits(
                input=logit, target=y, reduction='mean',
            )
            loss.backward()

            # 对齐 base_model: tf.clip_by_global_norm(gradients, 5)
            # 注意：Embedding 使用 sparse=True（对齐 TF IndexedSlices），而 PyTorch 官方
            #       nn.utils.clip_grad_norm_ 走 _foreach_norm(SparseCUDA) 会抛 NotImplementedError
            #       （报错栈：aten::linalg_vector_norm 不支持 SparseCUDA backend），
            #       因此改为手动实现 clip_by_global_norm_tf_：sparse 只对 coalesce 的 values()
            #       参与范数，严格等价于 TF 对 IndexedSlices 的 global_norm 聚合方式。
            if grad_clip_norm > 0:
                _gn = clip_by_global_norm_tf_(model.parameters(), clip_norm=grad_clip_norm)

            optimizer.step()
            global_step += 1

            # --- 指标累积 ------------------------------------------------
            loss_val = float(loss.detach().cpu().item())
            epoch_loss_sum += loss_val * B_local
            epoch_samples += B_local
            epoch_batches += 1
            window_loss_sum += loss_val * B_local
            window_samples += B_local

            with torch.no_grad():
                score = torch.sigmoid(logit).cpu().numpy()
                y_np = y.cpu().numpy()
            epoch_y_true.append(y_np)
            epoch_y_score.append(score)

            # --- 每 N 步的进度日志 ------------------------------------
            if print_every > 0 and (step_idx % print_every == 0):
                elapsed = time.time() - epoch_t0
                speed = (window_samples or 1) / max(elapsed, 1e-6)
                # 本窗口内的 loss 平均 + AUC 近似
                win_y = np.concatenate(epoch_y_true[-print_every:])
                win_s = np.concatenate(epoch_y_score[-print_every:])
                win_auc = _safe_roc_auc(win_y, win_s)
                win_avg_loss = window_loss_sum / max(window_samples, 1)
                print(
                    f'  [train] Epoch={global_epoch_step:<3d}  step={step_idx:<6d}  '
                    f'global_step={global_step:<8d}  '
                    f'loss={win_avg_loss:.4f}  '
                    f'train_auc={win_auc:.4f}  '
                    f'speed={speed:,.0f} samples/s  lr={current_lr:.6f}'
                )

            # --- 对齐 base_model: 每 1000 个 global_step 评估 -------------
            if global_step % eval_every_steps == 0:
                eval_t0 = time.time()
                test_gauc, test_auc, test_loss_avg = _eval(model, test_loader, device)
                eval_elapsed = time.time() - eval_t0

                window_avg_loss = window_loss_sum / max(window_samples, 1)
                win_y = np.concatenate(epoch_y_true[-min(window_samples, print_every * 2):]) if epoch_y_true else np.array([], dtype=np.float32)
                win_s = np.concatenate(epoch_y_score[-min(window_samples, print_every * 2):]) if epoch_y_score else np.array([], dtype=np.float32)
                window_auc = _safe_roc_auc(win_y, win_s) if len(win_y) > 0 else float('nan')

                train_steps.append(int(global_step))
                train_loss_history.append(float(window_avg_loss))
                train_auc_history.append(float(window_auc) if not np.isnan(window_auc) else float('nan'))
                eval_steps.append(int(global_step))
                eval_loss_history.append(float(test_loss_avg))
                eval_gauc_history.append(float(test_gauc))
                if test_auc is not None:
                    eval_auc_history.append(float(test_auc))
                else:
                    eval_auc_history.append(float('nan'))

                auc_str = 'None' if test_auc is None else f'{test_auc:.4f}'
                print(
                    f'Epoch {global_epoch_step}  Global_step {global_step}\t'
                    f'Train_loss: {window_avg_loss:.4f}\t'
                    f'Eval_Loss: {test_loss_avg:.6f}\t'
                    f'Eval_GAUC: {test_gauc:.4f}\tEval_AUC: {auc_str}\t'
                    f'(Eval time: {eval_elapsed:,.1f}s)'
                )
                sys.stdout.flush()

                # 对齐 base_model: best_auc 提升 → 存 ckpt
                if test_gauc > best_auc:
                    best_auc = test_gauc
                    save_checkpoint(model, optimizer, global_epoch_step, global_step, save_path)
                    print(f'  [CKPT] saved to `{save_path}`  (best_gauc={best_auc:.6f})')

                # 重置窗口
                window_loss_sum = 0.0
                window_samples = 0

                model.train()  # _eval 内有 model.eval()

            # --- 对齐 base_model: 第 336000 步 → lr *= 0.1 ----------------
            if global_step == lr_decay_step:
                old_lr = current_lr
                current_lr = old_lr * 0.1
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
                print(f'[LR] decay @ global_step={global_step}: {old_lr} → {current_lr}')

        # --- Epoch 结束 summary --------------------------------------------
        epoch_elapsed = time.time() - epoch_t0
        avg_loss = epoch_loss_sum / max(epoch_samples, 1)
        ep_y = np.concatenate(epoch_y_true) if epoch_y_true else np.array([], dtype=np.float32)
        ep_s = np.concatenate(epoch_y_score) if epoch_y_score else np.array([], dtype=np.float32)
        train_auc = _safe_roc_auc(ep_y, ep_s)

        # 每个 epoch 结束也评估一次（确保 epoch 级的 best_auc 也能更新）
        eval_t0 = time.time()
        test_gauc, test_auc, test_loss_avg = _eval(model, test_loader, device)
        eval_elapsed = time.time() - eval_t0
        if test_gauc > best_auc:
            best_auc = test_gauc
            save_checkpoint(model, optimizer, global_epoch_step, global_step, save_path)
            print(f'  [CKPT] saved @ epoch {global_epoch_step} to `{save_path}`  (best_gauc={best_auc:.6f})')

        train_steps.append(int(global_step))
        train_loss_history.append(float(avg_loss))
        train_auc_history.append(float(train_auc) if not np.isnan(train_auc) else float('nan'))
        eval_steps.append(int(global_step))
        eval_loss_history.append(float(test_loss_avg))
        eval_gauc_history.append(float(test_gauc))
        if test_auc is not None:
            eval_auc_history.append(float(test_auc))
        else:
            eval_auc_history.append(float('nan'))

        auc_str = 'NaN (train set single class)' if np.isnan(train_auc) else f'{train_auc:.6f}'
        test_auc_str = 'None' if test_auc is None else f'{test_auc:.4f}'

        print('=' * 70)
        print(
            f'[Epoch {global_epoch_step} DONE]  samples={epoch_samples:,}  '
            f'batches={epoch_batches:,}  elapsed={epoch_elapsed:,.1f}s  '
            f'(total {time.time() - start_time:,.1f}s)'
        )
        print(f'  avg_loss  = {avg_loss:.6f}   (per-sample averaged loss)')
        print(f'  train_auc = {auc_str}')
        print(f'  eval_loss = {test_loss_avg:.6f}   (pos+neg BCEWithLogits averaged)')
        print(f'  test_gauc = {test_gauc:.4f}   test_auc = {test_auc_str}')
        print(f'  best_gauc = {best_auc:.6f}  (eval time {eval_elapsed:,.1f}s)')
        print('=' * 70)
        sys.stdout.flush()

        # 对齐 base_model.global_epoch_step_op (epoch + 1)
        global_epoch_step += 1

    # ------------------------------------------------------------------
    # 全部 Epoch 结束：绘制训练/评估曲线并保存（虚线，当前目录）
    # ------------------------------------------------------------------
    total_elapsed = time.time() - start_time
    print(f'\n[ALL DONE]  {num_epochs} epochs  total_time={total_elapsed:,.1f}s')
    print(f'best test_gauc: {best_auc}')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    if len(train_steps) > 0:
        ax.plot(train_steps, train_loss_history, linestyle='--', linewidth=1.8,
                label=f'Train Loss (start={train_loss_history[0]:.4f}, end={train_loss_history[-1]:.4f})')
    if len(eval_steps) > 0:
        ax.plot(eval_steps, eval_loss_history, linestyle='--', linewidth=1.8,
                label=f'Eval  Loss (start={eval_loss_history[0]:.4f}, end={eval_loss_history[-1]:.4f})')
    ax.set_xlabel('Global Step', fontsize=12)
    ax.set_ylabel('BCEWithLogits Loss (per-sample averaged)', fontsize=12)
    ax.set_title('Train / Eval Loss Curve', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)

    ax = axes[1]
    if len(train_steps) > 0:
        ax.plot(train_steps, train_auc_history, linestyle='--', linewidth=1.8,
                label='Train AUC')
    if len(eval_steps) > 0:
        ax.plot(eval_steps, eval_auc_history, linestyle='--', linewidth=1.8,
                label='Eval  AUC (pairwise)')
        ax.plot(eval_steps, eval_gauc_history, linestyle='--', linewidth=1.8,
                label='Eval  Group AUC')
    ax.set_xlabel('Global Step', fontsize=12)
    ax.set_ylabel('AUC / Group AUC', fontsize=12)
    ax.set_title('Train AUC / Eval AUC / Eval GAUC Curve', fontsize=14)
    ax.set_ylim([0.45, 1.0])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_curves_dashed.png')
    fig.savefig(save_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[FIG] 训练曲线（虚线）已保存至: {save_fig_path}')


if __name__ == '__main__':
    best_auc = 0.0
    main()
