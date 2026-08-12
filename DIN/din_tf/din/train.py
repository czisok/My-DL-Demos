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
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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


# ====================================================================
# 5. 模型定义 (MyBaseModel: 均值池化版的基础 DIN 模型，不含 Attention)
# ====================================================================

class MyBaseModel(nn.Module):
    """
    简化版 DIN：用**均值池化**代替注意力的基础 CTR 模型。

    前向链路：
        uid / iid / hist_i
          └─ Embedding
              ├─ i_emb = cat(iid_emb, cate_emb)      [B, H]  (H = hidden_units)
              ├─ h_emb = cat(hist_i_emb, hist_c_emb)  [B, T, H]
              │        └─ mean(dim=1) → bn → linear → u_emb [B, H]
              └─ mlp_input = cat([u_emb, i_emb, u_emb*i_emb]) → [B, 3H]
                        └─ bn → MLP(3H→80→40→1) → logit [B, 1]
    """

    def __init__(self, u_cnt: int, i_cnt: int, cate_cnt: int, cate_list, hidden_units: int):
        super().__init__()
        self.hidden_units = hidden_units

        self.uid_embs = nn.Embedding(u_cnt, hidden_units)
        self.iid_embs = nn.Embedding(i_cnt, hidden_units // 2)
        self.cate_embs = nn.Embedding(cate_cnt, hidden_units // 2)

        cate_tensor = torch.as_tensor(list(cate_list), dtype=torch.long)
        self.register_buffer('cate_list', cate_tensor, persistent=True)

        self.h_fc = nn.Linear(hidden_units, hidden_units)
        self.hist_bn = nn.BatchNorm1d(hidden_units)

        self.i_bn = nn.BatchNorm1d(hidden_units * 3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_units * 3, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1),
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: (uid, iid, hist_i, seq_len)   — y (label) 由外层 loss 计算，不传入 forward
        Returns:
            logit: [B, 1]   (未过 sigmoid，配合 BCEWithLogitsLoss)
        """
        uid, iid, hist_i, sl = x

        ic = self.cate_list[iid]
        hc = self.cate_list[hist_i]

        i_emb = torch.cat([self.iid_embs(iid), self.cate_embs(ic)], dim=1)     # [B, H]
        h_emb = torch.cat([self.iid_embs(hist_i), self.cate_embs(hc)], dim=-1)  # [B, T, H]

        h_emb = torch.mean(h_emb, dim=1)                                        # [B, H]
        h_emb = self.hist_bn(h_emb)
        u_emb = self.h_fc(h_emb)                                                # [B, H]

        mlp_input = torch.cat([u_emb, i_emb, u_emb * i_emb], dim=-1)             # [B, 3H]
        mlp_input = self.i_bn(mlp_input)
        return self.mlp(mlp_input)                                              # [B, 1]


# ====================================================================
# 6. 训练 & AUC 计算工具
# ====================================================================

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
        B_local = int(y.numel())

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
    total_samples = 0   # 累计：一共评估了多少条 user（= len(uid) 累加；保证最后一个不满 batch 也正确归一化）
    score_arr = []

    for x in loader:
        uid, iid_pos, iid_neg, hist_i, sl = x
        uid = uid.to(device)
        iid_pos = iid_pos.to(device)
        iid_neg = iid_neg.to(device)
        hist_i = hist_i.to(device)
        sl = sl.to(device)
        pos_data = (uid, iid_pos, hist_i, sl)
        neg_data = (uid, iid_neg, hist_i, sl)
        pos_logit = model(pos_data)  # [B, 1]
        neg_logit = model(neg_data)  # [B, 1]

        pos_score = torch.sigmoid(pos_logit)
        neg_score = torch.sigmoid(neg_logit)

        p_and_n = torch.concat([pos_score, neg_score], dim=1)

        s = pos_logit - neg_logit
        u_auc = torch.mean((s > 0).float()).cpu()
        B_local = int(uid.numel())
        auc_sum += float(u_auc) * B_local   # 累计 batch 内正>负的 user 数 = batch_acc × B_local
        total_samples += B_local
        score_arr += _auc_arr(p_and_n)

    # ⚠️ 原 bug: auc_sum / len(loader) —— len(loader) 是 batch 数（通常很小）
    #          而 auc_sum 已经是 acc × B 的累积（按 samples 计算），因此分母必须是总 sample 数
    #          否则当 len(loader) < total_samples 时，会得到 >>1 的荒谬值。
    test_gauc = auc_sum / total_samples if total_samples > 0 else 0.0
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        
    return test_gauc, Auc
        
        
# ====================================================================
# 7. 入口：加载数据 → 构建模型 → 训练一个 epoch
# ====================================================================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(1234)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ------------------------------------------------------------------
    # 数据 & 超参
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

    train_batch_size = 128
    # test_batch_size = 4   # 如需评估请后续接入
    hidden_units = 128
    LEARNING_RATE = 1e-3

    data_loader = build_train_dataloader(train_set, batch_size=train_batch_size)
    test_data_loader = build_test_dataloader(test_set, batch_size=64)

    # ------------------------------------------------------------------
    # 模型 & 优化器
    # ------------------------------------------------------------------
    model = MyBaseModel(
        u_cnt=user_count,
        i_cnt=item_count,
        cate_cnt=cate_count,
        cate_list=cate_list,
        hidden_units=hidden_units,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {total_params:,}')

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    test_gauc, test_auc = _eval(model, test_data_loader, device)
    print(f"test_gauc: {test_gauc}, test auc: {test_auc}")
    t_start = time.time()
    avg_loss, train_auc, total_samples, n_batches = train_one_epoch(
        model, data_loader, optimizer, device, print_every=500,
    )
    elapsed = time.time() - t_start
    test_gauc, test_auc = _eval(model, test_data_loader, device)
    print(f"test_gauc: {test_gauc}, test auc: {test_auc}")
    
    print('=' * 70)
    print(
        f'[Epoch Summary] samples={total_samples:,}  batches={n_batches:,}  '
        f'elapsed={elapsed:,.1f}s'
    )
    print(f'  avg_loss = {avg_loss:.6f}   (per-sample averaged loss)')
    print(f'  train_auc= {train_auc:.6f}' if not np.isnan(train_auc) else
          '  train_auc= NaN (train set only has one class; skip AUC)')
    print('=' * 70)

    # 保留原脚本末尾输出，方便与历史日志对比
    loss_total = avg_loss * total_samples
    print(f'batch total loss (legacy): {loss_total:.6f}')


if __name__ == '__main__':
    
    main()
