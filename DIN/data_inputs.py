
from __future__ import annotations

import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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




def load_dataset(path: str):
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
