"""
DIN 数据输入迭代器 (PyTorch 版)
与 input.py (TF1版) 和 din_tf2/input.py 逻辑完全一致。
纯 numpy，不依赖 TensorFlow 或 PyTorch，直接复用 din_tf2 版本即可。
"""

import numpy as np


class DataInput:
    """
    训练数据迭代器：对 train_set 按 batch_size 切分。
    train_set 每条记录: (user_id, hist_item_list, target_item_id, label)
    返回: (step_id, batch_tuple)
        batch_tuple = (u, i, y, hist_i, sl)
            u:      [B]     user ids
            i:      [B]     target item ids (正/负样本混排)
            y:      [B]     labels (0/1)
            hist_i: [B, T]  user 历史 item id，右侧 padding 0
            sl:     [B]     每个 user 的历史长度
    """

    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size:
                       min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[1]))
        max_sl = max(sl) if sl else 1

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, y, hist_i, sl)


class DataInputTest:
    """
    评估/测试数据迭代器：对 test_set 按 batch_size 切分。
    test_set 每条记录: (user_id, hist_item_list, (pos_item_id, neg_item_id))
    返回: (step_id, batch_tuple)
        batch_tuple = (u, i, j, hist_i, sl)
            u:      [B]     user ids
            i:      [B]     正样本 item ids
            j:      [B]     负样本 item ids
            hist_i: [B, T]  user 历史 item id
            sl:     [B]     每个 user 的历史长度
    """

    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size:
                       min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        u, i, j, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
        max_sl = max(sl) if sl else 1

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, j, hist_i, sl)
