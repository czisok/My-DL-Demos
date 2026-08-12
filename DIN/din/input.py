from __future__ import annotations
import pickle
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# ===================== 1. 自定义 AmazonReview Dataset =====================
class AmazonReviewDataset(Dataset):
    def __init__(self, data_path: str):
        with open('dataset.pkl', 'rb') as f:
            self.train_set = pickle.load(f)  # (uid, hist_i, iid, y)
            self.test_set = pickle.load(f)   # (uid, hist_i, (pos_iid, neg_iid))
            self.cate_list = pickle.load(f)  # cate_list[i] is the category of item i(iid)
            self.user_count, self.item_count, self.cate_count = pickle.load(f)

    def __len__(self):
        """返回数据集总样本数量，必须实现"""
        return len(self.train_set)

    def __getitem__(self, index: int) -> Dict:
        """
        根据索引取单条样本，dataloader 内部会调用
        index: 样本下标
        """
        uid, hist_i, iid, y = self.train_set[index]
        cate = self.cate_list[iid]
        return {
            "uid": uid,
            "hist_i": hist_i,
            "iid": iid,
            "cate": cate,
            "y": y,
        }
        


# ===================== 2. 构建DataLoader 封装函数 =====================
def build_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,          # 训练集开启打乱，验证集False
        num_workers=num_workers,  # linux建议大于0；windows设为0避免bug
        pin_memory=True,          # GPU训练加速
        drop_last=False           # 是否丢弃不足一个batch的样本
    )
    return loader

# ===================== 3. 主调用示例 =====================
if __name__ == "__main__":
    # 方式A：纯原始文本（简单基线模型）
    train_dataset = AmazonReviewDataset(data_path="train_reviews.jsonl")
    train_loader = build_dataloader(train_dataset, batch_size=16, shuffle=True)

    # 方式B：搭配BERT分词器（NLP标准方案，需要安装transformers）
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # train_dataset = AmazonReviewDataset(data_path="train_reviews.jsonl", tokenizer=tokenizer)
    # train_loader = build_dataloader(train_dataset, batch_size=16)

    # 遍历dataloader 测试输出
    print(f"数据集总样本数: {len(train_dataset)}")
    for batch_idx, batch_data in enumerate(train_loader):
        print("=" * 40)
        # 如果使用bert分词：
        # print(batch_data["input_ids"].shape)   # [batch_size, seq_len]
        # print(batch_data["label"])

        # 无分词器版本：
        print("batch文本样例:", batch_data["text"][0])
        print("batch标签:", batch_data["label"])

        if batch_idx >= 2:  # 只打印前3个batch测试
            break
