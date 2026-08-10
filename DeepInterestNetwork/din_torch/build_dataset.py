"""
DIN 数据集构建 (PyTorch 版)
与 din/build_dataset.py 逻辑完全一致。纯 Python/numpy，无需改动 TF 相关代码。
生成 dataset.pkl：(train_set, test_set, cate_list, (user_count, item_count, cate_count))
"""

import random
import pickle

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
    pos_list = hist['asin'].tolist()

    def gen_neg_item():
        neg = pos_list[0]
        while neg in pos_list:
            neg = random.randint(0, item_count - 1)
        return neg

    neg_list = [gen_neg_item() for _ in range(len(pos_list))]

    for i in range(1, len(pos_list)):
        hist_seq = pos_list[:i]
        if i != len(pos_list) - 1:
            train_set.append((reviewerID, hist_seq, pos_list[i], 1))
            train_set.append((reviewerID, hist_seq, neg_list[i], 0))
        else:
            label = (pos_list[i], neg_list[i])
            test_set.append((reviewerID, hist_seq, label))

random.shuffle(train_set)
random.shuffle(test_set)

assert len(test_set) == user_count

with open('dataset.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
