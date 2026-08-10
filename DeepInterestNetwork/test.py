"""
Amazon 2014 Electronics 数据集预处理脚本
======================================
功能：从原始 gzip 压缩的 JSON 数据构建 DIN 模型所需的训练集与测试集

数据流向：
    原始 JSON.gz → DataFrame → ID 映射 → 按用户分组 → 构建正负样本 → 序列化保存
"""

import ast
import gzip
import pickle
import random

import numpy as np
import pandas as pd

# ============================================================
# 随机种子固定，保证实验可复现
# ============================================================
random.seed(1234)
np.random.seed(1234)


# ============================================================
# 数据加载相关函数
# ============================================================

def parse(path):
    """
    逐行解析 gzip 压缩的 JSON 文件，生成器方式返回每条记录的字典

    Args:
        path: gzip 文件路径

    Yields:
        dict: 一条 JSON 记录解析后的字典
    """
    with gzip.open(path, 'rb') as g:
        for line in g:
            # 使用 ast.literal_eval 替代 eval，避免任意代码执行的安全风险
            yield ast.literal_eval(line)


def getDF(path):
    """
    将 gzip JSON 文件读取为 pandas DataFrame

    Args:
        path: gzip 文件路径

    Returns:
        pd.DataFrame: 完整的数据表
    """
    records = {}
    for idx, record in enumerate(parse(path)):
        records[idx] = record
    return pd.DataFrame.from_dict(records, orient='index')


# ============================================================
# ID 映射与数据预处理
# ============================================================

def build_map(df, col_name, col_new_name=None):
    """
    将指定列的离散值映射为从 0 开始的连续整数 ID，
    并原地替换 DataFrame 中该列的值。
    Args:
        df (_type_): _description_
        col_name (_type_): _description_
        col_new_name (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple: (映射字典 {原值: 新ID}, 排序后的原始值列表)
    """
    if col_new_name is None:
        col_new_name = col_name
    # 按字母/数字排序后分配 ID，保证映射的确定性
    unique_vals = sorted(df[col_name].unique().tolist())
    val_to_id = dict(zip(unique_vals, range(len(unique_vals))))
    df[col_new_name] = df[col_name].map(lambda x: val_to_id[x])
    return val_to_id, unique_vals


# ============================================================
# 训练/测试集构建
# ============================================================

def gen_neg_item(pos_list, item_count):
    """
    为一个正样本生成一个不在用户正样本列表中的负样本（随机采样商品）

    Args:
        pos_list: 用户的正样本商品 ID 列表
        item_count: 商品总数

    Returns:
        int: 负样本商品 ID
    """
    neg = pos_list[0]
    while neg in pos_list:
        neg = random.randint(0, item_count - 1)
    return neg


def build_train_test_sets(reviews_df, item_count):
    """
    按用户分组构建训练集和测试集：
    - 每个用户的行为序列中，前 N-1 个用于训练（留最后一个做测试）
    - 训练时每个正样本配一个负样本
    - 测试时为每个用户的最后一个正样本配一个负样本

    Args:
        reviews_df: 已按 reviewerID + unixReviewTime 排序的评论 DataFrame
        item_count: 商品总数

    Returns:
        tuple: (train_set, test_set)
            - train_set: [(user_id, hist_item_list, target_item, label), ...]
            - test_set:  [(user_id, hist_item_list, (pos_item, neg_item)), ...]
    """
    train_set = []
    test_set = []

    for reviewer_id, hist in reviews_df.groupby('reviewerID'):
        pos_list = hist['asin'].tolist()
        seq_len = len(pos_list)

        # 为每个正样本预先生成对应的负样本
        neg_list = [gen_neg_item(pos_list, item_count) for _ in range(seq_len)]

        # 滑动窗口构造序列：用前 i 个行为预测第 i 个
        for i in range(1, seq_len):
            user_hist = pos_list[:i]
            if i != seq_len - 1:
                # 训练样本：正样本(label=1) + 负样本(label=0)
                train_set.append((reviewer_id, user_hist, pos_list[i], 1))
                train_set.append((reviewer_id, user_hist, neg_list[i], 0))
            else:
                # 测试样本：最后一个行为作为 ground truth，正+负成对
                test_label = (pos_list[i], neg_list[i])
                test_set.append((reviewer_id, user_hist, test_label))

    return train_set, test_set


# ============================================================
# 主流程
# ============================================================

def main():
    """
    1. 加载原始gz文件
    2. meta中的categories字段，只保留最细粒度的最后一级分类
    3. 商品id (asin) 转为数字id、用户id (reviewerID为数字id、为数字id、分类id (categories) 转为数字id
    4. reviews_df 按 reviewerID + unixReviewTime 排序，确保每个用户的行为序列按时间顺序，只保留'reviewerID', 'asin', 'unixReviewTime'三列
    
    """
    # --------------------------------------------------------
    # 1. 加载原始数据
    # --------------------------------------------------------
    reviews_data_path = 'amazon_2014/reviews_Electronics_5.json.gz'
    meta_data_path = 'amazon_2014/meta_Electronics.json.gz'
    reviews_df = getDF(reviews_data_path)
    meta_df = getDF(meta_data_path)

    # categories 字段是嵌套列表，只保留最细粒度的最后一级分类
    meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

    # 过滤 meta：只保留在评论数据中出现过的商品
    meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
    meta_df = meta_df.reset_index(drop=True)

    # --------------------------------------------------------
    # 2. ID 映射（字符串 → 连续整数）
    # --------------------------------------------------------
    asin_map, asin_key = build_map(meta_df, 'asin')
    cate_map, cate_key = build_map(meta_df, 'categories')
    reviewer_map, reviewer_key = build_map(reviews_df, 'reviewerID')

    user_count = len(reviewer_map)
    item_count = len(asin_map)
    cate_count = len(cate_map)
    example_count = reviews_df.shape[0]

    print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))

    # --------------------------------------------------------
    # 3. 排序与列筛选
    # --------------------------------------------------------
    # meta_df 按 asin 升序排列，保证 cate_list 的索引与 asin 的 ID 一一对应
    meta_df = meta_df.sort_values('asin').reset_index(drop=True)

    # reviews_df 的 asin 也需要映射（注意：build_map 只修改了 meta_df 的 asin，reviews_df 的 asin 是字符串原值）
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
    # 按用户 + 时间排序，确保行为序列的时序正确性
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime']).reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

    # 构造 cate_list: 索引 i 对应商品 i 的类别 ID
    # 由于 meta_df 已按 asin 排序，第 i 行就是 asin=i 的商品信息
    cate_list = np.array(meta_df['categories'].tolist(), dtype=np.int32)

    print("cate_list len : %d\tmeta_df shape: %s\treviews_df shape: %s" %
          (len(cate_list), meta_df.shape, reviews_df.shape))

    # --------------------------------------------------------
    # 4. 构建训练集和测试集
    # --------------------------------------------------------
    train_set, test_set = build_train_test_sets(reviews_df, item_count)

    # 打乱顺序，防止训练时模式过拟合
    random.shuffle(train_set)
    random.shuffle(test_set)

    # 校验：每个用户恰好对应一条测试样本
    assert len(test_set) == user_count

    # --------------------------------------------------------
    # 5. 序列化保存
    # --------------------------------------------------------
    output_path = './amazon_2014/din_raw_data/electronics_dataset.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)

    print('Dataset saved to %s' % output_path)
    print('  train samples: %d' % len(train_set))
    print('  test  samples: %d' % len(test_set))


if __name__ == '__main__':
    main()
