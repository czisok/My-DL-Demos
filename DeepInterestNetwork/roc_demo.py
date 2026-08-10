"""
ROC 曲线绘制 Demo
=================
功能：
    1. 构造正负样本，mock 推荐模型的预测得分
    2. 手动遍历每个阈值，计算离散的 (FPR, TPR) 点
    3. 用梯形法计算 AUC
    4. 绘制 ROC 曲线并标注关键信息
"""

import random
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Mock 数据：构造正负样本和模型预测得分
# ============================================================

def mock_data(n_pos=200, n_neg=300, seed=42):
    """
    模拟推荐模型的输出：
    - 正样本（用户点击的商品）得分整体偏高
    - 负样本（用户未点击的商品）得分整体偏低
    - 两者有重叠，模拟真实场景的不确定性

    Args:
        n_pos: 正样本数量
        n_neg: 负样本数量
        seed: 随机种子

    Returns:
        tuple: (labels, scores)
            - labels: 0/1 数组，1=正样本，0=负样本
            - scores: 模型预测得分，范围 [0, 1]
    """
    random.seed(seed)
    np.random.seed(seed)

    # 正样本：均值 0.7，标准差 0.15 的正态分布（截断到 [0,1]）
    pos_scores = np.random.normal(loc=0.7, scale=0.15, size=n_pos)
    pos_scores = np.clip(pos_scores, 0.0, 1.0)

    # 负样本：均值 0.3，标准差 0.2 的正态分布（截断到 [0,1]）
    neg_scores = np.random.normal(loc=0.3, scale=0.2, size=n_neg)
    neg_scores = np.clip(neg_scores, 0.0, 1.0)

    # 合并并打标签
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # 打乱顺序（模拟真实数据分布）
    indices = np.random.permutation(len(labels))
    return labels[indices], scores[indices]


# ============================================================
# 2. 手动计算 TPR / FPR，得到 ROC 离散点
# ============================================================

def compute_roc_points(labels, scores):
    """
    手动计算 ROC 曲线上的离散点。
    思路：把所有样本的得分都当作阈值，每个阈值下统计 TP / FP / TN / FN，
    然后计算 TPR 和 FPR。

    Args:
        labels: 真实标签，0/1 数组
        scores: 模型预测得分

    Returns:
        tuple: (fpr_list, tpr_list, thresholds)
            - fpr_list: 每个阈值对应的 FPR
            - tpr_list: 每个阈值对应的 TPR
            - thresholds: 对应的阈值列表
    """
    P = int(labels.sum())           # 正样本总数
    N = int(len(labels) - P)        # 负样本总数

    # 按得分从高到低排序，这样遍历的时候相当于阈值从高到低移动
    sorted_indices = np.argsort(-scores)   # 降序排列的索引
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    fpr_list = [0.0]   # 初始点：阈值无穷大，没有样本被预测为正 → (0, 0)
    tpr_list = [0.0]
    thresholds = [float('inf')]

    tp = 0  # 当前被预测为正的正样本数
    fp = 0  # 当前被预测为正的负样本数

    i = 0
    n = len(sorted_labels)

    while i < n:
        # ---------- 处理得分相同的样本（同一阈值一次性处理） ----------
        current_score = sorted_scores[i]
        j = i
        while j < n and sorted_scores[j] == current_score:
            if sorted_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        # 计算当前阈值下的 TPR 和 FPR
        tpr = tp / P
        fpr = fp / N

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thresholds.append(current_score)

        i = j  # 跳到下一个不同的得分

    return fpr_list, tpr_list, thresholds


# ============================================================
# 3. 用梯形法手动计算 AUC
# ============================================================

def compute_auc(fpr_list, tpr_list):
    """
    梯形法计算 AUC：
    AUC = Σ (FPR[i] - FPR[i-1]) * (TPR[i-1] + TPR[i]) / 2

    Args:
        fpr_list: FPR 列表（按从小到大排）
        tpr_list: TPR 列表

    Returns:
        float: AUC 值
    """
    auc = 0.0
    for i in range(1, len(fpr_list)):
        delta_fpr = fpr_list[i] - fpr_list[i - 1]
        avg_tpr = (tpr_list[i - 1] + tpr_list[i]) / 2.0
        auc += delta_fpr * avg_tpr
    return auc


# ============================================================
# 4. 绘制 ROC 曲线
# ============================================================

def plot_roc(fpr_list, tpr_list, auc_val, n_pos, n_neg):
    """
    绘制 ROC 曲线，包含：
    - ROC 曲线（蓝色线）
    - 随机猜测基准线（红色虚线）
    - 离散的 (FPR, TPR) 点（绿色圆点）
    - AUC 标注
    """
    plt.figure(figsize=(8, 8))

    # ROC 曲线
    plt.plot(fpr_list, tpr_list, 'b-', linewidth=2, label='ROC Curve')

    # 离散点（每隔几个点画一个，避免太密）
    step = max(1, len(fpr_list) // 20)
    plt.scatter(fpr_list[::step], tpr_list[::step], c='green', s=30,
                zorder=5, label='Discrete Threshold Points')

    # 随机猜测基准线 y = x
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Guess (AUC=0.5)')

    # 坐标轴设置
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve\n(Positive samples: %d, Negative samples: %d)' % (n_pos, n_neg),
              fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 标注 AUC
    plt.text(0.6, 0.3, 'AUC = %.4f' % auc_val,
             fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 标注对角线上方的区域
    plt.fill_between(fpr_list, tpr_list, alpha=0.15, color='blue')

    plt.tight_layout()

    # 保存图片
    output_path = './roc_curve_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('ROC curve saved to %s' % output_path)

    plt.show()


# ============================================================
# 5. 主函数
# ============================================================

def main():
    # ---- Step 1: 生成 mock 数据 ----
    n_pos, n_neg = 200, 300
    labels, scores = mock_data(n_pos=n_pos, n_neg=n_neg, seed=42)
    print('=' * 60)
    print('Step 1: Mock 数据生成完成')
    print('  正样本数: %d' % n_pos)
    print('  负样本数: %d' % n_neg)
    print('  得分范围: [%.3f, %.3f]' % (scores.min(), scores.max()))
    print('  正样本平均得分: %.3f' % scores[labels == 1].mean())
    print('  负样本平均得分: %.3f' % scores[labels == 0].mean())

    # ---- Step 2: 计算 ROC 离散点 ----
    fpr_list, tpr_list, thresholds = compute_roc_points(labels, scores)
    print('\n' + '=' * 60)
    print('Step 2: ROC 离散点计算完成')
    print('  离散点数量: %d' % len(fpr_list))
    print('  前 5 个点 (FPR, TPR, threshold):')
    for i in range(min(5, len(fpr_list))):
        print('    (%.4f, %.4f, %s)' % (fpr_list[i], tpr_list[i],
                                        'inf' if thresholds[i] == float('inf') else '%.4f' % thresholds[i]))

    # ---- Step 3: 计算 AUC ----
    auc_val = compute_auc(fpr_list, tpr_list)
    print('\n' + '=' * 60)
    print('Step 3: AUC 计算完成')
    print('  AUC (手动梯形法) = %.6f' % auc_val)

    # 用 sklearn 验证（如果安装了）
    try:
        from sklearn.metrics import roc_auc_score
        auc_sklearn = roc_auc_score(labels, scores)
        print('  AUC (sklearn)      = %.6f' % auc_sklearn)
        print('  差值: %.6e' % abs(auc_val - auc_sklearn))
    except ImportError:
        print('  (未安装 sklearn，跳过对比验证)')

    # ---- Step 4: 绘制 ROC 曲线 ----
    print('\n' + '=' * 60)
    print('Step 4: 绘制 ROC 曲线...')
    plot_roc(fpr_list, tpr_list, auc_val, n_pos, n_neg)


if __name__ == '__main__':
    main()
