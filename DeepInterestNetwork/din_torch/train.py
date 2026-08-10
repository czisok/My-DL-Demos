"""
DIN Training Script - PyTorch 版
=================================
与 TF1 版 din/train.py 的**训练 / 评估 / 测试逻辑完全一致**，
仅将 TensorFlow 1.x API 改为 PyTorch 风格。

TF1 → PyTorch 关键对应：
  tf.Session / sess.run                     → model.train_step / eval_step / test_step
  tf.set_random_seed                        → torch.manual_seed + np.random.seed + random.seed
  tf.GPUOptions(allow_growth=True)          → torch.cuda 按需分配（torch 默认）
  os.environ['CUDA_VISIBLE_DEVICES']='1'    → os.environ['CUDA_VISIBLE_DEVICES']='1' (不变)
  tf.train.GradientDescentOptimizer         → torch.optim.SGD
  clip_by_global_norm(gradients, 5)         → nn.utils.clip_grad_norm_(model.parameters(), 5)
  global_step.eval() / assign_op.eval()     → model.global_step (python int)
  model.save(sess, path)                    → torch.save({state_dict, ...}, path)
  calc_auc / _auc_arr / _eval / _test       → 纯 numpy 逻辑，**一行不改**
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import random

import numpy as np
import torch

from input import DataInput, DataInputTest
from model import Model


# ============================================================
# 随机种子 & 设备 (PyTorch 方式)
# ============================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    # 类似 allow_growth，不一次性占满显存
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_per_process_memory_fraction(1.0, i)
        except Exception:
            pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ============================================================
# 超参数（与原 train.py 完全一致）
# ============================================================
train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100


# ============================================================
# 加载数据
# ============================================================
with open('dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0


# ============================================================
# AUC 计算（与 TF1 版逐行相同，不动）
# ============================================================
def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]
        tp2 += record[1]
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:, 0]
    score_n = score[:, 1]
    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0, 1, s])
    for s in score_n.tolist():
        score_arr.append([1, 0, s])
    return score_arr


# ============================================================
# 评估 / 测试（用 model.eval_step 替代 sess.run）
# ============================================================

def _eval(model, ckpt_path):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        auc_, score_ = model.eval_step(uij[0], uij[1], uij[2], uij[3], uij[4])
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        # 保存 checkpoint（对应原 model.save(sess, 'save_path/ckpt')）
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        model.save(ckpt_path)
    return test_gauc, Auc


def _test(model):
    predicted_users_num = 0
    score_arr = []
    print("test sub items")
    for _, uij in DataInputTest(test_set, predict_batch_size):
        if predicted_users_num >= predict_users_num:
            break
        score_ = model.test_step(uij[0], uij[1], uij[2], uij[3], uij[4])
        score_arr.append(score_)
        predicted_users_num += predict_batch_size
    return score_[0]


# ============================================================
# 主训练循环 (PyTorch 标准风格)
# ============================================================

def main():
    # ---- 模型 & 优化器 (SGD with momentum=0, 对应原 TF1 GradientDescentOptimizer) ----
    model = Model(
        user_count=user_count,
        item_count=item_count,
        cate_count=cate_count,
        cate_list=cate_list,
        predict_batch_size=predict_batch_size,
        predict_ads_num=predict_ads_num,
        device=device,
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # 初始 lr=1.0

    # dummy build：走一次训练前向，确保所有 lazy 子模块（若有）都 build 好，
    # 并打印参数数量
    dummy_B = 4
    dummy_T = 5
    dummy_u = np.zeros([dummy_B], dtype=np.int64)
    dummy_i = np.zeros([dummy_B], dtype=np.int64)
    dummy_j = np.zeros([dummy_B], dtype=np.int64)
    dummy_y = np.zeros([dummy_B], dtype=np.float32)
    dummy_hist = np.zeros([dummy_B, dummy_T], dtype=np.int64)
    dummy_sl = np.full([dummy_B], dummy_T, dtype=np.int64)
    # 训练模式
    _ = model.train_step(dummy_u, dummy_i, dummy_y, dummy_hist, dummy_sl, optimizer, lr=1.0)
    # eval 模式
    _ = model.eval_step(dummy_u, dummy_i, dummy_j, dummy_hist, dummy_sl)
    total_params = sum(int(p.numel()) for p in model.parameters())
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    print(f'Model built. total params: {total_params}, trainable params: {trainable_params}')

    # 重置计数器和优化器状态（dummy forward 污染了 global_step 等）
    model.global_step = 0
    model.global_epoch_step = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    # ---- Checkpoint 目录 ----
    os.makedirs('save_path', exist_ok=True)
    ckpt_path = 'save_path/ckpt.pth'

    # ---- 初始评估 ----
    test_gauc0, Auc0 = _eval(model, ckpt_path)
    print('test_gauc: %.4f\t test_auc: %.4f' % (test_gauc0, Auc0))
    sys.stdout.flush()

    lr = 1.0
    start_time = time.time()

    for _ in range(50):
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0

        for _, uij in DataInput(train_set, train_batch_size):
            # 学习率在 336000 步衰减到 0.1（由 train_step 内部读取 param_groups['lr']）
            loss = model.train_step(
                uij[0], uij[1], uij[2], uij[3], uij[4],
                optimizer=optimizer, lr=lr,
            )
            loss_sum += loss

            gstep = int(model.global_step)
            if gstep % 1000 == 0:
                test_gauc, Auc = _eval(model, ckpt_path)
                epoch_cnt = int(model.global_epoch_step)
                print(
                    'Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f'
                    % (epoch_cnt, gstep, loss_sum / 1000, test_gauc, Auc)
                )
                sys.stdout.flush()
                loss_sum = 0.0

            if gstep % 336000 == 0:
                lr = 0.1

        epoch_done = int(model.global_epoch_step)
        print('Epoch %d DONE\tCost time: %.2f' % (epoch_done, time.time() - start_time))
        sys.stdout.flush()
        model.global_epoch_step += 1

    print('best test_gauc:', best_auc)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
