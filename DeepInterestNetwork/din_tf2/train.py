"""
DIN Training Script - TF2 版
=============================
与 TF1 版 din/train.py 保持**完全相同的训练逻辑和评估逻辑**，
仅将 TensorFlow 1.x Session/Graph API 改为 TF2 的 Eager + GradientTape + Keras Optimizer。

关键 TF1 → TF2 对应关系：
  - tf.Session / sess.run(feed_dict)     → 直接调用 tf.constant + model.train_on_batch / eval_on_batch
  - tf.GPUOptions(allow_growth=True)     → tf.config.experimental.list_physical_devices('GPU') + set_memory_growth
  - tf.set_random_seed(1234)             → tf.random.set_seed(1234)
  - tf.train.GradientDescentOptimizer    → tf.keras.optimizers.SGD
  - tf.clip_by_global_norm (tf.gradients)→ tape.gradient + tf.clip_by_global_norm
  - global_step.eval() / assign_op.eval()→ model.global_step (tf.Variable) 直接 .assign_add / read_value
  - tf.global_variables_initializer()    → 变量在 build 时/第一次前向时自动创建，无需显式 init
  - tf.train.Saver / saver.save(sess, p) → tf.train.Checkpoint + manager.save 或直接 ckpt.save
  - loss = model.train(sess, uij, lr)    → loss = model.train_on_batch(*uij拆包, lr=lr, optimizer=opt)

calc_auc / _auc_arr / _eval / _test 的纯 numpy/python 逻辑**一行不改**。
"""

import os
import sys
import time
import pickle
import random

import numpy as np
import tensorflow as tf

from input import DataInput, DataInputTest
from model import Model


# ============================================================
# GPU / 随机种子配置 (TF2 API)
# ============================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 允许 GPU 显存按需增长（替代 TF1 的 allow_growth=True）
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)   # 替代 TF1 的 tf.set_random_seed


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
# AUC 计算（逻辑与原 train.py 保持完全一致，不做任何修改）
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
# 评估 / 测试（用 model.eval_on_batch 替代 sess.run）
# ============================================================

def _eval(model, ckpt_manager):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        auc_, score_ = model.eval_on_batch(uij[0], uij[1], uij[2], uij[3], uij[4])
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        # 保存模型 checkpoint（对应原 model.save(sess, 'save_path/ckpt')）
        save_path = ckpt_manager.save() if ckpt_manager is not None else None
        if save_path is None:
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.save('save_path/ckpt')
    return test_gauc, Auc


def _test(model):
    predicted_users_num = 0
    score_arr = []
    print("test sub items")
    for _, uij in DataInputTest(test_set, predict_batch_size):
        if predicted_users_num >= predict_users_num:
            break
        score_ = model.test_on_batch(uij[0], uij[1], uij[2], uij[3], uij[4])
        score_arr.append(score_)
        predicted_users_num += predict_batch_size
    return score_[0]


# ============================================================
# 主训练循环
# ============================================================

def main():
    # ---- 模型 & 优化器 ----
    model = Model(user_count, item_count, cate_count, cate_list,
                  predict_batch_size, predict_ads_num)
    # TF2 SGD 替代 TF1 GradientDescentOptimizer；初始 lr=1.0，每 336000 步降到 0.1
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    # ---- 触发 build（与 TF1 global_variables_initializer 对应，确保变量被创建）----
    # 用 dummy 输入走一次前向，保证所有权重变量和 BatchNorm 的均值/方差都创建好
    dummy_B = 4
    dummy_T = 5
    dummy_u = np.zeros([dummy_B], dtype=np.int32)
    dummy_i = np.zeros([dummy_B], dtype=np.int32)
    dummy_j = np.zeros([dummy_B], dtype=np.int32)
    dummy_y = np.zeros([dummy_B], dtype=np.float32)
    dummy_hist = np.zeros([dummy_B, dummy_T], dtype=np.int32)
    dummy_sl = np.full([dummy_B], dummy_T, dtype=np.int32)
    # 走一次 training=True 前向，保证 keras.layers.BatchNormalization 的 running mean/var 变量被创建
    _ = model.forward_train(
        tf.constant(dummy_u), tf.constant(dummy_i), tf.constant(dummy_y),
        tf.constant(dummy_hist), tf.constant(dummy_sl),
        training=True,
    )
    # 同时走一次 eval，保证 j 分支也 build（共享权重，但形状推导需要）
    _ = model.forward_eval(
        tf.constant(dummy_u), tf.constant(dummy_i), tf.constant(dummy_j),
        tf.constant(dummy_hist), tf.constant(dummy_sl),
        training=False,
    )
    print('Model variables built. #Trainable params: %d' %
          sum(int(np.prod(v.shape)) for v in model.trainable_variables))

    # ---- Checkpoint（对应原 TF1 Saver）----
    os.makedirs('save_path', exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory='save_path', max_to_keep=5,
                                              checkpoint_name='ckpt')

    # ---- 初始评估（与原 train.py 相同）----
    test_gauc0, Auc0 = _eval(model, ckpt_manager)
    print('test_gauc: %.4f\t test_auc: %.4f' % (test_gauc0, Auc0))
    sys.stdout.flush()

    lr = 1.0
    start_time = time.time()

    for _ in range(50):
        random.shuffle(train_set)
        epoch_size = round(len(train_set) / train_batch_size)
        loss_sum = 0.0

        for _, uij in DataInput(train_set, train_batch_size):
            # 学习率 lr 在 model.train_on_batch 内部自动同步到 optimizer
            loss = model.train_on_batch(uij[0], uij[1], uij[2], uij[3], uij[4],
                                        lr=lr, optimizer=optimizer)
            loss_sum += loss

            # global_step 在 train_on_batch 中已经 assign_add(1)
            gstep = int(model.global_step.numpy())
            if gstep % 1000 == 0:
                test_gauc, Auc = _eval(model, ckpt_manager)
                epoch_cnt = int(model.global_epoch_step.numpy())
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                      (epoch_cnt, gstep, loss_sum / 1000, test_gauc, Auc))
                sys.stdout.flush()
                loss_sum = 0.0

            if gstep % 336000 == 0:
                lr = 0.1

        epoch_done = int(model.global_epoch_step.numpy())
        print('Epoch %d DONE\tCost time: %.2f' % (epoch_done, time.time() - start_time))
        sys.stdout.flush()
        # 对应原 TF1 的 model.global_epoch_step_op.eval()
        model.global_epoch_step.assign_add(1)

    print('best test_gauc:', best_auc)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
