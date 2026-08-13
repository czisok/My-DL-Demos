from __future__ import annotations
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import os
import random
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_inputs import load_dataset
from base_model import MyBaseModel
from din_model import MyDINModel
from model_utils import clip_by_global_norm_tf_
from data_inputs import build_train_dataloader, build_test_dataloader
import matplotlib
matplotlib.use('Agg')


best_auc = 0.0
# DATASET_PATH = (
#     '/Users/bytedance/Downloads/dataset_for_dl/amazon_review_data/amazon_2014/data_for_din/electronics_dataset.pkl'
# )

DATASET_PATH = (
    '/home/zhangbo.999/jupyter_workspace/dataset/amazon_review_data/'
    'amazon_2014/din_raw_data/electronics_dataset.pkl'
)

def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """安全计算 AUC：当只有 1 个类别时返回 NaN，避免 sklearn UndefinedMetricWarning。"""
    n_pos = int(np.sum(y_true > 0.5))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


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
            uid_2b = torch.cat([uid,     uid],     dim=0)   # [2B]
            iid_2b = torch.cat([iid_pos, iid_neg], dim=0)   # [2B]
            hist_i_2b = torch.cat([hist_i,  hist_i],  dim=0)   # [2B, T]
            sl_2b = torch.cat([sl,      sl],      dim=0)   # [2B]
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


MODEL_MAP = {
    'base_model': MyBaseModel,
    'din': MyDINModel,
}


def main(
    model_name: str = 'base_model',
    num_epochs: int = 50,
    train_batch_size: int = 32,
    test_batch_size: int = 512,
    base_lr: float = 1.0,            # 对齐 base_model: GradientDescent lr=1.0
    eval_every_steps: int = 1000,    # 对齐 base_model: global_step % 1000 == 0 评估
    lr_decay_step: int = 336000,     # 对齐 base_model: 第 336k step 时 lr *= 0.1
    save_path: str | None = None,
    dataset_path: str | None = None,
    seed: int = 1234,
    print_every: int = 500,
    grad_clip_norm: float = 5.0,     # 对齐 base_model: clip_by_global_norm(5)
):
    """
    多 epoch 训练主函数（对齐 base_model/train.py 的调度逻辑）。

    Args:
        model_name:   可选 'base_model' / 'din'，分别使用 MyBaseModel / MyDINModel
        num_epochs:   训练 epoch 数
        train_batch_size / test_batch_size: 训练/评估 batch
        base_lr:      初始学习率
        eval_every_steps: 每多少 global_step 评估 & 存 ckpt 一次
        lr_decay_step:    在第几步将 lr *= 0.1
        save_path:    ckpt 保存路径；为 None 时自动用 save_path/{model_name}/ckpt.pt
        dataset_path: 数据集 .pkl 路径；为 None 时使用模块级 DATASET_PATH（或环境变量 DIN_DATASET_PATH）
        seed:         全局随机种子
        print_every:  每多少个 train batch 打印一次进度日志
        grad_clip_norm: global_norm 裁剪阈值（5 对齐 TF base_model）

    调度逻辑对比 base_model：
      • num_epochs = 50 (默认)
      • 训练前跑一次基线 _eval
      • 每次 global_step % 1000 == 0 → _eval，输出 Epoch / Global_step / Train_loss / Eval_GAUC / Eval_AUC
      • global_step == lr_decay_step → lr = base_lr * 0.1
      • 若 test_gauc > best_auc → torch.save ckpt
      • 每 epoch 结束输出 summary（samples/batches/elapsed/avg_loss/train_auc/best_gauc）
    """
    if model_name not in MODEL_MAP:
        raise ValueError(f'model_name={model_name!r} 不合法，可选值: {sorted(MODEL_MAP)}')

    set_seed(seed)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = dataset_path or os.environ.get('DIN_DATASET_PATH') or DATASET_PATH
    if save_path is None:
        save_path = os.path.join('save_path', model_name, 'ckpt.pt')

    print(f'Using device: {device}')
    print(f'model_name : {model_name}  (class={MODEL_MAP[model_name].__name__})')
    print(f'save_path  : {save_path}')

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    print(f'Loading dataset from: {data_path}')
    if not os.path.exists(data_path):
        print(f'[WARN] 数据集路径不存在：{data_path}', file=sys.stderr)
        print('       可通过 --dataset_path 参数、DIN_DATASET_PATH 环境变量，或修改文件内 DATASET_PATH 常量传入正确路径。', file=sys.stderr)
        sys.exit(1)

    train_set, test_set, cate_list, (user_count, item_count, cate_count) = load_dataset(data_path)
    print(
        f'Loaded: train={len(train_set):,}  test={len(test_set):,}  '
        f'users={user_count:,}  items={item_count:,}  cates={cate_count:,}'
    )

    train_loader = build_train_dataloader(train_set, batch_size=train_batch_size, shuffle=False, seed=seed)
    test_loader = build_test_dataloader(test_set, batch_size=test_batch_size, shuffle=False, seed=seed)

    # ------------------------------------------------------------------
    # 模型 & 优化器（对齐 base_model: SGD + clip_by_global_norm(5)）
    # ------------------------------------------------------------------
    hidden_units = 128
    ModelClass = MODEL_MAP[model_name]
    model = ModelClass(
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
    print(f'Model class: {ModelClass.__name__}  params: {total_params:,}')
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
    save_fig_fname = f'training_curves_dashed_{model_name}.png'
    save_fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_fig_fname)
    fig.savefig(save_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[FIG] 训练曲线（虚线）已保存至: {save_fig_path}')


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='DIN / BaseModel PyTorch training')
    parser.add_argument(
        '--model_name',
        type=str,
        default='base_model',
        choices=sorted(MODEL_MAP.keys()),
        help="选择训练模型：'base_model' → 平均池化基线，'din' → DIN Attention 版（含候选-历史交互打分）",
    )
    parser.add_argument('--num_epochs',        type=int,   default=50)
    parser.add_argument('--train_batch_size',  type=int,   default=32)
    parser.add_argument('--test_batch_size',   type=int,   default=512)
    parser.add_argument('--base_lr',           type=float, default=1.0)
    parser.add_argument('--eval_every_steps',  type=int,   default=1000)
    parser.add_argument('--lr_decay_step',     type=int,   default=336000)
    parser.add_argument('--save_path',         type=str,   default=None,
                        help='ckpt 保存路径（默认 save_path/{model_name}/ckpt.pt）')
    parser.add_argument('--dataset_path',      type=str,   default=None,
                        help='数据集 pkl 路径（优先级 > DIN_DATASET_PATH 环境变量 > 文件内 DATASET_PATH 常量）')
    parser.add_argument('--seed',              type=int,   default=1234)
    parser.add_argument('--print_every',       type=int,   default=500)
    parser.add_argument('--grad_clip_norm',    type=float, default=5.0)
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    best_auc = 0.0
    main(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        base_lr=args.base_lr,
        eval_every_steps=args.eval_every_steps,
        lr_decay_step=args.lr_decay_step,
        save_path=args.save_path,
        dataset_path=args.dataset_path,
        seed=args.seed,
        print_every=args.print_every,
        grad_clip_norm=args.grad_clip_norm,
    )
