"""
DIN (Deep Interest Network) - PyTorch 版模型实现
===================================================
与 TF1 版 din/model.py 的**建模逻辑完全一致**：
  结构：
    Embedding (user:128, item:64, cate:64)
      → Attention MLP (concat[q,k,q-k,q*k] → 80sigm→40sigm→1None, mask→scale→softmax→weighted sum)
      → BatchNorm1d (hist_bn) + Linear(128→128)  (hist_fcn)
      → DIN FCN:  concat[u, i, u*i]
                   → BN (b1)
                   → Linear 80 (sigmoid or Dice)
                   → Linear 40 (sigmoid or Dice)
                   → Linear 1
      → logit = item_bias + FCN_out
  权重复用：
    正样本 i、负样本 j、多候选 item sub：完全共享 Attention MLP / hist_bn / hist_fcn / FCN 参数
  训练：
    loss = BCEWithLogitsLoss (等价 reduce_mean(sigmoid_cross_entropy_with_logits(logits, y)))
    grad_clip = clip_grad_norm_(params, max_norm=5)

TF1 → PyTorch 关键 API 对照：
  tf.placeholder + sess.run(feed_dict)        → forward(train/eval/test) 显式传 torch tensors
  tf.get_variable + variable_scope            → nn.Embedding / nn.Linear / nn.Parameter
  tf.layers.batch_normalization + reuse       → 同一个 BatchNorm1d 实例多次调用
  tf.layers.dense + reuse                     → 同一个 nn.Linear 实例多次调用
  tf.nn.embedding_lookup(W, ids)              → W(ids)   (nn.Embedding 的 __call__ 语义)
  tf.gather(params, indices)                  → params[indices] or params.index_select(0, indices)
  tf.concat([...], axis=dim)                  → torch.cat([...], dim=dim)
  tf.range / tf.shape(x)[0]                   → torch.arange / x.shape[0]
  tf.sequence_mask(lengths, max_len)          → torch.arange(max_len)[None, :] < lengths[:, None]
  tf.where(mask, a, b)                        → torch.where(mask, a, b)
  tf.reduce_mean / sum / etc                  → torch.mean / sum / etc
  tf.to_float                                 → .to(torch.float32)
  tf.train.GradientDescentOptimizer           → torch.optim.SGD
  clip_by_global_norm                         → torch.nn.utils.clip_grad_norm_
  tf.train.Saver save/restore                 → torch.save / torch.load (state_dict)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dice import DiceLayer


# ===============================================================
# 1. Attention MLP（对应原 TF1 的 attention() + attention_multi_items() 中的共享权重）
# ===============================================================

class AttentionMLP(nn.Module):
    """
    attention 中三层共享的打分 MLP。
    输入 concat[q, k, q-k, q*k]，输出 [*, 1] 的 raw score (未 sigmoid)
    TF1 原结构：f1_att(Dense 80, sigmoid) → f2_att(Dense 40, sigmoid) → f3_att(Dense 1, None)
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.f1 = nn.Linear(in_dim, 80)
        self.f2 = nn.Linear(80, 40)
        self.f3 = nn.Linear(40, 1)
        # TF1 默认 glorot_uniform (fan_avg) 初始化 + 零 bias；PyTorch 默认 kaiming_uniform
        # 这里显式对齐为均匀 xavier_uniform + bias=0 与 TF1 尽量接近
        for m in (self.f1, self.f2, self.f3):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.f1(x))
        x = torch.sigmoid(self.f2(x))
        x = self.f3(x)
        return x


# ===============================================================
# 2. Attention 核心函数 (单 query 与 多 query)
# ===============================================================

def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    等价于 tf.sequence_mask(lengths, max_len)，返回 bool [B, max_len]
    """
    B = lengths.shape[0]
    return torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)


def attention_core(
    queries: torch.Tensor,
    keys: torch.Tensor,
    keys_length: torch.Tensor,
    attn_mlp: AttentionMLP,
) -> torch.Tensor:
    """
    对应原 TF1 attention()：
        queries: [B, H]    当前 target item 的 embedding
        keys:    [B, T, H] user 历史序列 embeddings
        keys_length: [B]   每个 user 的有效历史长度

    返回: [B, 1, H]  加权后的 user 兴趣向量（可 squeeze 后作为 u_emb）
    """
    B, T, H = keys.shape
    q = queries.unsqueeze(1).expand(-1, T, -1)                      # [B, T, H]  对应 TF tile+reshape
    din_all = torch.cat([q, keys, q - keys, q * keys], dim=-1)       # [B, T, 4H]
    d3 = attn_mlp(din_all)                                          # [B, T, 1]
    outputs = d3.permute(0, 2, 1)                                   # [B, 1, T]

    # Mask (padding 位置设为 -∞)
    key_masks = _sequence_mask(keys_length, T).unsqueeze(1)         # [B, 1, T]
    paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)            # 对应 TF 的 padding 值
    outputs = torch.where(key_masks, outputs, paddings)             # [B, 1, T]

    # Scale
    outputs = outputs / (H ** 0.5)

    # Softmax + weighted sum
    outputs = F.softmax(outputs, dim=-1)                            # [B, 1, T]
    outputs = torch.matmul(outputs, keys)                           # [B, 1, H]
    return outputs


def attention_multi_items_core(
    queries: torch.Tensor,
    keys: torch.Tensor,
    keys_length: torch.Tensor,
    attn_mlp: AttentionMLP,
) -> torch.Tensor:
    """
    对应原 TF1 attention_multi_items()：
        queries: [B, N, H]  多个候选 item embeddings (N = predict_ads_num)
        keys:    [B, T, H]  user 历史
        keys_length: [B]

    返回: [B, N, H]  每个候选 item 对应的加权 user 兴趣向量
    """
    B, N, H = queries.shape
    T = keys.shape[1]

    q = queries.unsqueeze(2).expand(-1, -1, T, -1)                  # [B, N, T, H]
    k = keys.unsqueeze(1).expand(-1, N, -1, -1)                     # [B, N, T, H]
    din_all = torch.cat([q, k, q - k, q * k], dim=-1)                # [B, N, T, 4H]

    d3 = attn_mlp(din_all)                                          # [B, N, T, 1]
    outputs = d3.squeeze(-1).unsqueeze(-2)                          # [B, N, 1, T]

    # Mask
    base_mask = _sequence_mask(keys_length, T)                      # [B, T]
    key_masks = base_mask.unsqueeze(1).unsqueeze(1).expand(-1, N, 1, -1)  # [B, N, 1, T]
    paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = torch.where(key_masks, outputs, paddings)             # [B, N, 1, T]

    # Scale + softmax
    outputs = outputs / (H ** 0.5)
    outputs = F.softmax(outputs, dim=-1)                            # [B, N, 1, T]

    # 展开成 [B*N, 1, T] × [B*N, T, H] → 再 reshape 回来
    flat_out = outputs.reshape(B * N, 1, T)
    flat_k = k.reshape(B * N, T, H)
    out = torch.matmul(flat_out, flat_k)                             # [B*N, 1, H]
    return out.reshape(B, N, H)


# ===============================================================
# 3. DIN 模型主体 (sigmoid FCN 版，对应 TF1 din/model.py)
# ===============================================================

class DINModel(nn.Module):
    """
    DIN 主体。与 TF1 版 Model 类的计算图一一对应。

    对外接口（与 TF1 版 train/eval/test 语义一致）：
        train_step(u, i, y, hist_i, sl)   → float loss  （内部 backward + clip + opt.step + zero_grad）
        eval_step(u, i, j, hist_i, sl)    → (float mf_auc, ndarray [B,2] p_and_n)
        test_step(u, i, j, hist_i, sl)    → ndarray [B, N, 1] logits_sub

    直接调用 forward(u, i, hist_i, sl) 可以拿到 logits (仅 i 分支的 logit, [B])
    """

    def __init__(
        self,
        user_count: int,
        item_count: int,
        cate_count: int,
        cate_list,
        predict_batch_size: int,
        predict_ads_num: int,
        fcn_activation: str = 'sigmoid',
        use_dice: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.predict_batch_size = predict_batch_size
        self.predict_ads_num = predict_ads_num
        self.use_dice = use_dice
        self.hidden = 128

        # 设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # cate_list: np.array 或 list, 长度 item_count
        self.register_buffer(
            'cate_list',
            torch.as_tensor(list(cate_list), dtype=torch.long, device=self.device),
            persistent=True,
        )

        # --- Embedding & Bias (对应 TF1 get_variable) ---
        # TF1 的初始化没显式指定，我们用 xavier_uniform 对齐通常做法
        self.user_emb = nn.Embedding(user_count, self.hidden)
        self.item_emb = nn.Embedding(item_count, self.hidden // 2)
        self.cate_emb = nn.Embedding(cate_count, self.hidden // 2)
        self.item_bias = nn.Embedding(item_count, 1)  # item_bias shape [N,1]，方便 gather
        nn.init.zeros_(self.item_bias.weight)
        for emb in (self.user_emb, self.item_emb, self.cate_emb):
            nn.init.xavier_uniform_(emb.weight)

        # --- 共享 Attention MLP ---
        attn_in = self.hidden * 4  # 4H = q + k + (q-k) + (q*k)
        self.attn_mlp = AttentionMLP(attn_in)

        # --- 历史聚合 BN + Dense(128→128)（i/j/sub 共享）---
        # TF1: hist_bn inputs=hist_i [B,1,H] 做 2D/3D BN；我们在 H 维上做 BN1d
        self.hist_bn = nn.BatchNorm1d(self.hidden)
        self.hist_fcn = nn.Linear(self.hidden, self.hidden)
        nn.init.xavier_uniform_(self.hist_fcn.weight)
        nn.init.zeros_(self.hist_fcn.bias)

        # --- DIN FCN (b1 BN + f1 80 + f2 40 + f3 1)（i/j/sub 共享）---
        # input: concat[u, i, u*i] → H + H + H = 3H
        din_in = self.hidden * 3
        self.din_bn = nn.BatchNorm1d(din_in)
        self.f1 = nn.Linear(din_in, 80)
        self.f2 = nn.Linear(80, 40)
        self.f3 = nn.Linear(40, 1)
        for m in (self.f1, self.f2, self.f3):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        # Dice (use_dice=True 时替代 sigmoid)
        if self.use_dice:
            self.dice_1 = DiceLayer(80, name='dice_1')
            self.dice_2 = DiceLayer(40, name='dice_2')
        else:
            self.dice_1 = None
            self.dice_2 = None

        # 训练计数器（对应 TF1 的 global_step / global_epoch_step）
        self.global_step = 0
        self.global_epoch_step = 0

    # ---------------------------------------------------------------
    # 辅助：取 item emb + cate emb 拼接 + item bias
    # ---------------------------------------------------------------
    def _item_emb(self, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        item_ids: [B] or [B, ...]
        返回 (emb, bias) 形状分别是 [..., H] 和 [...]（bias 挤压掉最后 1 维）
        """
        ic = self.cate_list[item_ids]
        i_emb = torch.cat([self.item_emb(item_ids), self.cate_emb(ic)], dim=-1)
        i_b = self.item_bias(item_ids).squeeze(-1)
        return i_emb, i_b

    def _hist_emb(self, hist_i: torch.Tensor) -> torch.Tensor:
        """hist_i [B, T] → h_emb [B, T, H]  (H = item_half + cate_half = 128)"""
        hc = self.cate_list[hist_i]
        return torch.cat([self.item_emb(hist_i), self.cate_emb(hc)], dim=-1)

    # ---------------------------------------------------------------
    # 辅助：user 表示（attention → hist_bn → hist_fcn）
    # ---------------------------------------------------------------
    def _user_repr(self, target_emb, hist_emb, sl):
        """
        target_emb: [B, H] 或 [B, N, H] (multi items)
        hist_emb:   [B, T, H]
        sl:         [B] (int tensor)
        返回: u_emb: [B, H] 或 [B*N, H]
        """
        if target_emb.dim() == 3:
            # 多候选情况
            hist = attention_multi_items_core(target_emb, hist_emb, sl, self.attn_mlp)  # [B, N, H]
            B, N, H = hist.shape
            hist_2d = hist.reshape(B * N, H)
        else:
            hist = attention_core(target_emb, hist_emb, sl, self.attn_mlp)  # [B, 1, H]
            H = hist.shape[-1]
            hist_2d = hist.reshape(-1, H)                                     # [B, H]

        # hist_bn: BN1d 作用在最后一维 H，输入要求 [*, H]
        hist_2d = self.hist_bn(hist_2d)
        hist_2d = self.hist_fcn(hist_2d)
        return hist_2d

    # ---------------------------------------------------------------
    # 辅助：DIN FCN 前向
    # ---------------------------------------------------------------
    def _fcn(self, u_emb_2d, i_emb_2d):
        """
        u_emb_2d: [*, H]
        i_emb_2d: [*, H]
        返回: [*] 标量（squeeze 后的 FCN 输出，不含 item_bias）
        """
        din = torch.cat([u_emb_2d, i_emb_2d, u_emb_2d * i_emb_2d], dim=-1)  # [*, 3H]
        din = self.din_bn(din)
        x = self.f1(din)
        x = self.dice_1(x) if self.use_dice else torch.sigmoid(x)
        x = self.f2(x)
        x = self.dice_2(x) if self.use_dice else torch.sigmoid(x)
        x = self.f3(x)                                          # [*, 1]
        return x.squeeze(-1)                                    # [*]

    # ---------------------------------------------------------------
    # 标准 forward: 仅 i 分支，返回 logits [B]（对应 TF1 self.logits）
    # ---------------------------------------------------------------
    def forward(self, u, i, hist_i, sl):
        u = torch.as_tensor(u, dtype=torch.long, device=self.device)
        i = torch.as_tensor(i, dtype=torch.long, device=self.device)
        hist_i = torch.as_tensor(hist_i, dtype=torch.long, device=self.device)
        sl = torch.as_tensor(sl, dtype=torch.long, device=self.device)

        i_emb, i_b = self._item_emb(i)
        h_emb = self._hist_emb(hist_i)
        u_emb_i = self._user_repr(i_emb, h_emb, sl)             # [B, H]
        d3_i = self._fcn(u_emb_i, i_emb)
        return i_b + d3_i

    # ---------------------------------------------------------------
    # 训练：一次 step，返回 float loss
    # ---------------------------------------------------------------
    def train_step(self, u, i, y, hist_i, sl, optimizer, lr: Optional[float] = None) -> float:
        """
        等价于原 TF1 Model.train(sess, uij, l)。
        内部：zero_grad → forward → loss.backward → clip_grad_norm(5) → optimizer.step → global_step++
        """
        # 设置学习率
        if lr is not None:
            for pg in optimizer.param_groups:
                pg['lr'] = float(lr)

        u_t = torch.as_tensor(u, dtype=torch.long, device=self.device)
        i_t = torch.as_tensor(i, dtype=torch.long, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        hist_t = torch.as_tensor(hist_i, dtype=torch.long, device=self.device)
        sl_t = torch.as_tensor(sl, dtype=torch.long, device=self.device)

        self.train()
        optimizer.zero_grad(set_to_none=True)

        i_emb, i_b = self._item_emb(i_t)
        h_emb = self._hist_emb(hist_t)
        u_emb_i = self._user_repr(i_emb, h_emb, sl_t)
        d3_i = self._fcn(u_emb_i, i_emb)
        logits = i_b + d3_i

        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()

        self.global_step += 1
        return float(loss.detach().cpu().numpy())

    # ---------------------------------------------------------------
    # 评估：返回 (mf_auc float, p_and_n ndarray [B,2])
    # ---------------------------------------------------------------
    @torch.no_grad()
    def eval_step(self, u, i, j, hist_i, sl) -> Tuple[float, np.ndarray]:
        """
        等价于原 TF1 Model.eval(sess, uij)。
        返回 (mf_auc, p_and_n)
          - mf_auc: 正样本得分 > 负样本得分的 batch 平均比例
          - p_and_n: shape [B, 2], 第 0 列正样本 sigmoid, 第 1 列负样本 sigmoid
        """
        self.eval()
        u_t = torch.as_tensor(u, dtype=torch.long, device=self.device)
        i_t = torch.as_tensor(i, dtype=torch.long, device=self.device)
        j_t = torch.as_tensor(j, dtype=torch.long, device=self.device)
        hist_t = torch.as_tensor(hist_i, dtype=torch.long, device=self.device)
        sl_t = torch.as_tensor(sl, dtype=torch.long, device=self.device)

        i_emb, i_b = self._item_emb(i_t)
        j_emb, j_b = self._item_emb(j_t)
        h_emb = self._hist_emb(hist_t)

        u_emb_i = self._user_repr(i_emb, h_emb, sl_t)
        u_emb_j = self._user_repr(j_emb, h_emb, sl_t)

        d3_i = self._fcn(u_emb_i, i_emb)
        d3_j = self._fcn(u_emb_j, j_emb)

        x = (i_b - j_b) + (d3_i - d3_j)
        # TF1: reduce_mean(to_float(x > 0))
        mf_auc = (x > 0).to(torch.float32).mean().item()

        score_i = torch.sigmoid(i_b + d3_i).unsqueeze(-1)    # [B, 1]
        score_j = torch.sigmoid(j_b + d3_j).unsqueeze(-1)    # [B, 1]
        p_and_n = torch.cat([score_i, score_j], dim=-1).cpu().numpy()   # [B, 2]
        return float(mf_auc), p_and_n

    # ---------------------------------------------------------------
    # 测试：返回 logits_sub ndarray [B, predict_ads_num, 1]
    # ---------------------------------------------------------------
    @torch.no_grad()
    def test_step(self, u, i, j, hist_i, sl) -> np.ndarray:
        """
        等价于原 TF1 Model.test(sess, uij)。
        取前 predict_ads_num 个商品与 user 表示打分，shape [B, N, 1]
        """
        self.eval()
        hist_t = torch.as_tensor(hist_i, dtype=torch.long, device=self.device)
        sl_t = torch.as_tensor(sl, dtype=torch.long, device=self.device)

        h_emb = self._hist_emb(hist_t)
        B = hist_t.shape[0]
        N = self.predict_ads_num

        # 前 N 个商品的 emb + bias
        top_ids = torch.arange(N, device=self.device, dtype=torch.long)
        ic_all = self.cate_list[top_ids]
        item_emb_all = torch.cat([self.item_emb(top_ids), self.cate_emb(ic_all)], dim=-1)   # [N, H]
        item_emb_sub = item_emb_all.unsqueeze(0).expand(B, -1, -1)                         # [B, N, H]
        item_b_sub = self.item_bias(top_ids).squeeze(-1)                                   # [N]

        # sub-user 表示
        u_emb_sub = self._user_repr(item_emb_sub, h_emb, sl_t)  # [B*N, H]
        i_emb_flat = item_emb_sub.reshape(B * N, self.hidden)   # [B*N, H]
        d3_sub = self._fcn(u_emb_sub, i_emb_flat)               # [B*N]
        d3_sub = d3_sub.reshape(B, N)                            # [B, N]

        logits_sub = torch.sigmoid(item_b_sub.unsqueeze(0) + d3_sub)   # [B, N]
        return logits_sub.unsqueeze(-1).cpu().numpy()                  # [B, N, 1]

    # ---------------------------------------------------------------
    # save / restore（对应 TF1 Saver）
    # ---------------------------------------------------------------
    def save(self, path: str):
        """同时保存 state_dict 和 step 计数，便于完整恢复"""
        state = {
            'model_state_dict': self.state_dict(),
            'global_step': self.global_step,
            'global_epoch_step': self.global_epoch_step,
        }
        torch.save(state, path)

    def restore(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            self.load_state_dict(ckpt['model_state_dict'])
            self.global_step = ckpt.get('global_step', self.global_step)
            self.global_epoch_step = ckpt.get('global_epoch_step', self.global_epoch_step)
        else:
            # 兼容直接保存 state_dict 的场景
            self.load_state_dict(ckpt)


# 与 TF1 din/model.py 使用方式一致，Model = DINModel
Model = DINModel
