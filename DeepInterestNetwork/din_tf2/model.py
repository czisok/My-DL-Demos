"""
DIN (Deep Interest Network) - TF2 版模型实现
================================================
与 TF1 版 din/model.py 的建模逻辑保持完全一致：
  - 结构：Embedding → Attention(MLP) → BatchNorm → Dense(128) → FCN(80→40→1)
  - 共享权重：正样本 i 分支、负样本 j 分支、sub-item 多分支之间 FCN / Attention 参数共享
  - 评估指标：mf_auc (正样本得分 > 负样本得分的比例) + p_and_n (正负样本 sigmoid 概率)
  - 损失：正样本 logit 对 y 做 sigmoid_cross_entropy_with_logits 的均值

TF1 → TF2 的主要迁移点：
  1. tf.placeholder        → 前向函数 train_step / eval_step / test_step 的入参
  2. tf.get_variable       → tf.Variable (由 build() 或直接创建)
  3. tf.layers.dense/bn    → tf.keras.layers.Dense/BatchNormalization (一次实例化后重复调用来复用权重)
  4. tf.to_float(x>0)      → tf.cast(x>0, tf.float32)
  5. tf.train.Optimizer    → tf.keras.optimizers.SGD
  6. sess.run(train_op)    → tf.GradientTape (放在 train.py 中，模型内部提供前向和损失计算接口)
  7. tf.train.Saver        → tf.train.Checkpoint
  8. tf.assign + global_step → 用 python int 或 tf.Variable 跟踪（train.py 中做）
"""

import tensorflow as tf

from Dice import dice


# ============================================================
# Attention Layer (单个 query 对应一条历史)
# ============================================================

class AttentionMLPLayer(tf.keras.layers.Layer):
    """
    对应原 TF1 的 attention() + attention_multi_items() 中共享的三层 MLP：
    f1_att(80, sigmoid) → f2_att(40, sigmoid) → f3_att(1, None)
    同一实例多次调用会自动复用权重（对应原 reuse=AUTO_REUSE）。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f1 = tf.keras.layers.Dense(80, activation=tf.nn.sigmoid, name='f1_att')
        self.f2 = tf.keras.layers.Dense(40, activation=tf.nn.sigmoid, name='f2_att')
        self.f3 = tf.keras.layers.Dense(1, activation=None, name='f3_att')

    def call(self, din_all):
        return self.f3(self.f2(self.f1(din_all)))


def _attention_core(queries, keys, keys_length, attn_mlp):
    """
    单 query 的 attention 计算：queries:[B,H], keys:[B,T,H], keys_length:[B]
    与原 attention() 函数逻辑完全对应：
        tile+reshape queries → din_all concat[q,k,q-k,q*k] → MLP → mask → scale → softmax → weighted sum
    """
    queries_hidden_units = queries.shape[-1]
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)

    d_layer_3_all = attn_mlp(din_all)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
    outputs = d_layer_3_all

    # Mask 掉 padding 位置
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.shape[-1] ** 0.5)

    # Softmax + weighted sum
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]
    return outputs


def _attention_multi_items_core(queries, keys, keys_length, attn_mlp):
    """
    多 query（N 个商品）的 attention：queries:[B,N,H], keys:[B,T,H]
    与原 attention_multi_items() 函数逻辑完全对应。
    """
    queries_hidden_units = queries.shape[-1]
    queries_nums = queries.shape[1]
    max_len = tf.shape(keys)[1]

    queries = tf.tile(queries, [1, 1, max_len])
    queries = tf.reshape(queries, [-1, queries_nums, max_len, queries_hidden_units])

    keys = tf.tile(keys, [1, queries_nums, 1])
    keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units])

    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    d_layer_3_all = attn_mlp(din_all)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
    outputs = d_layer_3_all

    # Mask
    key_masks = tf.sequence_mask(keys_length, max_len)
    key_masks = tf.tile(key_masks, [1, queries_nums])
    key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)

    # Scale + softmax
    outputs = outputs / (keys.shape[-1] ** 0.5)
    outputs = tf.nn.softmax(outputs)
    outputs = tf.reshape(outputs, [-1, 1, max_len])
    keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])

    # Weighted sum
    outputs = tf.matmul(outputs, keys)
    outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])
    return outputs


# ============================================================
# DIN Model (sigmoid FCN 版，对应原 din/model.py)
# ============================================================

class DINModel(tf.keras.Model):
    """
    DIN 模型主体。建模结构与 TF1 版完全一致。

    - 前向接口：
        forward_train(u, i, y, hist_i, sl, training)     # 返回 loss（用于训练）
        forward_eval(u, i, j, hist_i, sl, training)      # 返回 (mf_auc, p_and_n)
        forward_test(u, i, j, hist_i, sl, training)      # 返回 logits_sub
    - 权重复用：
        通过同一个 Dense / BatchNormalization Layer 对象多次 __call__ 实现参数共享，
        对应原 TF1 版中 reuse=True / reuse=AUTO_REUSE。
    """

    def __init__(self, user_count, item_count, cate_count, cate_list,
                 predict_batch_size, predict_ads_num,
                 fcn_activation=tf.nn.sigmoid,
                 use_dice=False, **kwargs):
        """
        Args:
            user_count, item_count, cate_count: 各类数量
            cate_list: list/array，长度 item_count，cate_list[i] = 商品 i 的类别 id
            predict_batch_size, predict_ads_num: test 时预测的 batch 和 ad 数量
            fcn_activation: FCN 中间层激活，默认 sigmoid（对应原 model.py）
            use_dice: 是否用 Dice 替代 sigmoid（对应原 model_dice.py 的行为，通过参数复用）
        """
        super().__init__(**kwargs)
        self._hidden = 128
        self._predict_batch_size = predict_batch_size
        self._predict_ads_num = predict_ads_num
        self._use_dice = use_dice
        self._fcn_activation = fcn_activation

        # -------- Embedding & Bias --------
        self.user_emb_w = tf.Variable(
            tf.random.truncated_normal([user_count, self._hidden], stddev=0.01),
            name='user_emb_w', trainable=True,
        )
        self.item_emb_w = tf.Variable(
            tf.random.truncated_normal([item_count, self._hidden // 2], stddev=0.01),
            name='item_emb_w', trainable=True,
        )
        self.item_b = tf.Variable(
            tf.zeros([item_count], dtype=tf.float32),
            name='item_b', trainable=True,
        )
        self.cate_emb_w = tf.Variable(
            tf.random.truncated_normal([cate_count, self._hidden // 2], stddev=0.01),
            name='cate_emb_w', trainable=True,
        )
        self.cate_list = tf.constant(cate_list, dtype=tf.int64)

        # -------- Attention MLP (i / j / sub 三个路径共享) --------
        self._attn_mlp = AttentionMLPLayer(name='attn_mlp')

        # -------- 历史聚合 BN + Dense (i/j 共享，sub 也共享) --------
        self._hist_bn = tf.keras.layers.BatchNormalization(name='hist_bn')
        self._hist_fcn = tf.keras.layers.Dense(self._hidden, name='hist_fcn')

        # -------- DIN FCN (i/j/sub 共享) --------
        self._din_bn = tf.keras.layers.BatchNormalization(name='b1')
        self._f1 = tf.keras.layers.Dense(80, activation=None, name='f1')
        self._f2 = tf.keras.layers.Dense(40, activation=None, name='f2')
        self._f3 = tf.keras.layers.Dense(1, activation=None, name='f3')

        # 可在 build_dataset.py 或 train.py 中手动设置，但不要求
        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step', dtype=tf.int64)

    # -------------------------------------------------------------
    # 工具：取 item embedding + item bias
    # -------------------------------------------------------------
    def _item_emb(self, item_ids):
        ic = tf.gather(self.cate_list, item_ids)
        i_emb = tf.concat([
            tf.nn.embedding_lookup(self.item_emb_w, item_ids),
            tf.nn.embedding_lookup(self.cate_emb_w, ic),
        ], axis=1)
        i_b = tf.gather(self.item_b, item_ids)
        return i_emb, i_b

    def _hist_emb(self, hist_i):
        hc = tf.gather(self.cate_list, hist_i)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(self.item_emb_w, hist_i),
            tf.nn.embedding_lookup(self.cate_emb_w, hc),
        ], axis=2)
        return h_emb

    def _apply_fcn(self, din, training):
        """
        三层 FCN：f1→(act or dice)→f2→(act or dice)→f3
        与原 model.py 的结构一致：sigmoid 激活，或用 Dice 替代
        """
        x = self._din_bn(din, training=training)
        x = self._f1(x)
        if self._use_dice:
            x = dice(x, name='dice_1')
        else:
            x = self._fcn_activation(x)
        x = self._f2(x)
        if self._use_dice:
            x = dice(x, name='dice_2')
        else:
            x = self._fcn_activation(x)
        x = self._f3(x)
        return x

    def _user_repr(self, target_emb, hist_emb, sl, training):
        """
        单目标商品对应的用户表示：
            attention → hist_bn → reshape → hist_fcn
        注意：第一次调用 i_emb 时 hist_bn / hist_fcn 会 build，之后调用 j_emb 复用。
        """
        hist = _attention_core(target_emb, hist_emb, sl, self._attn_mlp)
        hist = self._hist_bn(hist, training=training)
        hist = tf.reshape(hist, [-1, self._hidden])
        hist = self._hist_fcn(hist)
        return hist

    # -------------------------------------------------------------
    # 对外前向接口
    # -------------------------------------------------------------

    def call(self, inputs, training=None):
        # keras.Model 需要实现 call，但这里用显式的 forward_* 方法更清晰
        return self.forward_train(*inputs, training=training)

    def forward_train(self, u, i, y, hist_i, sl, training=True):
        """
        训练前向：给定正样本 i 及其标签 y，返回 BCE loss (标量)
        对应原 TF1 的 self.loss = reduce_mean(sigmoid_cross_entropy_with_logits(...))
        返回值与原 model.train() 的 loss 相同。
        """
        i_emb, i_b = self._item_emb(i)
        h_emb = self._hist_emb(hist_i)
        u_emb_i = self._user_repr(i_emb, h_emb, sl, training=training)

        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        d_layer_3_i = self._apply_fcn(din_i, training=training)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])

        logits = i_b + d_layer_3_i
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
        )
        return loss, logits

    def forward_eval(self, u, i, j, hist_i, sl, training=False):
        """
        评估前向：返回 (mf_auc, p_and_n)
          - mf_auc: batch 内正样本得分 > 负样本得分的比例（对应原 self.mf_auc）
          - p_and_n: shape [B, 2]，第 0 列正样本 sigmoid 概率，第 1 列负样本 sigmoid 概率
        """
        i_emb, i_b = self._item_emb(i)
        j_emb, j_b = self._item_emb(j)
        h_emb = self._hist_emb(hist_i)

        u_emb_i = self._user_repr(i_emb, h_emb, sl, training=training)
        u_emb_j = self._user_repr(j_emb, h_emb, sl, training=training)

        din_i = tf.concat([u_emb_i, i_emb, u_emb_i * i_emb], axis=-1)
        d_layer_3_i = self._apply_fcn(din_i, training=training)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])

        din_j = tf.concat([u_emb_j, j_emb, u_emb_j * j_emb], axis=-1)
        d_layer_3_j = self._apply_fcn(din_j, training=training)
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

        x = i_b - j_b + d_layer_3_i - d_layer_3_j
        mf_auc = tf.reduce_mean(tf.cast(x > 0, tf.float32))

        score_i = tf.sigmoid(i_b + d_layer_3_i)
        score_j = tf.sigmoid(j_b + d_layer_3_j)
        score_i = tf.reshape(score_i, [-1, 1])
        score_j = tf.reshape(score_j, [-1, 1])
        p_and_n = tf.concat([score_i, score_j], axis=-1)
        return mf_auc, p_and_n

    def forward_test(self, u, i, j, hist_i, sl, training=False):
        """
        多候选商品预测：对前 predict_ads_num 个商品批量打分
        对应原 TF1 的 self.logits_sub，shape [B, predict_ads_num, 1]
        """
        h_emb = self._hist_emb(hist_i)

        # 取前 predict_ads_num 个商品的 embedding
        ic_all = tf.gather(self.cate_list, tf.range(self._predict_ads_num, dtype=tf.int64))
        item_emb_all = tf.concat([
            self.item_emb_w[:self._predict_ads_num, :],
            tf.nn.embedding_lookup(self.cate_emb_w, ic_all),
        ], axis=1)
        item_emb_sub = tf.expand_dims(item_emb_all, 0)
        batch_size = tf.shape(hist_i)[0]
        item_emb_sub = tf.tile(item_emb_sub, [batch_size, 1, 1])

        # Multi-items attention
        hist_sub = _attention_multi_items_core(item_emb_sub, h_emb, sl, self._attn_mlp)
        hist_sub = self._hist_bn(hist_sub, training=training)
        hist_sub = tf.reshape(hist_sub, [-1, self._hidden])
        hist_sub = self._hist_fcn(hist_sub)

        u_emb_sub = hist_sub
        item_emb_sub_flat = tf.reshape(item_emb_sub, [-1, self._hidden])
        din_sub = tf.concat([u_emb_sub, item_emb_sub_flat, u_emb_sub * item_emb_sub_flat], axis=-1)
        d_layer_3_sub = self._apply_fcn(din_sub, training=training)
        d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, self._predict_ads_num])

        logits_sub = tf.sigmoid(self.item_b[:self._predict_ads_num] + d_layer_3_sub)
        logits_sub = tf.reshape(logits_sub, [-1, self._predict_ads_num, 1])
        return logits_sub

    # -------------------------------------------------------------
    # 兼容原 train.py 对 model 的调用接口（纯 Python 方法，不返回张量）
    # -------------------------------------------------------------

    def train_on_batch(self, u, i, y, hist_i, sl, lr, optimizer):
        """
        执行一次梯度更新，返回 float loss。
        对应原 Model.train(sess, uij, l)。
        """
        u_t = tf.constant(u, dtype=tf.int32)
        i_t = tf.constant(i, dtype=tf.int32)
        y_t = tf.constant(y, dtype=tf.float32)
        hist_t = tf.constant(hist_i, dtype=tf.int32)
        sl_t = tf.constant(sl, dtype=tf.int32)

        # TF2 keras.optimizers.SGD 的标准学习率属性是 learning_rate，不是 lr
        # 先把外部 lr 同步到 optimizer（如果当前值与传入 lr 不一致的话）
        try:
            cur_lr = float(optimizer.learning_rate.numpy())
        except Exception:
            cur_lr = None
        if cur_lr is None or abs(cur_lr - float(lr)) > 1e-12:
            optimizer.learning_rate.assign(lr)

        with tf.GradientTape() as tape:
            loss, _ = self.forward_train(u_t, i_t, y_t, hist_t, sl_t, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.global_step.assign_add(1)
        return float(loss.numpy())

    def eval_on_batch(self, u, i, j, hist_i, sl):
        """
        对应原 Model.eval(sess, uij) → (mf_auc, p_and_n)
        """
        u_t = tf.constant(u, dtype=tf.int32)
        i_t = tf.constant(i, dtype=tf.int32)
        j_t = tf.constant(j, dtype=tf.int32)
        hist_t = tf.constant(hist_i, dtype=tf.int32)
        sl_t = tf.constant(sl, dtype=tf.int32)
        mf_auc, p_and_n = self.forward_eval(u_t, i_t, j_t, hist_t, sl_t, training=False)
        return float(mf_auc.numpy()), p_and_n.numpy()

    def test_on_batch(self, u, i, j, hist_i, sl):
        """
        对应原 Model.test(sess, uij) → logits_sub ndarray
        """
        u_t = tf.constant(u, dtype=tf.int32)
        i_t = tf.constant(i, dtype=tf.int32)
        j_t = tf.constant(j, dtype=tf.int32)
        hist_t = tf.constant(hist_i, dtype=tf.int32)
        sl_t = tf.constant(sl, dtype=tf.int32)
        logits_sub = self.forward_test(u_t, i_t, j_t, hist_t, sl_t, training=False)
        return logits_sub.numpy()

    def save(self, ckpt_path):
        """对应原 Model.save(sess, path)。由调用方通过 CheckpointManager 或直接 save。"""
        ckpt = tf.train.Checkpoint(model=self)
        ckpt.save(ckpt_path)

    def restore(self, ckpt_path):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt.restore(ckpt_path)


# 为了保持与原 din/model.py 相同的 import 方式，提供一个同名 Model 别名
Model = DINModel
