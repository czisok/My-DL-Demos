import tensorflow as tf
import os
import time
import pickle
import random
import numpy as np
import sys

# ==================== 方案 3：限制 GPU 显存增长 ====================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    alphas = tf.Variable(tf.zeros(_x.shape[-1]), name='alpha' + name, dtype=tf.float32)
    beta = tf.Variable(tf.zeros(_x.shape[-1]), name='beta' + name, dtype=tf.float32)

    input_shape = _x.shape.as_list()
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]

    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def parametric_relu(_x):
    alphas = tf.Variable(tf.zeros(_x.shape[-1]), name='alpha', dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


class DataInput:
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

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, y, hist_i, sl)


class DataInputTest:
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

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
        self.i += 1

        u, i, j, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1

        return self.i, (u, i, j, hist_i, sl)


# ==================== Attention Functions ====================

class AttentionLayer(tf.keras.layers.Layer):
    """Attention mechanism for DIN."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(80, activation='sigmoid', name='f1_att')
        self.dense2 = tf.keras.layers.Dense(40, activation='sigmoid', name='f2_att')
        self.dense3 = tf.keras.layers.Dense(1, activation=None, name='f3_att')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        queries, keys, keys_length = inputs
        queries_hidden_units = queries.shape[-1]
        # Tile queries to match time steps
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])
        queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        d_layer_1_all = self.dense1(din_all)
        d_layer_2_all = self.dense2(d_layer_1_all)
        d_layer_3_all = self.dense3(d_layer_2_all)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
        outputs = d_layer_3_all

        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        outputs = outputs / (keys.shape[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]

        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return outputs


class AttentionMultiItemsLayer(tf.keras.layers.Layer):
    """Attention for multiple candidate items."""

    def __init__(self, attention_layer, **kwargs):
        super(AttentionMultiItemsLayer, self).__init__(**kwargs)
        self.attention_layer = attention_layer

    def call(self, inputs):
        queries, keys, keys_length = inputs
        # queries: [B, N, H], keys: [B, T, H]
        queries_hidden_units = queries.shape[-1]
        queries_nums = queries.shape[1]
        max_len = tf.shape(keys)[1]

        queries_tiled = tf.tile(queries, [1, 1, max_len])
        queries_tiled = tf.reshape(queries_tiled, [-1, queries_nums, max_len, queries_hidden_units])

        keys_tiled = tf.tile(keys, [1, queries_nums, 1])
        keys_tiled = tf.reshape(keys_tiled, [-1, queries_nums, max_len, queries_hidden_units])

        din_all = tf.concat([queries_tiled, keys_tiled, queries_tiled - keys_tiled, queries_tiled * keys_tiled], axis=-1)
        d_layer_1_all = self.attention_layer.dense1(din_all)
        d_layer_2_all = self.attention_layer.dense2(d_layer_1_all)
        d_layer_3_all = self.attention_layer.dense3(d_layer_2_all)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
        outputs = d_layer_3_all

        # Mask
        key_masks = tf.sequence_mask(keys_length, max_len)  # [B, T]
        key_masks = tf.tile(key_masks, [1, queries_nums])
        key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len])
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)

        # Scale
        outputs = outputs / (keys.shape[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
        outputs = tf.reshape(outputs, [-1, 1, max_len])
        keys_reshaped = tf.reshape(keys_tiled, [-1, max_len, queries_hidden_units])

        # Weighted sum
        outputs = tf.matmul(outputs, keys_reshaped)
        outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])
        return outputs


# ==================== Model ====================

class DINModel(tf.keras.Model):
    def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
        super(DINModel, self).__init__()

        self.predict_batch_size = predict_batch_size
        self.predict_ads_num = predict_ads_num
        hidden_units = 128

        self.user_emb_w = tf.Variable(tf.random.normal([user_count, hidden_units], stddev=0.01), name='user_emb_w')
        self.item_emb_w = tf.Variable(tf.random.normal([item_count, hidden_units // 2], stddev=0.01), name='item_emb_w')
        self.item_b = tf.Variable(tf.zeros([item_count]), name='item_b')
        self.cate_emb_w = tf.Variable(tf.random.normal([cate_count, hidden_units // 2], stddev=0.01), name='cate_emb_w')
        self.cate_list = tf.constant(cate_list, dtype=tf.int64)

        self.hidden_units = hidden_units

        # Attention layer (shared)
        self.attention_layer = AttentionLayer(name='attention')
        self.attention_multi_layer = AttentionMultiItemsLayer(self.attention_layer, name='attention_multi')

        # Batch normalization layers
        self.hist_bn = tf.keras.layers.BatchNormalization(name='hist_bn')
        self.din_bn = tf.keras.layers.BatchNormalization(name='din_bn')

        # History FCN
        self.hist_fcn = tf.keras.layers.Dense(hidden_units, name='hist_fcn')

        # FCN layers
        self.fcn_layer1 = tf.keras.layers.Dense(80, activation='sigmoid', name='f1')
        self.fcn_layer2 = tf.keras.layers.Dense(40, activation='sigmoid', name='f2')
        self.fcn_layer3 = tf.keras.layers.Dense(1, activation=None, name='f3')

    def _get_item_emb(self, item_ids):
        ic = tf.gather(self.cate_list, item_ids)
        i_emb = tf.concat([
            tf.nn.embedding_lookup(self.item_emb_w, item_ids),
            tf.nn.embedding_lookup(self.cate_emb_w, ic),
        ], axis=-1)
        return i_emb

    def _get_hist_emb(self, hist_i):
        hc = tf.gather(self.cate_list, hist_i)
        h_emb = tf.concat([
            tf.nn.embedding_lookup(self.item_emb_w, hist_i),
            tf.nn.embedding_lookup(self.cate_emb_w, hc),
        ], axis=2)
        return h_emb

    def _user_representation(self, item_emb, h_emb, sl, training=False):
        hist = self.attention_layer([item_emb, h_emb, sl])
        hist = self.hist_bn(hist, training=training)
        hist = tf.reshape(hist, [-1, self.hidden_units])
        hist = self.hist_fcn(hist)
        return hist

    def _fcn_net(self, u_emb, i_emb, training=False):
        din = tf.concat([u_emb, i_emb, u_emb * i_emb], axis=-1)
        din = self.din_bn(din, training=training)
        d1 = self.fcn_layer1(din)
        d2 = self.fcn_layer2(d1)
        d3 = self.fcn_layer3(d2)
        return d3

    def call_train(self, u, i, y, hist_i, sl, training=True):
        """Forward pass for training (single positive item)."""
        i_emb = self._get_item_emb(i)
        i_b = tf.gather(self.item_b, i)
        h_emb = self._get_hist_emb(hist_i)

        u_emb_i = self._user_representation(i_emb, h_emb, sl, training=training)

        d_layer_3_i = self._fcn_net(u_emb_i, i_emb, training=training)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])

        logits = i_b + d_layer_3_i
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
        return loss, logits

    def call_eval(self, u, i, j, hist_i, sl, training=False):
        """Forward pass for evaluation (positive + negative)."""
        i_emb = self._get_item_emb(i)
        i_b = tf.gather(self.item_b, i)
        j_emb = self._get_item_emb(j)
        j_b = tf.gather(self.item_b, j)
        h_emb = self._get_hist_emb(hist_i)

        u_emb_i = self._user_representation(i_emb, h_emb, sl, training=training)
        u_emb_j = self._user_representation(j_emb, h_emb, sl, training=training)

        d_layer_3_i = self._fcn_net(u_emb_i, i_emb, training=training)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = self._fcn_net(u_emb_j, j_emb, training=training)
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])

        x = i_b - j_b + d_layer_3_i - d_layer_3_j
        mf_auc = tf.reduce_mean(tf.cast(x > 0, tf.float32))

        score_i = tf.sigmoid(i_b + d_layer_3_i)
        score_j = tf.sigmoid(j_b + d_layer_3_j)
        score_i = tf.reshape(score_i, [-1, 1])
        score_j = tf.reshape(score_j, [-1, 1])
        p_and_n = tf.concat([score_i, score_j], axis=-1)

        return mf_auc, p_and_n

    def call_test(self, u, i, j, hist_i, sl, training=False):
        """Forward pass for batch prediction on sub-items."""
        h_emb = self._get_hist_emb(hist_i)

        # Build sub-item embeddings
        item_emb_all = tf.concat([
            self.item_emb_w,
            tf.nn.embedding_lookup(self.cate_emb_w, self.cate_list)
        ], axis=1)
        item_emb_sub = item_emb_all[:self.predict_ads_num, :]
        item_emb_sub = tf.expand_dims(item_emb_sub, 0)
        batch_size = tf.shape(hist_i)[0]
        item_emb_sub = tf.tile(item_emb_sub, [batch_size, 1, 1])

        hist_sub = self.attention_multi_layer([item_emb_sub, h_emb, sl])
        hist_sub = self.hist_bn(hist_sub, training=training)
        hist_sub = tf.reshape(hist_sub, [-1, self.hidden_units])
        hist_sub = self.hist_fcn(hist_sub)

        u_emb_sub = hist_sub
        item_emb_sub_flat = tf.reshape(item_emb_sub, [-1, self.hidden_units])
        d_layer_3_sub = self._fcn_net(u_emb_sub, item_emb_sub_flat, training=training)
        d_layer_3_sub = tf.reshape(d_layer_3_sub, [-1, self.predict_ads_num])
        logits_sub = tf.sigmoid(self.item_b[:self.predict_ads_num] + d_layer_3_sub)
        logits_sub = tf.reshape(logits_sub, [-1, self.predict_ads_num, 1])
        return logits_sub


# ==================== Utility Functions ====================

def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
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
    for s in score_p.numpy().tolist():
        score_arr.append([0, 1, s])
    for s in score_n.numpy().tolist():
        score_arr.append([1, 0, s])
    return score_arr


# ==================== Main Training ====================

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

train_batch_size = 32
test_batch_size = 512
predict_batch_size = 32
predict_users_num = 1000
predict_ads_num = 100

with open('/home/zhangbo.999/jupyter_workspace/dataset/amazon_review_data/amazon_2014/din_raw_data/electronics_dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

best_auc = 0.0

model = DINModel(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

# Build the model by running a dummy forward pass
dummy_hist = np.zeros([1, 1], dtype=np.int64)
dummy_i = np.array([0], dtype=np.int32)
dummy_sl = np.array([1], dtype=np.int32)
dummy_u = np.array([0], dtype=np.int32)
dummy_y = np.array([0.0], dtype=np.float32)
_ = model.call_train(dummy_u, dummy_i, dummy_y, dummy_hist, dummy_sl, training=False)

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


# ==================== 方案 2：关键函数加 @tf.function ====================

@tf.function
def train_step(u, i, y, hist_i, sl, lr):
    optimizer.learning_rate.assign(lr)
    with tf.GradientTape() as tape:
        loss, logits = model.call_train(u, i, y, hist_i, sl, training=True)
    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    return loss


@tf.function
def eval_step(u, i, j, hist_i, sl):
    """Eval forward in graph mode to reduce peak memory."""
    return model.call_eval(u, i, j, hist_i, sl, training=False)


@tf.function
def test_step(u, i, j, hist_i, sl):
    """Test forward in graph mode to reduce peak memory."""
    return model.call_test(u, i, j, hist_i, sl, training=False)


def _eval(model):
    auc_sum = 0.0
    score_arr = []
    for _, uij in DataInputTest(test_set, test_batch_size):
        u = tf.constant(uij[0], dtype=tf.int32)
        i = tf.constant(uij[1], dtype=tf.int32)
        j = tf.constant(uij[2], dtype=tf.int32)
        hist_i = tf.constant(uij[3], dtype=tf.int32)
        sl = tf.constant(uij[4], dtype=tf.int32)
        auc_, score_ = eval_step(u, i, j, hist_i, sl)
        score_arr += _auc_arr(score_)
        auc_sum += auc_.numpy() * len(uij[0])
    test_gauc = auc_sum / len(test_set)
    Auc = calc_auc(score_arr)
    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        checkpoint.save('save_path/ckpt')
    return test_gauc, Auc


def _test(model):
    predicted_users_num = 0
    score_arr = []
    print("test sub items")
    for _, uij in DataInputTest(test_set, predict_batch_size):
        if predicted_users_num >= predict_users_num:
            break
        u = tf.constant(uij[0], dtype=tf.int32)
        i = tf.constant(uij[1], dtype=tf.int32)
        j = tf.constant(uij[2], dtype=tf.int32)
        hist_i = tf.constant(uij[3], dtype=tf.int32)
        sl = tf.constant(uij[4], dtype=tf.int32)
        score_ = test_step(u, i, j, hist_i, sl)
        score_arr.append(score_)
        predicted_users_num += predict_batch_size
    return score_arr[-1][0]


# Initial evaluation
print('test_gauc: %.4f\t test_auc: %.4f' % _eval(model))
sys.stdout.flush()

lr = 1.0
start_time = time.time()
global_step = 0

for epoch in range(50):
    random.shuffle(train_set)
    loss_sum = 0.0

    for _, uij in DataInput(train_set, train_batch_size):
        u = tf.constant(uij[0], dtype=tf.int32)
        i = tf.constant(uij[1], dtype=tf.int32)
        y = tf.constant(uij[2], dtype=tf.float32)
        hist_i = tf.constant(uij[3], dtype=tf.int32)
        sl = tf.constant(uij[4], dtype=tf.int32)

        loss = train_step(u, i, y, hist_i, sl, lr)
        loss_sum += loss.numpy()
        global_step += 1

        if global_step % 1000 == 0:
            test_gauc, Auc = _eval(model)
            print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                  (epoch, global_step, loss_sum / 1000, test_gauc, Auc))
            sys.stdout.flush()
            loss_sum = 0.0

        if global_step % 336000 == 0:
            lr = 0.1

    print('Epoch %d DONE\tCost time: %.2f' % (epoch, time.time() - start_time))
    sys.stdout.flush()

print('best test_gauc:', best_auc)
sys.stdout.flush()
