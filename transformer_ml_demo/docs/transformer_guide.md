# Transformer 模型原理、张量维度与中英翻译实战

## 1. 文档目标

本文从机器翻译任务出发，完整解释原始 Encoder-Decoder Transformer 的数据流、
Encoder Block、Decoder Block、多头注意力、掩码、前馈网络、损失计算、训练监控和
自回归推理。每个关键步骤都标明张量形状，并与本项目的 PyTorch 实现一一对应。

Transformer 由 Vaswani 等人在 2017 年提出。原始模型完全依赖注意力机制，不使用
循环网络或卷积，并以机器翻译作为主要实验任务。[cite:1] PyTorch 官方也把
`nn.Transformer` 定位为原始架构的基础参考实现；本项目为了教学透明度，手写了内部
模块，而没有直接调用该封装。[cite:5]

## 2. 统一符号

| 符号 | 含义 | 贯穿示例 |
|---|---|---:|
| $B$ | batch size | 32 |
| $S$ | 源语言序列长度 | 64 |
| $T$ | 目标语言序列长度 | 48 |
| $D=d_{model}$ | 模型隐藏维度 | 512 |
| $H$ | 注意力头数 | 8 |
| $d_k,d_v$ | 每个头的 Query/Key/Value 维度 | $512/8=64$ |
| $d_{ff}$ | 前馈网络中间维度 | 2048 |
| $V_s$ | 源语言词表大小 | 16000 |
| $V_t$ | 目标语言词表大小 | 16000 |
| $N_e,N_d$ | Encoder/Decoder Block 数量 | 6、6 |

实际 batch 内的 $S$ 和 $T$ 由当前 batch 最长样本决定，不要求二者相等。PAD 补齐只
用于形成矩形张量，padding mask 保证注意力不会读取补齐位置。

## 3. Transformer 总体结构

![原始 Transformer 经典结构图](../assets/transformer_architecture_original.png)

图 1：Vaswani 等人的原始 Transformer Figure 1。左侧是 Encoder Stack，右侧是
Decoder Stack。图片来自论文网页，原页面允许在注明出处的学术或新闻作品中复用图表。
[cite:1]

整体计算可以压缩为四步：

1. 英文 token IDs 经源语言 Embedding 和位置编码得到 $X\in\mathbb{R}^{B\times S\times D}$。
2. $N_e$ 个 Encoder Block 把 $X$ 编码为 memory
   $M\in\mathbb{R}^{B\times S\times D}$。
3. 右移后的中文 token IDs 经目标 Embedding 和位置编码得到
   $Y\in\mathbb{R}^{B\times T\times D}$，再经过 $N_d$ 个 Decoder Block。
4. 线性输出层把每个目标位置的 $D$ 维状态映射为 $V_t$ 类 logits：
   $L\in\mathbb{R}^{B\times T\times V_t}$。

Encoder 和 Decoder 都是“堆叠多个结构相同但参数不共享的 Block”。原论文使用 6 层、
$D=512$、$H=8$、$d_{ff}=2048$；这些数值是设计选择，不是架构硬约束。
[cite:1][cite:4]

## 4. 输入数据与分词

### 4.1 平行语料

项目默认使用 `Helsinki-NLP/opus-100` 的 `en-zh` 配置。OPUS-100 是以英语为中心、
覆盖 100 种语言的平行语料集合，来自多个 OPUS 子语料和多个领域。官方说明其每个语言
对最多随机抽取 100 万个训练句对、2000 个验证句对和 2000 个测试句对；`en-zh`
配置正好包含 100 万/2000/2000 条。[cite:2][cite:3]

选择它的原因：

- 已提供标准训练、验证和测试划分，避免手工切分造成数据泄漏。
- 中英训练集足够大，同时可通过 `max_train_samples` 截取小子集做教学实验。
- Hugging Face `datasets` 可以自动下载和缓存，运行入口简单。
- 数据领域混杂，适合学习完整工程，但不是高质量商业翻译基准。

重要限制：Hugging Face 数据卡将许可证标记为 `unknown`，而且 OPUS-100 聚合多个上游
来源。研究和教学可按数据卡说明使用；商业使用前必须核验具体上游语料授权。[cite:3]

### 4.2 SentencePiece BPE

训练集分别训练英文和中文两个 SentencePiece BPE 模型，特殊 token 固定为：

| token | ID | 用途 |
|---|---:|---|
| `<pad>` | 0 | batch 补齐，损失与注意力均忽略 |
| `<unk>` | 1 | 未知子词 |
| `<bos>` | 2 | 目标序列开始；源序列也统一添加 |
| `<eos>` | 3 | 序列结束 |

对英文 `"I love NLP."`，假设分词后为 4 个子词，则加入边界 token 后：

$$
\text{source ids}=[2,\ i_1,\ i_2,\ i_3,\ i_4,\ 3]
$$

单样本形状为 $[S_i]$；collator 按 batch 最长长度补 PAD，得到
$[B,S]$。中文目标同理得到 $[B,T_{full}]$。

## 5. Embedding 与位置编码

### 5.1 Token Embedding

源词表参数矩阵为 $E_s\in\mathbb{R}^{V_s\times D}$：

$$
[B,S]\xrightarrow{\text{lookup }E_s}[B,S,D]
$$

贯穿示例：

$$
[32,64]\rightarrow[32,64,512]
$$

目标 Embedding 独立使用 $E_t\in\mathbb{R}^{V_t\times D}$：

$$
[B,T]\rightarrow[B,T,D]=[32,48,512]
$$

代码按照原论文把 embedding 乘以 $\sqrt{D}$，使其数值尺度与位置编码更匹配。

### 5.2 正弦位置编码

注意力本身不包含顺序概念，因此给第 $pos$ 个位置加入确定性向量：

$$
PE(pos,2i)=\sin\left(pos/10000^{2i/D}\right)
$$

$$
PE(pos,2i+1)=\cos\left(pos/10000^{2i/D}\right)
$$

位置编码形状为 $[1,L,D]$，第 1 维可以广播到整个 batch：

$$
[B,L,D]+[1,L,D]\rightarrow[B,L,D]
$$

因此位置编码只注入位置信息，不改变张量形状。

## 6. Scaled Dot-Product Attention

Attention 接受 Query、Key 和 Value。先看单个头：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

若 Query 长度为 $L_q$，Key/Value 长度为 $L_k$：

| 步骤 | 张量形状 |
|---|---|
| $Q$ | $[B,H,L_q,d_k]$ |
| $K$ | $[B,H,L_k,d_k]$ |
| $V$ | $[B,H,L_k,d_v]$ |
| $K^\top$ | $[B,H,d_k,L_k]$ |
| $QK^\top$ | $[B,H,L_q,L_k]$ |
| mask 后的 scores | $[B,H,L_q,L_k]$ |
| softmax 后的权重 | $[B,H,L_q,L_k]$ |
| 权重乘 $V$ | $[B,H,L_q,d_v]$ |

除以 $\sqrt{d_k}$ 是因为 $d_k$ 增大时点积方差随之增大，未经缩放的分数容易把
softmax 推入梯度很小的饱和区。[cite:1]

Softmax 沿最后一维 $L_k$ 计算，所以每个 Query 对所有 Key 的注意力权重之和为 1。

## 7. Multi-Head Attention 的完整维度

输入 `query`、`key`、`value` 最初都以 $D$ 维表示。以 $Q$ 为例：

### 7.1 线性投影

$$
[B,L_q,D]\times[D,D]\rightarrow[B,L_q,D]
$$

示例：

$$
[32,64,512]\rightarrow[32,64,512]
$$

$K,V$ 分别经过自己的参数矩阵，形状规则相同。

### 7.2 拆分注意力头

因为 $D=H\cdot d_k$：

$$
[B,L_q,D]\rightarrow[B,L_q,H,d_k]
\rightarrow[B,H,L_q,d_k]
$$

示例：

$$
[32,64,512]\rightarrow[32,64,8,64]
\rightarrow[32,8,64,64]
$$

第二个箭头是 `transpose(1, 2)`，将头维移到序列维之前，便于批量矩阵乘。

### 7.3 每个头独立注意力

Encoder 自注意力中 $L_q=L_k=S=64$：

$$
[32,8,64,64]\times[32,8,64,64]^\top
\rightarrow[32,8,64,64]
$$

最后两个 64 含义不同：倒数第二维是 Query 位置，最后一维是 Key 位置。

权重再乘 $V$：

$$
[32,8,64,64]\times[32,8,64,64]
\rightarrow[32,8,64,64]
$$

前一个张量最后两维是 $[S,S]$，后一个是 $[S,d_v]$，结果为 $[S,d_v]$。

### 7.4 合并注意力头

$$
[B,H,L_q,d_v]\rightarrow[B,L_q,H,d_v]
\rightarrow[B,L_q,Hd_v]=[B,L_q,D]
$$

示例：

$$
[32,8,64,64]\rightarrow[32,64,8,64]
\rightarrow[32,64,512]
$$

最后再经过 $W^O\in\mathbb{R}^{D\times D}$，形状仍是 $[B,L_q,D]$。多头机制让不同头
在不同表示子空间和位置关系上并行学习；各头不是独立模型，拼接后还要由 $W^O$ 混合。
[cite:1]

## 8. Encoder Block

原始 Post-LN Encoder Block 包含两个子层：

1. Multi-Head Self-Attention。
2. Position-wise Feed-Forward Network。

每个子层外都有残差连接、Dropout 和 LayerNorm。本项目默认与论文一致使用 Post-LN：

$$
x_1=\operatorname{LayerNorm}(x+\operatorname{Dropout}(\operatorname{MHA}(x)))
$$

$$
x_2=\operatorname{LayerNorm}(x_1+\operatorname{Dropout}(\operatorname{FFN}(x_1)))
$$

### 8.1 Encoder Self-Attention

输入 $x:[B,S,D]$，并令 $Q=K=V=x$：

$$
[32,64,512]\rightarrow[32,64,512]
$$

注意力权重为 $[B,H,S,S]=[32,8,64,64]$。每个源 token 都可以读取所有非 PAD 源
token，不需要因果遮挡。

### 8.2 源语言 Padding Mask

源 IDs 为 $[B,S]$。判断 `id != PAD_ID` 后扩展：

$$
[B,S]\rightarrow[B,1,1,S]
$$

在注意力分数 $[B,H,S,S]$ 上广播。mask 中 `False` 的 Key 位置在 softmax 前被设为
极小值，因此其概率近似 0。只需要屏蔽 Key 维；PAD Query 的输出即使存在，也不会参与
目标损失。

### 8.3 残差和 LayerNorm

注意力输出与输入都是 $[B,S,D]$，所以可以逐元素相加：

$$
[B,S,D]+[B,S,D]\rightarrow[B,S,D]
$$

LayerNorm 在最后一个维度 $D$ 上计算每个 token 的均值和方差，形状保持不变。
所有子层都输出 $D$ 维，正是残差连接能够成立的必要条件。

### 8.4 Position-wise FFN

每个位置独立使用同一组两层 MLP 参数：

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

维度变化：

$$
[B,S,D]\times[D,d_{ff}]\rightarrow[B,S,d_{ff}]
$$

$$
[B,S,d_{ff}]\times[d_{ff},D]\rightarrow[B,S,D]
$$

示例：

$$
[32,64,512]\rightarrow[32,64,2048]\rightarrow[32,64,512]
$$

“Position-wise”表示不同位置互不混合，但所有位置共享 $W_1,W_2$。位置之间的信息交换
已经由注意力完成，FFN 负责每个位置内部的非线性特征变换。

### 8.5 Encoder Stack

每个 Block 输入输出都是 $[B,S,D]$，所以可以堆叠 $N_e$ 层：

$$
[B,S,D]\xrightarrow{N_e\text{ blocks}}[B,S,D]
$$

最终输出称为 `memory`，它保留每个源位置的表示，不是把整句压缩成单个向量。

## 9. Decoder Block

Decoder Block 比 Encoder 多一个交叉注意力子层：

1. Masked Multi-Head Self-Attention。
2. Encoder-Decoder Cross-Attention。
3. Position-wise FFN。

### 9.1 目标序列右移

若真实中文序列为：

$$
[BOS,y_1,y_2,\ldots,y_n,EOS]
$$

训练时构造：

$$
\text{decoder input}=[BOS,y_1,\ldots,y_n]
$$

$$
\text{label}=[y_1,y_2,\ldots,y_n,EOS]
$$

若原目标张量是 $[B,T_{full}]$，切片后输入和标签都是
$[B,T_{full}-1]$。本文后续把这个长度记为 $T$。

### 9.2 Decoder Causal Self-Attention

$Q,K,V$ 全部来自 Decoder 当前状态 $[B,T,D]$。权重形状：

$$
[B,H,T,T]
$$

示例为 $[32,8,48,48]$。为了防止位置 $t$ 偷看未来标签，使用下三角 mask：

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
$$

代码还把目标 padding mask $[B,1,1,T]$ 与因果 mask $[1,1,T,T]$ 做逻辑与，广播后
得到 $[B,1,T,T]$。因此 Decoder 既不能看未来，也不能把 PAD 当作有效历史。

### 9.3 Encoder-Decoder Cross-Attention

这是理解 Encoder-Decoder 连接的关键：

- Query 来自 Decoder：$Q:[B,T,D]$。
- Key 和 Value 来自 Encoder memory：$K,V:[B,S,D]$。

拆头后：

$$
Q:[B,H,T,d_k],\quad K,V:[B,H,S,d_k]
$$

打分：

$$
[B,H,T,d_k]\times[B,H,d_k,S]
\rightarrow[B,H,T,S]
$$

贯穿示例：

$$
[32,8,48,64]\times[32,8,64,64]
\rightarrow[32,8,48,64]
$$

结果最后一维的 64 是源长度 $S$，不是每头维度。随后：

$$
[B,H,T,S]\times[B,H,S,d_v]
\rightarrow[B,H,T,d_v]
$$

合并头后回到 $[B,T,D]=[32,48,512]$。含义是：每个中文生成位置都可以对全部英文
位置分配注意力，从 Encoder memory 读取当前翻译所需的信息。

Cross-Attention 使用源 padding mask $[B,1,1,S]$；它不使用因果 mask，因为完整源句
在翻译开始前已经可见。

### 9.4 Decoder FFN 与输出

FFN 的维度规则与 Encoder 相同：

$$
[B,T,D]\rightarrow[B,T,d_{ff}]\rightarrow[B,T,D]
$$

经过 $N_d$ 个 Decoder Block 后仍为 $[B,T,D]$。

## 10. 输出层、Softmax 与损失

输出投影参数为 $W_o\in\mathbb{R}^{D\times V_t}$：

$$
[B,T,D]\times[D,V_t]\rightarrow[B,T,V_t]
$$

示例：

$$
[32,48,512]\rightarrow[32,48,16000]
$$

每个 $[V_t]$ 向量是某个目标位置对整个中文词表的未归一化 logits。训练使用带
label smoothing 的交叉熵：

$$
\mathcal{L}
=-\frac{1}{N}\sum_{i=1}^{N}\sum_{v=1}^{V_t}q_{i,v}\log p_{i,v}
$$

其中 $q$ 不再是严格 one-hot；正确类别保留大部分概率质量，少量质量分配给其他类别。
原论文使用 $\epsilon_{ls}=0.1$。[cite:1]

计算前把：

$$
\text{logits }[B,T,V_t]\rightarrow[BT,V_t]
$$

$$
\text{labels }[B,T]\rightarrow[BT]
$$

`ignore_index=PAD_ID` 确保目标 PAD 不计入损失。项目将目标 Embedding 权重与输出投影
权重共享，减少参数量并遵循原论文的权重共享设计。

## 11. 训练流程

### 11.1 单个 batch

1. `source_ids: [B,S]` 和 `target_ids: [B,T_full]` 移到设备。
2. 构造 `decoder_input=target_ids[:,:-1]` 与 `labels=target_ids[:,1:]`。
3. 前向得到 `logits: [B,T,V_t]`。
4. 计算忽略 PAD 的 label-smoothed cross entropy。
5. 反向传播；可按配置进行梯度累积。
6. 梯度裁剪后更新 Adam 参数。
7. 按 Noam 规则更新学习率。

### 11.2 优化器与 Noam 学习率

使用论文参数的 Adam：$\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$。学习率：

$$
lr=factor\cdot D^{-0.5}
\cdot\min(step^{-0.5},step\cdot warmup^{-1.5})
$$

warmup 阶段线性上升，之后按 $step^{-0.5}$ 衰减。`base.yaml` 使用 4000 warmup
steps，`quickstart.yaml` 为了小实验缩短到 400。

### 11.3 监控指标

| 指标 | 计算方式 | 解释 |
|---|---|---|
| loss | 忽略 PAD 的平均交叉熵 | 优化目标，越低越好 |
| perplexity | $\exp(loss)$ | 平均不确定性的指数表达 |
| token accuracy | 非 PAD 位置 argmax 正确比例 | 直观但不衡量整句质量 |
| BLEU | 解码文本与参考译文的语料级 n-gram 指标 | 更接近翻译质量 |
| learning rate | 当前 Noam 学习率 | 诊断 warmup 和恢复训练 |
| elapsed time | 从训练开始的累计耗时 | 监控长周期任务 |

训练日志同时打印到终端并追加到 `metrics.jsonl`。验证 loss 使用 teacher forcing；
BLEU 使用真实自回归解码，计算更慢但更符合部署行为。

## 12. 推理过程的维度

推理时没有真实中文标签，必须逐 token 生成：

1. 英文 IDs $[B,S]$ 只编码一次，得到 memory $[B,S,D]$。
2. 初始目标为 `<bos>`，形状 $[B,1]$。
3. Decoder 输出 $[B,1,D]$，投影为 $[B,1,V_t]$。
4. 取最后位置 argmax 得到 $[B]$ 的下一个 token。
5. 拼接后目标变为 $[B,2]$，重复解码。
6. 所有样本生成 `<eos>` 或达到 `max_length` 后停止。

在第 $t$ 步，Decoder 输入长度为 $t$，自注意力矩阵为 $[B,H,t,t]$，交叉注意力矩阵为
$[B,H,t,S]$。当前实现采用贪心搜索；它简单且完全可运行，但 beam search 通常能改善
翻译结果。

## 13. 三类注意力对比

| 位置 | Query | Key/Value | 分数形状 | mask |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder $[B,S,D]$ | Encoder $[B,S,D]$ | $[B,H,S,S]$ | 源 PAD |
| Decoder Self-Attention | Decoder $[B,T,D]$ | Decoder $[B,T,D]$ | $[B,H,T,T]$ | 目标 PAD + causal |
| Decoder Cross-Attention | Decoder $[B,T,D]$ | Encoder $[B,S,D]$ | $[B,H,T,S]$ | 源 PAD |

最容易混淆的是 Cross-Attention：输出长度由 Query 决定，所以输出仍是 $T$ 个目标位置；
读取范围由 Key/Value 决定，所以注意力最后一维是源长度 $S$。

## 14. 计算复杂度与内存

Self-Attention 的主要注意力矩阵是 $L\times L$，时间和显存复杂度随序列长度近似为
$O(L^2)$。Encoder 为 $S^2$，Decoder 自注意力为 $T^2$，Cross-Attention 为 $TS$。
因此长句训练的显存增长主要来自注意力矩阵，而不是 token IDs。

FFN 主要复杂度约为 $O(BLd_{model}d_{ff})$。在常用配置中，FFN 参数和计算占比也很高，
不能只把 Transformer 理解为注意力层。

## 15. 代码映射

| 原理模块 | 代码位置 |
|---|---|
| 数据下载、清洗、切分限制 | `transformer_mt/data.py::load_parallel_data` |
| SentencePiece 训练与编码 | `transformer_mt/data.py::SentencePieceTokenizer` |
| batch padding | `transformer_mt/data.py::TranslationCollator` |
| 正弦位置编码 | `transformer_mt/model.py::SinusoidalPositionalEncoding` |
| 多头注意力 | `transformer_mt/model.py::MultiHeadAttention` |
| FFN | `transformer_mt/model.py::PositionwiseFeedForward` |
| Encoder Block | `transformer_mt/model.py::EncoderLayer` |
| Decoder Block | `transformer_mt/model.py::DecoderLayer` |
| mask 构造 | `Transformer.make_source_mask/make_target_mask` |
| Noam 调度 | `transformer_mt/utils.py::NoamScheduler` |
| loss、指标、BLEU | `transformer_mt/training.py` |
| 训练入口 | `train.py` |
| 贪心解码与推理入口 | `transformer_mt/inference.py`、`translate.py` |

## 16. 运行方法

安装依赖：

```bash
python -m pip install -r requirements.txt
```

快速验证完整流水线：

```bash
python train.py --config configs/quickstart.yaml
```

使用 10 万条数据和标准规模模型：

```bash
python train.py --config configs/base.yaml
```

恢复训练：

```bash
python train.py --config configs/base.yaml --resume outputs/base/last.pt
```

推理：

```bash
python translate.py \
  --checkpoint outputs/quickstart/best.pt \
  --text "Attention is all you need."
```

运行测试：

```bash
pytest -q
```

## 17. 实验解释与改进方向

`quickstart` 的目标是确认数据、分词、模型、loss、检查点和 infer 全部打通，不应把其
BLEU 当作模型能力结论。仅 2000 个句对和 3 个 epoch 很容易欠拟合。

获得更合理结果时，建议依次调整：

1. 增加 `max_train_samples`，先到 10 万，再根据资源使用完整 100 万。
2. 增加训练轮数，并观察验证 loss 和 BLEU，而不是只看训练 loss。
3. 根据显存调整 batch size 和梯度累积，保持较稳定的有效 batch token 数。
4. 加入按 token 数动态组 batch，减少 PAD 浪费。
5. 推理从贪心搜索升级为 beam search，并增加 length penalty。
6. 对训练集做语言识别、重复句、极端长度和错位句对清洗。
7. 大规模训练可改用 Flash Attention、混合精度和分布式数据并行。

## 18. 结论

Encoder 的职责是把 $[B,S]$ 源 token 转成上下文化 memory $[B,S,D]$；Decoder 使用
因果自注意力建模已生成目标前缀，再用 Cross-Attention 读取源 memory，最终输出
$[B,T,V_t]$ 的词表 logits。整个结构的核心不只是注意力公式，而是三类注意力的
Query/Key/Value 来源、mask 语义以及残差连接要求所有子层保持 $D$ 维。

只要牢牢记住“注意力分数形状永远是 $[B,H,L_q,L_k]$”，Encoder 的 $S\times S$、
Decoder 的 $T\times T$ 和 Cross-Attention 的 $T\times S$ 就可以统一推导。

## Sources

- [cite:1] Vaswani et al., “Attention Is All You Need”:
  https://arxiv.org/html/1706.03762v7
- [cite:2] OPUS-100 official corpus page:
  https://opus.nlpl.eu/OPUS-100
- [cite:3] Hugging Face, `Helsinki-NLP/opus-100` dataset card:
  https://huggingface.co/datasets/Helsinki-NLP/opus-100
- [cite:4] Harvard NLP, “The Annotated Transformer”:
  http://nlp.seas.harvard.edu/2018/04/01/attention.html
- [cite:5] PyTorch official Transformer documentation:
  https://docs.pytorch.org/docs/main/generated/torch.nn.Transformer.html
