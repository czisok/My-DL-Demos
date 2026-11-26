# PyTorch Transformer 中英翻译

这是一个从零实现的 Encoder-Decoder Transformer 教学项目。核心模块没有调用
`torch.nn.Transformer` 或 `torch.nn.MultiheadAttention`，包括多头注意力、位置编码、
Encoder/Decoder Block、掩码、Noam 学习率、训练指标、BLEU 评估和贪心解码。

详细原理、每一步张量维度和公式见：

- [Transformer 原理与中英翻译实战](docs/transformer_guide.md)
- `docs/transformer_guide.docx`

## 环境安装

建议使用 Python 3.10 以上版本：

```bash
python -m pip install -r requirements.txt
```

## 快速训练

`quickstart` 配置使用 OPUS-100 `en-zh` 的 2,000 条训练样本和一个小模型，用于验证
完整流水线。首次运行会下载数据并训练英、中两个 SentencePiece BPE 分词器。

```bash
python train.py --config configs/quickstart.yaml
```

正式配置默认使用 100,000 条训练样本和论文同量级的模型结构：

```bash
python train.py --config configs/base.yaml
```

训练日志会输出 loss、perplexity、token accuracy、BLEU、学习率和累计耗时，并写入
`outputs/<run>/metrics.jsonl`。检查点位于 `outputs/<run>/best.pt` 和 `last.pt`。

中断后恢复：

```bash
python train.py \
  --config configs/base.yaml \
  --resume outputs/base/last.pt
```

只做快速训练冒烟测试时，可跳过较慢的自回归 BLEU：

```bash
python train.py --config configs/quickstart.yaml --skip-bleu
```

## 翻译推理

单句或多句：

```bash
python translate.py \
  --checkpoint outputs/quickstart/best.pt \
  --text "I love machine learning." \
  --text "Attention is all you need."
```

也支持标准输入：

```bash
printf "Hello world.\\nHow are you?\\n" | \
  python translate.py --checkpoint outputs/quickstart/best.pt
```

`quickstart` 只用于验证代码，样本和训练轮数太少，翻译质量不会好。需要有意义的结果，
应增加训练数据、训练轮数，并根据显存调整 `batch_size` 和梯度累积。

## 测试

```bash
pytest -q
```

测试覆盖注意力张量形状、Encoder-Decoder 输出形状、padding mask 和 causal mask。

## 工程结构

```text
.
├── configs/
│   ├── quickstart.yaml
│   └── base.yaml
├── transformer_mt/
│   ├── data.py          # OPUS-100、清洗、SentencePiece、DataLoader
│   ├── model.py         # 手写 Transformer 全部模块
│   ├── training.py      # loss、指标、BLEU、训练与检查点
│   ├── inference.py     # 自回归贪心解码
│   └── utils.py
├── train.py
├── translate.py
├── tests/
└── docs/
```

## 数据许可说明

OPUS-100 聚合了多个来源和领域，Hugging Face 数据卡的许可证字段为 `unknown`。
本项目用于教学和研究；用于产品或商业场景前，需要逐项核验上游语料的授权与合规要求。
