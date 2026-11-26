from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    source_vocab_size: int
    target_vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_length: int = 512
    pad_id: int = 0
    norm_first: bool = False


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, sequence, d_model]; position encoding keeps this shape.
        # 中文：位置编码逐元素相加，不改变 [B, L, D]。
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds configured maximum "
                f"{self.encoding.size(1)}"
            )
        return self.dropout(x + self.encoding[:, : x.size(1)].to(x.dtype))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        # [B, L, D] -> [B, L, H, d_k] -> [B, H, L, d_k].
        # 中文：把模型维 D 拆成 H 个头，每头维度 d_k=D/H。
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        # [B, H, L, d_v] -> [B, L, H, d_v] -> [B, L, D].
        # 中文：拼回所有注意力头，H*d_v=D。
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch, length, self.d_model)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self._split_heads(self.query_projection(query))
        k = self._split_heads(self.key_projection(key))
        v = self._split_heads(self.value_projection(value))

        # [B,H,Lq,d_k] @ [B,H,d_k,Lk] -> [B,H,Lq,Lk].
        # 中文：每个目标位置对每个键位置产生一个相关性分数。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)

        # [B,H,Lq,Lk] @ [B,H,Lk,d_v] -> [B,H,Lq,d_v].
        context = torch.matmul(attention, v)
        output = self.output_projection(self._merge_heads(context))
        return output, attention if need_weights else None


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, D] -> [B, L, d_ff] -> [B, L, D].
        # 中文：同一个前馈网络独立作用于每个 token，序列长度不变。
        return self.network(x)


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm_first = config.norm_first

    def forward(
        self, x: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.norm_first:
            normalized = self.norm1(x)
            attended, _ = self.self_attention(
                normalized, normalized, normalized, source_mask
            )
            x = x + self.dropout1(attended)
            x = x + self.dropout2(self.feed_forward(self.norm2(x)))
            return x

        attended, _ = self.self_attention(x, x, x, source_mask)
        x = self.norm1(x + self.dropout1(attended))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.dropout
        )
        self.cross_attention = MultiHeadAttention(
            config.d_model, config.num_heads, config.dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.norm_first = config.norm_first

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_first:
            normalized = self.norm1(x)
            attended, _ = self.self_attention(
                normalized, normalized, normalized, target_mask
            )
            x = x + self.dropout1(attended)
            attended, _ = self.cross_attention(
                self.norm2(x), memory, memory, source_mask
            )
            x = x + self.dropout2(attended)
            x = x + self.dropout3(self.feed_forward(self.norm3(x)))
            return x

        attended, _ = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout1(attended))
        attended, _ = self.cross_attention(x, memory, memory, source_mask)
        x = self.norm2(x + self.dropout2(attended))
        return self.norm3(x + self.dropout3(self.feed_forward(x)))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(
            config.source_vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.target_embedding = nn.Embedding(
            config.target_vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.position = SinusoidalPositionalEncoding(
            config.d_model, config.max_length, config.dropout
        )
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
        self.encoder_norm = (
            nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()
        )
        self.decoder_norm = (
            nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()
        )
        self.output_projection = nn.Linear(
            config.d_model, config.target_vocab_size, bias=False
        )
        self.output_projection.weight = self.target_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.source_embedding.weight[self.config.pad_id].zero_()
            self.target_embedding.weight[self.config.pad_id].zero_()

    def make_source_mask(self, source_ids: torch.Tensor) -> torch.Tensor:
        # [B, S] -> [B, 1, 1, S], broadcast over heads and query positions.
        return source_ids.ne(self.config.pad_id).unsqueeze(1).unsqueeze(2)

    def make_target_mask(self, target_ids: torch.Tensor) -> torch.Tensor:
        batch, target_length = target_ids.shape
        padding = target_ids.ne(self.config.pad_id).view(
            batch, 1, 1, target_length
        )
        causal = torch.tril(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=target_ids.device,
            )
        ).view(1, 1, target_length, target_length)
        # [B,1,1,T] & [1,1,T,T] -> [B,1,T,T].
        # 中文：同时屏蔽 PAD 键和未来 token。
        return padding & causal

    def encode(
        self, source_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_mask = self.make_source_mask(source_ids)
        x = self.source_embedding(source_ids) * math.sqrt(self.config.d_model)
        x = self.position(x)
        for layer in self.encoder_layers:
            x = layer(x, source_mask)
        return self.encoder_norm(x), source_mask

    def decode(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_mask = self.make_target_mask(target_ids)
        x = self.target_embedding(target_ids) * math.sqrt(self.config.d_model)
        x = self.position(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, target_mask, source_mask)
        return self.decoder_norm(x)

    def forward(
        self, source_ids: torch.Tensor, target_input_ids: torch.Tensor
    ) -> torch.Tensor:
        # source [B,S], target [B,T] -> logits [B,T,V_target].
        memory, source_mask = self.encode(source_ids)
        decoder_output = self.decode(target_input_ids, memory, source_mask)
        return self.output_projection(decoder_output)

    def export_config(self) -> dict[str, int | float | bool]:
        return asdict(self.config)
