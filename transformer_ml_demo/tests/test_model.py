import torch

from transformer_mt.model import (
    MultiHeadAttention,
    Transformer,
    TransformerConfig,
)


def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        source_vocab_size=50,
        target_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        max_length=32,
    )


def test_multi_head_attention_shapes() -> None:
    attention = MultiHeadAttention(d_model=32, num_heads=4, dropout=0.0)
    query = torch.randn(2, 5, 32)
    key_value = torch.randn(2, 7, 32)
    mask = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    output, weights = attention(
        query, key_value, key_value, mask, need_weights=True
    )
    assert output.shape == (2, 5, 32)
    assert weights is not None
    assert weights.shape == (2, 4, 5, 7)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6, rtol=1e-6
    )


def test_transformer_output_and_mask_shapes() -> None:
    model = Transformer(tiny_config())
    source = torch.tensor([[2, 5, 6, 3, 0], [2, 8, 3, 0, 0]])
    target = torch.tensor([[2, 9, 10, 3], [2, 11, 3, 0]])
    logits = model(source, target)
    assert logits.shape == (2, 4, 60)
    assert model.make_source_mask(source).shape == (2, 1, 1, 5)
    assert model.make_target_mask(target).shape == (2, 1, 4, 4)


def test_causal_mask_blocks_future_information() -> None:
    torch.manual_seed(7)
    model = Transformer(tiny_config()).eval()
    source = torch.tensor([[2, 4, 5, 3]])
    target_a = torch.tensor([[2, 10, 11, 12]])
    target_b = torch.tensor([[2, 10, 45, 46]])
    logits_a = model(source, target_a)
    logits_b = model(source, target_b)
    torch.testing.assert_close(
        logits_a[:, :2], logits_b[:, :2], atol=1e-5, rtol=1e-5
    )


def test_padding_embedding_stays_zero_after_initialization() -> None:
    model = Transformer(tiny_config())
    assert torch.count_nonzero(model.source_embedding.weight[0]) == 0
    assert torch.count_nonzero(model.target_embedding.weight[0]) == 0
