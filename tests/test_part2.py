# Every correctness claim in part 2 (attention from scratch).
import torch, torch.nn.functional as F
from tlib.attention import attention, MultiHeadAttention, Block, flash_attention

torch.manual_seed(0)


def test_attention_matches_pytorch_sdpa():
    q, k, v = (torch.randn(2, 8, 256, 64) for _ in range(3))
    for causal in (False, True):
        assert (attention(q, k, v, causal) - F.scaled_dot_product_attention(q, k, v, is_causal=causal)).abs().max() < 1e-5


def test_tiled_flash_attention_equals_plain_attention():
    q, k, v = (torch.randn(2, 8, 256, 64) for _ in range(3))
    assert (flash_attention(q, k, v, block=64) - attention(q, k, v)).abs().max() < 1e-5


def test_mha_gqa_mqa_match_sdpa_with_repeated_heads():
    x = torch.randn(2, 128, 512)
    for kv in (8, 2, 1):
        m = MultiHeadAttention(512, 8, kv); B, n, _ = x.shape
        q = m.wq(x).view(B, n, 8, 64).transpose(1, 2)
        k = m.wk(x).view(B, n, kv, 64).transpose(1, 2).repeat_interleave(8 // kv, 1)
        v = m.wv(x).view(B, n, kv, 64).transpose(1, 2).repeat_interleave(8 // kv, 1)
        ref = m.wo(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, n, -1))
        assert (m(x) - ref).abs().max() < 1e-5


def test_fewer_kv_heads_means_fewer_parameters_and_cache():
    p = lambda kv: sum(t.numel() for t in MultiHeadAttention(512, 8, kv).parameters())
    assert p(8) > p(2) > p(1)


def test_block_keeps_the_shape():
    assert Block(512, 8, 2)(torch.randn(2, 128, 512)).shape == (2, 128, 512)


def test_causal_mask_gives_future_tokens_zero_weight():
    x = torch.randn(1, 10, 64); m = MultiHeadAttention(64, 4).eval(); y1 = m(x)
    x2 = x.clone(); x2[0, 7] += 1.0
    assert (m(x2)[:, :7] - y1[:, :7]).abs().max() < 1e-6
