# Part 3 (serving): KV-cache maths, mixture of experts, speculative decoding.
import torch
from tlib.model import GPT
from tlib.serving import kv_bytes, MoE, speculative


def test_kv_cache_memory_llama_like():
    gib = 2 ** 30
    assert kv_bytes(32, 8, 128, 8192) == 1 * gib                    # GQA, 8k tokens, fp16: 1 GiB per sequence
    assert kv_bytes(32, 32, 128, 8192) == 4 * gib                   # full multi-head: 4x
    assert kv_bytes(32, 1, 128, 8192) == gib // 8                   # multi-query: 1/8


def test_moe_sparse_dispatch_equals_dense_reference():
    torch.manual_seed(0)
    m = MoE(32, 8, 2, 64); x = torch.randn(2, 10, 32)
    assert (m(x) - m.forward_dense(x)).abs().max() < 1e-5


def test_speculative_greedy_is_exactly_the_target_greedy():
    torch.manual_seed(0)
    target, draft = GPT(30, 64, 4, 3).eval(), GPT(30, 32, 2, 1).eval()      # untrained, different models: still exact
    ids = torch.randint(0, 30, (1, 5))
    out, stats = speculative(target, draft, ids, 40, k=4, greedy=True)
    ref = ids
    with torch.no_grad():
        for _ in range(40): ref = torch.cat([ref, target(ref)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, ref) and stats["target_calls"] <= 40
