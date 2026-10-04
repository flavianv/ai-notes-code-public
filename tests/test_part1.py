# Every number shown in part 1 of the series is checked here.
import torch
from tlib import bpe_train, bpe_encode, rope, GPT, next_token_loss


def test_bpe_merges_and_encoding():
    merges = bpe_train({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 4)
    assert merges == [("e", "s"), ("es", "t"), ("l", "o"), ("lo", "w")]
    assert bpe_encode("newest", merges) == ["n", "e", "w", "est"]
    assert bpe_encode("lowest", merges) == ["low", "est"]


def test_softmax_temperature_numbers():
    z = torch.tensor([2.0, 1.0, 0.1])
    assert [round(v, 2) for v in z.softmax(-1).tolist()] == [0.66, 0.24, 0.10]
    assert round((z / 0.5).softmax(-1)[0].item(), 2) == 0.86


def test_rope_depends_only_on_offset():
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)
    dot = lambda m, n: float(rope(q[None], torch.tensor([float(m)]))[0] @ rope(k[None], torch.tensor([float(n)]))[0])
    assert abs(dot(5, 3) - dot(105, 103)) < 1e-3
    assert abs(dot(5, 3) - dot(5, 4)) > 0.1


def test_causal_mask_hides_the_future():
    torch.manual_seed(0)
    m = GPT(50).eval()
    a = torch.randint(0, 50, (1, 10)); b = a.clone(); b[0, 7] = (b[0, 7] + 1) % 50
    with torch.no_grad():
        assert (m(a)[:, :7] - m(b)[:, :7]).abs().max() < 1e-5


def test_kv_cache_matches_full_pass():
    torch.manual_seed(0)
    m = GPT(50).eval(); a = torch.randint(0, 50, (1, 10))
    with torch.no_grad():
        full = m(a); caches = [{} for _ in m.blocks]
        steps = torch.cat([m(a[:, t:t + 1], t, caches) for t in range(10)], 1)
    assert (steps - full).abs().max() < 1e-4


def test_starting_loss_is_log_vocab():
    torch.manual_seed(0)
    m = GPT(65); ids = torch.randint(0, 65, (8, 33))
    with torch.no_grad(): assert abs(next_token_loss(m, ids).item() - 4.17) < 0.5
