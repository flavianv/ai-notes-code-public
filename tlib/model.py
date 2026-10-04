# A complete decoder-only transformer in plain PyTorch: RoPE, causal attention with a KV cache, pre-norm blocks, GPT.
import torch, torch.nn as nn, torch.nn.functional as F


# 2. Position: rotary embeddings rotate each (even, odd) pair of features by angle = position x frequency
def rope(x, pos, base=10000.0):                                          # x: (..., n, d), pos: (n,)
    d = x.size(-1)
    freqs = base ** (-torch.arange(0, d, 2, device=x.device) / d)       # one frequency per pair, fast to slow
    ang = pos[:, None] * freqs[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


# 3. The block: attention + MLP, each with a residual connection and a normalisation before it
class Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, start=0, cache=None):                           # start: position of x[:, 0]; cache: this layer's keys/values
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)   # each (B, heads, T, head_dim)
        pos = torch.arange(start, start + T, device=x.device, dtype=torch.float)
        q, k = rope(q, pos), rope(k, pos)                                # position enters here, on q and k only
        if cache is not None:                                            # KV cache: keep old keys and values, append the new ones
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        if T > 1 and k.size(2) > T:                                      # several new tokens on top of a cache (speculative decoding): the mask is offset
            mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(k.size(2) - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)    # causal mask: a token sees only earlier tokens
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))   # widen 4x, nonlinearity, project back

    def forward(self, x, start=0, cache=None):
        x = x + self.attn(self.n1(x), start, cache)                      # residual: add, don't replace
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)                              # token id -> vector
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)                      # vector -> one score (logit) per vocabulary entry

    def forward(self, ids, start=0, caches=None):                        # ids: (B, n) -> logits (B, n, vocab)
        x = self.embed(ids)
        for i, block in enumerate(self.blocks):
            x = block(x, start, None if caches is None else caches[i])
        return self.head(self.norm(x))


# 5. Training: next-token prediction at every position at once
def next_token_loss(model, ids):
    logits = model(ids[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
