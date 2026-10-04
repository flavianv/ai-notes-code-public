# Part 2: attention from scratch, exactly the code shown in the film (attention, MHA/GQA/MQA, pre-norm block, tiled FlashAttention).
import math
import torch, torch.nn as nn


def attention(q, k, v, causal=False):                  # q, k, v: (..., n, d)
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)    # (..., n, n): every query vs every key
    if causal:                                         # no peeking at future tokens
        n = q.size(-2)
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)                   # each row sums to 1
    return weights @ v                                 # weighted average of the values


class MultiHeadAttention(nn.Module):                   # n_kv_heads: = n_heads MHA, = 1 MQA, else GQA
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.h, self.kv = n_heads, n_kv_heads or n_heads
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, self.h * self.dh, bias=False)
        self.wk = nn.Linear(d_model, self.kv * self.dh, bias=False)
        self.wv = nn.Linear(d_model, self.kv * self.dh, bias=False)
        self.wo = nn.Linear(self.h * self.dh, d_model, bias=False)

    def forward(self, x, causal=True):
        B, n, _ = x.shape
        q = self.wq(x).view(B, n, self.h, self.dh).transpose(1, 2)     # (B, h, n, dh)
        k = self.wk(x).view(B, n, self.kv, self.dh).transpose(1, 2)    # (B, kv, n, dh)
        v = self.wv(x).view(B, n, self.kv, self.dh).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.kv, dim=1)              # a group of query heads
        v = v.repeat_interleave(self.h // self.kv, dim=1)              # shares one K and V
        out = attention(q, k, v, causal)                               # (B, h, n, dh)
        return self.wo(out.transpose(1, 2).reshape(B, n, -1))          # concat heads, mix


class Block(nn.Module):                                # a pre-norm transformer block
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))               # tokens talk to each other
        x = x + self.mlp(self.norm2(x))                # each token thinks on its own
        return x


def flash_attention(q, k, v, block=128):               # same result, never builds the n x n matrix
    n, d = q.shape[-2], q.shape[-1]
    out = torch.empty_like(q)
    for i in range(0, n, block):                       # one block of queries (kept in fast SRAM)
        qi = q[..., i:i + block, :] / math.sqrt(d)
        m = torch.full(qi.shape[:-1], float("-inf"), device=q.device)   # running max
        l = torch.zeros(qi.shape[:-1], device=q.device)                 # running sum
        acc = torch.zeros_like(qi)                                      # running output
        for j in range(0, n, block):                   # stream blocks of keys and values
            s = qi @ k[..., j:j + block, :].transpose(-2, -1)
            m_new = torch.maximum(m, s.amax(dim=-1))
            p = torch.exp(s - m_new[..., None])
            scale = torch.exp(m - m_new)               # rescale what we had so far
            l = l * scale + p.sum(dim=-1)
            acc = acc * scale[..., None] + p @ v[..., j:j + block, :]
            m = m_new
        out[..., i:i + block, :] = acc / l[..., None]
    return out
