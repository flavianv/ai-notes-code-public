# Serving tricks in plain PyTorch: KV-cache memory maths, a mixture-of-experts layer, and speculative decoding.
import math
import torch, torch.nn as nn, torch.nn.functional as F
from .model import Attention, GPT


# 1. KV-cache memory: 2 (keys and values) x layers x kv_heads x head_dim x tokens x bytes per number
def kv_bytes(layers, kv_heads, head_dim, tokens, batch=1, nbytes=2):
    return 2 * layers * kv_heads * head_dim * tokens * batch * nbytes


# 2. Mixture of experts: a router picks the top-k experts per token; only those experts run
class MoE(nn.Module):
    def __init__(self, d, n_experts=8, k=2, hidden=None):
        super().__init__()
        self.k, self.n = k, n_experts
        hidden = hidden or 2 * d
        self.router = nn.Linear(d, n_experts, bias=False)               # one score per expert, per token
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)) for _ in range(n_experts))

    def route(self, flat):
        probs = self.router(flat).softmax(-1)
        w, idx = probs.topk(self.k, -1)                                  # the k best experts for each token
        return probs, w / w.sum(-1, keepdim=True), idx                   # weights renormalised over the chosen k

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs, w, idx = self.route(flat)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)                # which tokens chose expert e
            if tok.numel():
                out.index_add_(0, tok, expert(flat[tok]) * w[tok, slot, None])   # run it on those tokens only
        f = F.one_hot(idx, self.n).sum((0, 1)).float() / idx.numel()
        self.load = f.detach()                                           # share of assignments per expert
        self.aux = self.n * (f * probs.mean(0)).sum()                    # load-balancing loss: smallest when the load is even
        return out.reshape(B, T, D)

    def forward_dense(self, x):                                          # reference: run every expert on every token, keep the chosen ones
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        _, w, idx = self.route(flat)
        full = torch.zeros(flat.size(0), self.n, device=x.device).scatter(1, idx, w)
        return sum(full[:, e:e + 1] * ex(flat) for e, ex in enumerate(self.experts)).reshape(B, T, D)


class MoEBlock(nn.Module):
    def __init__(self, d, n_heads, n_experts, k, hidden):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads)
        self.mlp = MoE(d, n_experts, k, hidden) if n_experts > 1 else nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x, start=0, cache=None):
        x = x + self.attn(self.n1(x), start, cache)
        return x + self.mlp(self.n2(x))


class MoEGPT(GPT):                                                       # the same GPT with the MLP in every block swapped for MoE (n_experts=1: a dense MLP)
    def __init__(self, vocab, d=128, n_heads=4, n_layers=4, n_experts=8, k=2, hidden=256):
        super().__init__(vocab, d, n_heads, n_layers)
        self.blocks = nn.ModuleList(MoEBlock(d, n_heads, n_experts, k, hidden) for _ in range(n_layers))

    def aux_loss(self):
        return sum(b.mlp.aux for b in self.blocks if isinstance(b.mlp, MoE))


# 3. Speculative decoding: a small draft model proposes k tokens, the big model checks them all in ONE pass
def cache_len(caches): return caches[0]["k"].size(2) if "k" in caches[0] else 0
def cache_cut(caches, n):
    for c in caches:
        if "k" in c: c["k"], c["v"] = c["k"][:, :, :n], c["v"][:, :, :n]

def probs_of(logits, temperature): return (logits / temperature).softmax(-1)

def residual_token(p, q, greedy, gen):
    if greedy: return p.argmax(-1, keepdim=True)
    r = (p - q).clamp(min=0)                                                      # what the target wants more than the draft did
    return torch.multinomial(r / r.sum(), 1, generator=gen)

@torch.no_grad()
def speculative(target, draft, ids, n_new, k=4, temperature=1.0, greedy=False, gen=None):
    tc, dc = [{} for _ in target.blocks], [{} for _ in draft.blocks]
    if ids.size(1) > 1: target(ids[:, :-1], 0, tc); draft(ids[:, :-1], 0, dc)     # caches hold everything except the last token
    start, calls, proposed, accepted = ids.size(1), 0, 0, 0
    while ids.size(1) - start < n_new:
        pos = ids.size(1) - 1
        x, props, qs = ids[:, -1:], [], []
        for j in range(k):                                                        # the draft proposes k tokens, one by one (cheap)
            q = probs_of(draft(x, pos + j, dc)[:, -1], temperature)
            t = q.argmax(-1, keepdim=True) if greedy else torch.multinomial(q, 1, generator=gen)
            props.append(t); qs.append(q); x = t
        lg = target(torch.cat([ids[:, -1:]] + props, 1), pos, tc); calls += 1     # the target scores all k proposals in one pass
        n_acc, new = 0, None
        for j in range(k):
            p, t = probs_of(lg[:, j], temperature), props[j]
            if greedy: ok = p.argmax(-1).item() == t.item()
            else: ok = torch.rand(1, generator=gen).item() < min(1.0, (p[0, t.item()] / qs[j][0, t.item()]).item())
            if not ok:                                                            # first rejection: take a token from the target instead
                new = residual_token(p, qs[j], greedy, gen)
                break
            n_acc += 1
        if new is None:                                                           # all k accepted: a free bonus token from the target
            p = probs_of(lg[:, k], temperature); new = p.argmax(-1, keepdim=True) if greedy else torch.multinomial(p, 1, generator=gen)
        ids = torch.cat([ids] + props[:n_acc] + [new], 1)
        cache_cut(tc, pos + 1 + n_acc)                                            # forget the rejected proposals
        if n_acc == k: draft(props[-1], pos + k, dc)                              # the draft has not seen its own last proposal yet
        else: cache_cut(dc, pos + 1 + n_acc)
        proposed += k; accepted += n_acc
    return ids[:, :start + n_new], {"target_calls": calls, "proposed": proposed, "accepted": accepted}
