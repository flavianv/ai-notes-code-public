# Generation: logits -> temperature -> top-k / top-p -> softmax -> sample, one token at a time with a KV cache.
import torch


def sample_probs(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k is not None:
        kth = logits.topk(top_k).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))        # drop everything below the k-th best
    probs = logits.softmax(-1)
    if top_p is not None:
        p, idx = probs.sort(descending=True)
        drop = p.cumsum(-1) - p > top_p                                  # drop the tail once the kept mass exceeds top_p
        probs = torch.zeros_like(probs).scatter(-1, idx, p.masked_fill(drop, 0.0))
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


@torch.no_grad()
def generate(model, ids, n_new, **sampling):
    caches = [{} for _ in model.blocks]
    logits = model(ids, 0, caches)[:, -1]                                # prefill: run the whole prompt once
    for _ in range(n_new):
        nxt = torch.multinomial(sample_probs(logits, **sampling), 1)
        ids = torch.cat([ids, nxt], 1)
        logits = model(nxt, ids.size(1) - 1, caches)[:, -1]              # decode: only the new token, keys/values come from the cache
    return ids


