# Understanding Transformers, A1: an induction circuit, superposition and a sparse autoencoder, all small enough to inspect.
import random, torch, torch.nn as nn, torch.nn.functional as F
from tlib.model import rope


# 1. An attention-only transformer that also returns its attention patterns, and can switch heads off
class AttnOnly(nn.Module):
    def __init__(self, vocab, d=64, n_heads=4, n_layers=2):
        super().__init__()
        self.h = n_heads
        self.embed = nn.Embedding(vocab, d)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.qkv = nn.ModuleList(nn.Linear(d, 3 * d, bias=False) for _ in range(n_layers))
        self.out = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, off=()):                                      # off: set of (layer, head) to zero out
        B, T = ids.shape
        x = self.embed(ids); pats = []
        pos = torch.arange(T, device=ids.device, dtype=torch.float)
        mask = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril()
        for l, (n, qkv, out) in enumerate(zip(self.norms, self.qkv, self.out)):
            q, k, v = qkv(n(x)).view(B, T, 3, self.h, -1).permute(2, 0, 3, 1, 4)
            q, k = rope(q, pos), rope(k, pos)
            att = (q @ k.transpose(-1, -2) / q.size(-1) ** 0.5).masked_fill(~mask, float("-inf")).softmax(-1)
            y = att @ v                                                  # (B, heads, T, head_dim)
            for (ll, hh) in off:
                if ll == l: y[:, hh] = 0
            x = x + out(y.transpose(1, 2).reshape(B, T, -1))             # no MLP: only attention writes to the residual stream
            pats.append(att)
        return self.head(self.norm(x)), pats


# 2. Data: BOS, `half` random tokens, the same tokens again. Only the repeat is predictable.
#    Training varies `half`, so a head that just looks a fixed distance back cannot solve it: the model must match content.
BOS_ID = 0


def repeat_batch(size, half=20, vocab=50, gen=None):
    x = torch.randint(1, vocab, (size, half), generator=gen)
    return torch.cat([torch.full((size, 1), BOS_ID), x, x], 1)


def train_induction(n_layers=2, steps=3000, half=20, vocab=50, seed=0, log=None, halves=(8, 32)):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = AttnOnly(vocab, n_layers=n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(steps):
        ids = repeat_batch(64, random.Random(seed * 100003 + step).randint(*halves) if halves else half, vocab, gen)
        logits, _ = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if log is not None and step % 50 == 0:
            log.append((step, *halves_loss(model, half, vocab)))
    return model.eval()


@torch.no_grad()
def halves_loss(model, half=20, vocab=50, off=(), size=256):
    ids = repeat_batch(size, half, vocab, torch.Generator().manual_seed(999))
    logits, _ = model(ids[:, :-1], off)
    l = F.cross_entropy(logits.reshape(-1, vocab), ids[:, 1:].reshape(-1), reduction="none").view(size, -1)
    return l[:, :half].mean().item(), l[:, half + 1:].mean().item()     # first half (random) vs repeated half (skip the first repeat token)


@torch.no_grad()
def repeat_accuracy(model, half=20, vocab=50, off=(), size=256):
    ids = repeat_batch(size, half, vocab, torch.Generator().manual_seed(999))
    logits, _ = model(ids[:, :-1], off)
    return (logits.argmax(-1) == ids[:, 1:])[:, half + 1:].float().mean().item()


@torch.no_grad()
def head_scores(model, half=20, vocab=50, size=256):
    """Per (layer, head): previous-token score = mean attention to position i-1;
    induction score = mean attention, in the repeat, to the token AFTER the earlier copy (offset half-1 back)."""
    ids = repeat_batch(size, half, vocab, torch.Generator().manual_seed(999))
    _, pats = model(ids)
    T = ids.size(1); prev, ind = {}, {}
    i = torch.arange(2, T); j = torch.arange(half + 2, T)
    for l, att in enumerate(pats):
        for h in range(att.size(1)):
            a = att[:, h].mean(0)
            prev[(l, h)] = a[i, i - 1].mean().item()
            ind[(l, h)] = a[j, j - half + 1].mean().item()
    return prev, ind


def group_ablation(model, half=20, vocab=50, thresh=0.5):
    """Switch heads off in groups, because training often builds the same head more than once."""
    prev, ind = head_scores(model, half, vocab)
    P = {k for k, v in prev.items() if k[0] == 0 and v > thresh}         # previous-token heads in layer 1
    I = {k for k, v in ind.items() if k[0] == 1 and v > thresh}          # induction heads in layer 2
    L0 = {k for k in prev if k[0] == 0}
    name = lambda S: sorted(f"{l}.{h}" for l, h in S)
    return {"prev_heads": name(P), "ind_heads": name(I), "acc": repeat_accuracy(model, half, vocab),
            "all_prev_off": repeat_accuracy(model, half, vocab, off=P), "non_prev_L0_off": repeat_accuracy(model, half, vocab, off=L0 - P),
            "all_ind_off": repeat_accuracy(model, half, vocab, off=I),
            "one_ind_kept": repeat_accuracy(model, half, vocab, off=I - {max(I, key=ind.get)}) if I else None}


# 3. Superposition toy model: 5 sparse features squeezed through 2 numbers, h = W x, x' = ReLU(W^T h + b)
def train_toy(n=5, m=2, sparsity=0.9, steps=5000, seed=0):
    torch.manual_seed(seed)
    W = nn.Parameter(torch.randn(m, n) * 0.1); b = nn.Parameter(torch.zeros(n))
    imp = 0.9 ** torch.arange(n, dtype=torch.float)                     # earlier features matter more
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        x = torch.rand(1024, n) * (torch.rand(1024, n) > sparsity)       # each feature is on with probability 1 - sparsity
        loss = (imp * (F.relu(x @ W.T @ W + b) - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()


def represented(W, thresh=0.5):                                          # a feature is represented if its column has real length
    return int((W.norm(dim=0) > thresh).sum())


# 4. Sparse autoencoder: learn a dictionary of directions that explains the 2-number codes as sparse combinations
def train_sae(W, sparsity=0.9, dict_size=10, l1=3e-3, steps=6000, seed=0):
    torch.manual_seed(seed)
    m, n = W.shape
    enc = nn.Linear(m, dict_size); dec = nn.Linear(dict_size, m)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=3e-3)
    for _ in range(steps):
        x = torch.rand(1024, n) * (torch.rand(1024, n) > sparsity)
        h = x @ W.T                                                      # the activations we are given, not the features
        f = F.relu(enc(h))
        loss = ((dec(f) - h) ** 2).sum(-1).mean() + l1 * (f.abs() * dec.weight.norm(dim=0)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return dec.weight.detach()                                           # (m, dict_size): one direction per column


def recovery(W, D):                                                      # for each true feature: best cosine with a dictionary direction
    Wn = F.normalize(W, dim=0); Dn = F.normalize(D, dim=0)
    return (Wn.T @ Dn).max(1).values


# 5. The draft's constructed MLP: 9 ReLU pair detectors compute the machine's transition exactly
def pair_detector_mlp(transitions):
    W1 = torch.zeros(9, 6); b1 = -torch.ones(9); W2 = torch.zeros(3, 9)
    acts = list(transitions)
    for s in range(3):
        for ai, a in enumerate(acts):
            u = 3 * s + ai
            W1[u, s] = 1; W1[u, 3 + ai] = 1                              # h = ReLU(state_s + action_a - 1)
            W2[transitions[a][s], u] = 1                                 # route the active pair to the next state
    return lambda state, action: W2 @ F.relu(W1 @ torch.cat([state, action]) + b1)
