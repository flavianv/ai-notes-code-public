# Understanding Transformers, A2: what one forward pass can compute on the 3-state machine, and what it stores along the way.
import math, random, torch, torch.nn as nn, torch.nn.functional as F
from tlib.model import GPT
from .automata import run, summarize, random_actions, ACT, BOS, S0, VOCAB

# The 6 functions on {0,1,2} that A, B and I can produce (A and B generate all permutations of 3 states)
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2), (0, 2, 1), (2, 1, 0)]


def balanced_depth(n):                                                   # stages of pairwise composition: ceil(log2 n)
    return math.ceil(math.log2(n)) if n > 1 else 0


# 1. Labels at every position: the state after each prefix ("state"), or "more A than B so far?" ("count")
def labelled_batch(n, size, task, rng):
    seqs = [random_actions(n, rng) for _ in range(size)]
    ids = torch.tensor([[BOS] + [ACT[a] for a in s] for s in seqs])
    if task == "state":
        y = torch.tensor([run(s)[1][1:] for s in seqs])
    else:
        y = torch.tensor([[int(s[:t + 1].count("A") > s[:t + 1].count("B")) for t in range(n)] for s in seqs])
    return ids, y, seqs


def train_labelled(task="state", n_layers=2, lengths=(1, 20), steps=2000, d=64, seed=0, log=None):
    torch.manual_seed(seed); rng = random.Random(seed)
    model = GPT(VOCAB, d=d, n_heads=4, n_layers=n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    k = 3 if task == "state" else 2
    for step in range(steps):
        ids, y, _ = labelled_batch(rng.randint(*lengths), 128, task, rng)
        logits = model(ids)[:, 1:, S0:S0 + k]                            # read the answer off the state-token logits
        loss = F.cross_entropy(logits.reshape(-1, k), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log is not None and step % 100 == 0: log.append((step, loss.item()))
    return model.eval()


@torch.no_grad()
def labelled_accuracy(model, task, n, size=500, seed=7):
    ids, y, _ = labelled_batch(n, size, task, random.Random(seed + n))
    k = 3 if task == "state" else 2
    pred = model(ids)[:, 1:, S0:S0 + k].argmax(-1)
    return {"last": (pred[:, -1] == y[:, -1]).float().mean().item(), "all": (pred == y).float().mean().item()}


# 2. Probe: does the residual stream hold the whole function of the prefix (6 classes), or only the state reached from 0 (3 classes)?
@torch.no_grad()
def residuals(model, ids):                                               # residual stream after the embedding and after each block
    x = model.embed(ids); out = [x]
    for b in model.blocks:
        x = b(x); out.append(x)
    return out


def probe(model, n=20, size=2000, seed=11):
    rng = random.Random(seed)
    ids, _, seqs = labelled_batch(n, size, "state", rng)
    perm = torch.tensor([[PERMS.index(summarize(s[:t + 1])) for t in range(n)] for s in seqs])
    state = torch.tensor([run(s)[1][1:] for s in seqs])
    res = [r[:, 1:].reshape(-1, r.size(-1)) for r in residuals(model, ids)]
    perm, state = perm.reshape(-1), state.reshape(-1)
    half = perm.numel() // 2                                             # fit on one half of the positions, score on the other
    out = []
    for r in res:
        row = {}
        for name, y, k in (("function", perm, 6), ("state", state, 3)):
            lin = nn.Linear(r.size(-1), k); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
            for _ in range(300):
                loss = F.cross_entropy(lin(r[:half]), y[:half]); opt.zero_grad(); loss.backward(); opt.step()
            row[name] = (lin(r[half:]).argmax(-1) == y[half:]).float().mean().item()
        out.append(row)
    return out                                                           # one dict per layer: probe accuracy for each target
