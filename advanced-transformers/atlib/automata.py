# The 3-state machine of "Understanding Transformers" and four ways to train a transformer on it.
import random, torch, torch.nn as nn, torch.nn.functional as F
from tlib.model import Attention, Block, GPT, rope

# 1. The machine: A rotates 0->1->2->0, B swaps 0 and 1, I does nothing
TRANSITIONS = {"A": (1, 2, 0), "B": (1, 0, 2), "I": (0, 1, 2)}
IDENTITY = (0, 1, 2)


# Two everyday finite-state machines from the A2 intro: a turnstile, and "am I inside a quoted string?"
TURNSTILE = {"coin": {"locked": "unlocked", "unlocked": "unlocked"}, "push": {"locked": "locked", "unlocked": "locked"}}


def turnstile(inputs, state="locked"):
    for x in inputs:
        state = TURNSTILE[x][state]
    return state


def in_string(text):                                                     # what a code editor tracks to colour strings
    inside, out = False, []
    for ch in text:
        if ch == '"': inside = not inside                                # a quote flips the state, anything else keeps it
        out.append(inside)
    return out


def run(actions, start=0):                                               # sequential solver: one state update per action
    state, trace = start, [start]
    for a in actions:
        state = TRANSITIONS[a][state]
        trace.append(state)
    return state, trace


def compose(later, earlier):                                             # a chunk summary is a whole function on states
    return tuple(later[earlier[s]] for s in range(3))


def summarize(actions):                                                  # balanced solver: depth log2(n) if branches run in parallel
    if not actions: return IDENTITY
    if len(actions) == 1: return TRANSITIONS[actions[0]]
    mid = len(actions) // 2
    return compose(summarize(actions[mid:]), summarize(actions[:mid]))


# The same tree trick on plain numbers: add in pairs, all at once. Returns every round, so the film can show them.
def tree_rounds(items, combine):
    rounds = [list(items)]
    while len(rounds[-1]) > 1:
        r = rounds[-1]
        rounds.append([combine(r[i], r[i + 1]) if i + 1 < len(r) else r[i] for i in range(0, len(r), 2)])
    return rounds                                                        # len(rounds) - 1 = ceil(log2 n) rounds


def table_rounds(actions):                                               # tree of tables: combine "left then right" in pairs
    return tree_rounds([TRANSITIONS[a] for a in actions], lambda left, right: compose(right, left))


# 2. Tokens: actions, the '=' separator, the three states, a start token
A, B, I, EQ, S0, BOS = 0, 1, 2, 3, 4, 7
VOCAB = 8
ACT = {"A": A, "B": B, "I": I}


def random_actions(n, rng, p=(1 / 3, 1 / 3, 1 / 3)):
    return "".join(rng.choices("ABI", weights=p, k=n))


# Each format returns (tokens, mask): mask marks the positions whose NEXT token is a state the model must predict
def encode(actions, fmt):
    _, trace = run(actions)
    acts = [ACT[a] for a in actions]
    if fmt in ("direct", "looped"):                                      # BOS a1..an = s_n
        toks = [BOS] + acts + [EQ, S0 + trace[-1]]
    elif fmt == "after":                                                 # BOS a1..an = s1..sn   (states written after the actions)
        toks = [BOS] + acts + [EQ] + [S0 + s for s in trace[1:]]
    elif fmt == "interleaved":                                           # BOS a1 s1 a2 s2 ... an sn   (each state right after its action)
        toks = [BOS] + [t for a, s in zip(acts, trace[1:]) for t in (a, S0 + s)]
    mask = [False] * (len(toks) - 1)
    for i in range(1, len(toks)):
        if toks[i] >= S0 and toks[i] != BOS: mask[i - 1] = True
    return toks, mask


def batch(n, size, fmt, rng, p=(1 / 3, 1 / 3, 1 / 3)):                  # one length per batch: no padding needed
    rows = [encode(random_actions(n, rng, p), fmt) for _ in range(size)]
    return torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows])


# 3. Looped transformer: ONE block applied again and again, the input re-added at every iteration
class Looped(nn.Module):
    def __init__(self, vocab=VOCAB, d=128, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.block = Block(d, n_heads)                                   # shared weights across iterations
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, loops):
        e = self.embed(ids); x = e
        for _ in range(loops):
            x = self.block(x + e)                                        # input injection keeps the actions visible
        return self.head(self.norm(x))


# Variants for the A3 follow-up: "interleaved-nope" (no position encoding), "interleaved-offset" (RoPE, random start position),
# "interleaved-window" (attention limited to the last 8 tokens)
class NoPEAttention(Attention):                                          # same head, without the RoPE rotation: only the causal mask orders tokens
    def forward(self, x, start=0, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1 and k.size(2) == T)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class WindowAttention(Attention):                                        # RoPE as usual, but each token may only look at the last WINDOW tokens
    WINDOW = 8

    def forward(self, x, start=0, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        pos = torch.arange(start, start + T, device=x.device, dtype=torch.float)
        q, k = rope(q, pos), rope(k, pos)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        S = k.size(2); qi = torch.arange(S - T, S, device=x.device)[:, None]; ki = torch.arange(S, device=x.device)[None, :]
        mask = (ki <= qi) & (ki > qi - self.WINDOW)                      # causal AND recent: the amount of context never grows
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


def split(fmt):                                                          # "interleaved-offset" -> ("interleaved", "offset")
    base, _, pos = fmt.partition("-")
    return base, pos or "rope"


def make_model(fmt, d=128, n_layers=4):
    base, pos = split(fmt)
    if base == "looped": return Looped(d=d)
    model = GPT(VOCAB, d=d, n_heads=4, n_layers=n_layers)
    if pos == "nope":
        for b in model.blocks: b.attn.__class__ = NoPEAttention         # same weights, no rotation
    if pos == "window":
        for b in model.blocks: b.attn.__class__ = WindowAttention
    return model


def logits_of(model, ids, fmt, start=0):
    return model(ids, loops=ids.size(1)) if fmt == "looped" else model(ids, start)   # loops grow with the input: one per token


# 4. Training: cross-entropy on the state tokens only
def train(fmt, steps=3000, max_len=20, bs=128, lr=1e-3, seed=0, dev="cpu", log=None):
    torch.manual_seed(seed); rng = random.Random(seed)
    model = make_model(fmt).to(dev)
    fmt, pos = split(fmt)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    for step in range(steps):
        toks, mask = batch(rng.randint(1, max_len), bs, fmt, rng)
        toks, mask = toks.to(dev), mask.to(dev)
        start = rng.randint(0, 512) if pos == "offset" else 0               # offset: every batch starts at a random position, so far positions are seen
        logits = logits_of(model, toks[:, :-1], fmt, start)
        loss = F.cross_entropy(logits[mask], toks[:, 1:][mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log is not None and step % 100 == 0: log.append((step, loss.item()))
    return model.eval()


# 5. Evaluation: the model writes its own states (greedy); we score the final state and the whole trace
@torch.no_grad()
def evaluate(model, fmt, n, size=500, seed=123, dev="cpu", p=(1 / 3, 1 / 3, 1 / 3)):
    fmt = split(fmt)[0]
    rng = random.Random(seed + n)
    seqs = [random_actions(n, rng, p) for _ in range(size)]
    truth = torch.tensor([run(s)[1][1:] for s in seqs])                  # (size, n) true states s1..sn
    acts = torch.tensor([[ACT[a] for a in s] for s in seqs], device=dev)
    bos = torch.full((size, 1), BOS, device=dev)
    eq = torch.full((size, 1), EQ, device=dev)
    if fmt in ("direct", "looped"):
        ids = torch.cat([bos, acts, eq], 1)
        final = logits_of(model, ids, fmt)[:, -1, S0:S0 + 3].argmax(-1).cpu()
        return {"final": (final == truth[:, -1]).float().mean().item(), "trace": None}
    caches = [{} for _ in model.blocks]
    if fmt == "after":                                                   # read the prompt, then write n states
        prompt = torch.cat([bos, acts, eq], 1)
        out = model(prompt, 0, caches)[:, -1]; pos = prompt.size(1); states = []
        for _ in range(n):
            s = out[:, S0:S0 + 3].argmax(-1); states.append(s)
            out = model((S0 + s)[:, None], pos, caches)[:, -1]; pos += 1
    else:                                                                # interleaved: action given, state written by the model, fed back
        model(bos, 0, caches); pos = 1; states = []
        for t in range(n):
            out = model(acts[:, t:t + 1], pos, caches)[:, -1]; pos += 1
            s = out[:, S0:S0 + 3].argmax(-1); states.append(s)
            model((S0 + s)[:, None], pos, caches); pos += 1
    pred = torch.stack(states, 1).cpu()
    wrong = pred != truth
    first = [int(w.nonzero()[0]) + 1 for w in wrong if w.any()]          # step of the first wrong state, for failed traces
    return {"final": (pred[:, -1] == truth[:, -1]).float().mean().item(),
            "trace": (~wrong.any(1)).float().mean().item(),
            "first_error_median": sorted(first)[len(first) // 2] if first else None}
