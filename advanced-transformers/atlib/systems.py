# Understanding Transformers, A4: the same trained model inside different systems.
# The model is the interleaved A3 model: it reads an action, writes the next state, and its own states are fed back.
import random, torch, torch.nn.functional as F
from .automata import TRANSITIONS, run, compose, random_actions, ACT, BOS, S0, make_model

PREFIX = {0: "", 1: "A", 2: "AA"}                                         # actions that take the machine from 0 to each start state


def load(path, dev="cpu"):
    m = make_model("interleaved"); m.load_state_dict(torch.load(path, map_location=dev)); return m.to(dev).eval()


# 1. Running the model: one batched pass over many action strings of the SAME length; greedy or sampled states
@torch.no_grad()
def roll(model, seqs, temp=0.0, gen=None, dev="cpu"):
    """Returns (states (B, n), total log-probability of the written states (B,))."""
    B, n = len(seqs), len(seqs[0])
    acts = torch.tensor([[ACT[a] for a in s] for s in seqs], device=dev)
    caches = [{} for _ in model.blocks]; model(torch.full((B, 1), BOS, device=dev), 0, caches); pos = 1
    states, logp = [], torch.zeros(B, device=dev)
    for t in range(n):
        out = model(acts[:, t:t + 1], pos, caches)[:, -1, S0:S0 + 3]; pos += 1
        lp = F.log_softmax(out, -1)
        s = lp.argmax(-1) if temp == 0 else torch.multinomial((lp / temp).softmax(-1), 1, generator=gen).squeeze(1)
        logp += lp.gather(1, s[:, None]).squeeze(1); states.append(s)
        model((S0 + s)[:, None], pos, caches); pos += 1
    return torch.stack(states, 1).cpu(), logp.cpu()


def trace_ok(seq, states):                                                # the exact checker: replay every transition
    return list(states) == run(seq)[1][1:]


# 2. Sampling: N candidate traces per problem, then three ways to pick one
def sampling(model, n, N=64, size=200, temp=1.0, seed=0, chunk=1600):
    rng = random.Random(seed + n); g = torch.Generator().manual_seed(seed)
    seqs = [random_actions(n, rng) for _ in range(size)]
    truth = [run(s)[0] for s in seqs]
    S, L = [], []
    rows = [s for s in seqs for _ in range(N)]
    for i in range(0, len(rows), chunk):
        st, lp = roll(model, rows[i:i + chunk], temp, g); S.append(st); L.append(lp)
    S, L = torch.cat(S).view(size, N, n), torch.cat(L).view(size, N)
    out = {"N": [], "single": None}
    final_ok = S[:, :, -1] == torch.tensor(truth)[:, None]
    valid = torch.tensor([[trace_ok(seqs[b], S[b, k].tolist()) for k in range(N)] for b in range(size)])
    out["single"] = final_ok[:, 0].float().mean().item()
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k > N: break
        f, v, l = final_ok[:, :k], valid[:, :k], L[:, :k]
        vote = [max(range(3), key=lambda s: ((S[b, :k, -1] == s).sum().item(), -s)) == truth[b] for b in range(size)]
        best = f[torch.arange(size), l.argmax(1)]
        chk = [bool(f[b][v[b]].any()) if v[b].any() else bool(f[b, 0]) for b in range(size)]   # a valid trace if any, else the first candidate
        out["N"].append({"k": k, "coverage": f.any(1).float().mean().item(), "vote": sum(vote) / size,
                         "likelihood": best.float().mean().item(), "checker": sum(chk) / size})
    return out


# 3. Search: propose states best-first, an exact checker accepts the first valid one; count checker calls
@torch.no_grad()
def search(model, n, size=200, seed=0, dev="cpu"):
    rng = random.Random(seed + n); seqs = [random_actions(n, rng) for _ in range(size)]
    B = size; acts = torch.tensor([[ACT[a] for a in s] for s in seqs], device=dev); state = torch.zeros(B, dtype=torch.long)
    caches = [{} for _ in model.blocks]; model(torch.full((B, 1), BOS, device=dev), 0, caches); pos = 1; calls = 0; top1 = 0
    for t in range(n):
        out = model(acts[:, t:t + 1], pos, caches)[:, -1, S0:S0 + 3]; pos += 1
        order = out.argsort(-1, descending=True).cpu()
        nxt = torch.tensor([TRANSITIONS[seqs[b][t]][state[b]] for b in range(B)])
        rank = (order == nxt[:, None]).float().argmax(1)                 # how many proposals the checker must reject before the right one
        calls += (rank + 1).sum().item(); top1 += (rank == 0).sum().item(); state = nxt
        model((S0 + state)[:, None].to(dev), pos, caches); pos += 1
    return {"success": 1.0, "checker_calls_per_step": calls / (B * n), "model_top1_per_step": top1 / (B * n)}


# 4. Memory harness: the harness stores {next index, state}; each call shows the model a 3-5 token context
@torch.no_grad()
def step(model, states, actions, dev="cpu"):
    """One transition per row, from a tiny in-distribution context: BOS, a prefix that reaches the state, the action."""
    out = torch.empty(len(states), dtype=torch.long)
    for s in (0, 1, 2):
        idx = [i for i, x in enumerate(states) if x == s]
        if not idx: continue
        ctx = [BOS] + [t for a, q in zip(PREFIX[s], run(PREFIX[s])[1][1:]) for t in (ACT[a], S0 + q)]
        ids = torch.tensor([ctx + [ACT[actions[i]]] for i in idx], device=dev)
        out[idx] = model(ids)[:, -1, S0:S0 + 3].argmax(-1).cpu()
    return out.tolist()


def harness(model, n, size=200, seed=0):
    rng = random.Random(seed + n); seqs = [random_actions(n, rng) for _ in range(size)]
    state = [0] * size
    for t in range(n): state = step(model, state, [s[t] for s in seqs])
    return {"final": sum(int(state[b] == run(seqs[b])[0]) for b in range(size)) / size, "calls": n * size}


# 5. Workers: chunks of 20 (inside the training range); each worker reports its chunk's whole table
def workers(model, n=200, chunk=20, size=200, seed=0):
    rng = random.Random(seed + n); seqs = [random_actions(n, rng) for _ in range(size)]
    truth = [run(s)[0] for s in seqs]; K = n // chunk
    chunks = [[s[k * chunk:(k + 1) * chunk] for k in range(K)] for s in seqs]
    tables = [[None] * K for _ in range(size)]; only0 = [[None] * K for _ in range(size)]
    for s0 in (0, 1, 2):                                                  # each worker runs the model 3 times, once per start state
        rows = [PREFIX[s0] + chunks[b][k] for b in range(size) for k in range(K)]
        st, _ = roll(model, rows)
        for r, (b, k) in enumerate((b, k) for b in range(size) for k in range(K)):
            fs = st[r, -1].item()
            tables[b][k] = (tables[b][k] or [None] * 3); tables[b][k][s0] = fs
            if s0 == 0: only0[b][k] = fs
    full = 0; naive = 0
    for b in range(size):
        tab = (0, 1, 2)
        for k in range(K): tab = compose(tuple(tables[b][k]), tab)
        full += tab[0] == truth[b]
        naive += only0[b][-1] == truth[b]                                 # final-state-only workers: the last chunk's answer, assuming it starts at 0
    # budget-matched sequential baseline: one model, chunk after chunk, carrying the state
    state = [0] * size
    for k in range(K):
        start = list(state)                                               # group by the state at the START of this chunk
        for s0 in (0, 1, 2):
            idx = [b for b in range(size) if start[b] == s0]
            if not idx: continue
            st, _ = roll(model, [PREFIX[s0] + chunks[b][k] for b in idx])
            for j, b in enumerate(idx): state[b] = st[j, -1].item()
    seq = sum(int(state[b] == truth[b]) for b in range(size))
    return {"tables": full / size, "final_only": naive / size, "sequential_chunks": seq / size, "model_calls_tables": 3 * K, "model_calls_sequential": K,
            "serial_rounds_tables": 1, "serial_rounds_sequential": K}


# 6. Reliability: per-step error inside the training range, and the (1 - eps)^T prediction
def reliability(model, n=20, size=1000, seed=0):
    rng = random.Random(seed + 999); seqs = [random_actions(n, rng) for _ in range(size)]
    st, _ = roll(model, seqs)
    truth = torch.tensor([run(s)[1][1:] for s in seqs])
    wrong = (st != truth); first = wrong.cumsum(1) == 0                  # steps before the first error
    errs = int((wrong & torch.cat([torch.ones(size, 1, dtype=torch.bool), first[:, :-1]], 1)).sum())   # first errors only
    steps = int(torch.cat([torch.ones(size, 1, dtype=torch.bool), first[:, :-1]], 1).sum())
    return {"steps": steps, "errors": errs, "trace_ok": (~wrong.any(1)).float().mean().item()}
