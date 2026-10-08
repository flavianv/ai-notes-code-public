"""Toy experiments shared by the reward hacking series (best of N, ensembles, audit loop, monitored training, learned world models). Results in results_fixes.json.
No language model is trained: these are finite synthetic examples that isolate one mechanism each. Seeds are fixed."""
import json
import numpy as np

R = {}
T = 20000
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
P_EXPLOIT = 0.02


def honest_q(rng, shape):
    return np.clip(rng.normal(0.6, 0.15, shape), 0, 1)

# ---------------------------------------------------------------- A. bound the optimization (early stopping, soft selection)
# honest candidate: quality q ~ N(0.6, 0.15), proxy = q + N(0, 0.1). exploit (prob 0.02): quality 0, proxy = 1.2 + N(0, 0.1).
def draw(rng, T, N, p=P_EXPLOIT):
    ex = rng.random((T, N)) < p
    q = np.where(ex, 0.0, honest_q(rng, (T, N)))
    proxy = np.where(ex, 1.2, q) + rng.normal(0, 0.1, (T, N))
    return ex, q, proxy

rng = np.random.default_rng(1)
raw = {'quality': [], 'hack_rate': [], 'proxy': []}
for N in BUDGETS:
    ex, q, proxy = draw(rng, T, N)
    k = proxy.argmax(1); idx = np.arange(T)
    raw['quality'].append(float(q[idx, k].mean())); raw['hack_rate'].append(float(ex[idx, k].mean())); raw['proxy'].append(float(proxy[idx, k].mean()))
best = int(np.argmax(raw['quality']))
R['bound'] = {'budgets': BUDGETS, 'raw': raw, 'early_stop_budget': BUDGETS[best], 'early_stop_quality': raw['quality'][best],
              'prior_quality': raw['quality'][0]}
# soft selection at N = 256: sample a candidate with probability proportional to exp(proxy / beta)  (KL-regularised selection)
soft = {}
rng = np.random.default_rng(2); ex, q, proxy = draw(rng, T, 256); idx = np.arange(T)
for beta in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1e9]:
    g = proxy / beta + rng.gumbel(size=proxy.shape); k = g.argmax(1)
    soft[str(beta)] = {'quality': float(q[idx, k].mean()), 'hack_rate': float(ex[idx, k].mean())}
R['bound']['soft_N256'] = soft

# ---------------------------------------------------------------- B. pessimistic ensembles: independent vs shared blind spots
# K evaluators each score every candidate. An exploit fools an evaluator with probability f = 0.5; a fooled evaluator scores 1.2, else the exploit's true level.
# independent: each evaluator is fooled independently.  correlated: one draw decides for all K evaluators.  aggregation = min over evaluators.
def ensemble(K, mode, N=256, f=0.5, seed=3):
    rng = np.random.default_rng(seed)
    ex = rng.random((T, N)) < P_EXPLOIT; q = np.where(ex, 0.0, honest_q(rng, (T, N)))
    if mode == 'independent': fooled = rng.random((T, N, K)) < f
    else: fooled = np.repeat((rng.random((T, N)) < f)[:, :, None], K, axis=2)
    base = np.repeat(q[:, :, None], K, axis=2)
    score = np.where(ex[:, :, None] & fooled, 1.2, base) + rng.normal(0, 0.1, (T, N, K))
    agg = score.min(2); k = agg.argmax(1); idx = np.arange(T)
    return float(q[idx, k].mean()), float(ex[idx, k].mean())
R['ensemble'] = {mode: {str(K): dict(zip(['quality', 'hack_rate'], ensemble(K, mode))) for K in [1, 2, 4, 8]} for mode in ['independent', 'correlated']}

# ---------------------------------------------------------------- C. the exploit you did not name, and the audit loop
# four exploit types, each drawn with probability 0.005 per candidate (0.02 in total), all scoring 1.2.
# A gate rejects the types it knows. Each round audits `A` selected outputs; every exploit found there is added to the gate.
def audit_loop(known0, A, rounds=6, N=256, seed=4):
    rng = np.random.default_rng(seed); gated = set(known0); out = []
    for r in range(rounds + 1):
        typ = rng.choice(5, size=(T, N), p=[.98, .005, .005, .005, .005])   # 0 = honest, 1..4 = exploit types
        q = np.where(typ > 0, 0.0, honest_q(rng, (T, N)))
        score = np.where(typ > 0, 1.2, q) + rng.normal(0, 0.1, (T, N))
        score = np.where(np.isin(typ, list(gated)), -9, score)
        k = score.argmax(1); idx = np.arange(T)
        out.append({'round': r, 'gated_types': len(gated), 'quality': float(q[idx, k].mean())})
        found = set(typ[idx, k][:A][typ[idx, k][:A] > 0].tolist()); gated |= found
    return out
R['audit'] = {'known_0_audit_50': audit_loop([], 50), 'known_1_audit_50': audit_loop([1], 50), 'known_0_audit_3': audit_loop([], 3)}

# ---------------------------------------------------------------- D. learning time: policing the policy vs fixing the reward  (3-action bandit, REINFORCE)
# actions: honest (reward 1.0, quality 1), visible hack (reward 1.2, quality 0), hidden hack (reward 1.2, quality 0).
# a chain-of-thought-style monitor flags a visible hack with probability 0.9 and never sees the hidden one.
def train(cond, seed, p_hidden0, steps=4000, batch=64, lr=0.5):
    rng = np.random.default_rng(seed); logits = np.log(np.array([1 - 0.02 - p_hidden0, 0.02, p_hidden0]))
    for _ in range(steps):
        p = np.exp(logits - logits.max()); p /= p.sum(); a = rng.choice(3, size=batch, p=p)
        r = np.array([1.0, 1.2, 1.2])[a]
        if cond == 'monitor_in_reward': r = np.where((a == 1) & (rng.random(batch) < 0.9), r - 2.0, r)
        if cond == 'fixed_reward': r = np.array([1.0, 0.0, 0.0])[a]
        adv = r - r.mean(); g = np.zeros(3)
        for j in range(3): g[j] = (adv * ((a == j) - p[j])).mean()
        logits = logits + lr * g
    p = np.exp(logits - logits.max()); p /= p.sum()
    return p
R['bandit'] = {}
for p_hidden0 in [0.001, 0.005, 0.02]:
    for cond in ['no_monitor', 'monitor_in_reward', 'fixed_reward']:
        P = np.array([train(cond, s, p_hidden0) for s in range(40)]).mean(0)
        R['bandit'][f'{cond}@hidden{p_hidden0}'] = {'p_honest': float(P[0]), 'p_visible_hack': float(P[1]), 'p_hidden_hack': float(P[2]), 'quality': float(P[0]),
                             'monitor_sees_hack_rate': float(0.9 * P[1])}

# ---------------------------------------------------------------- E. world model: replan from real outcomes, and stay near the data
true_s = lambda x: 0.8 - 0.6 * x ** 2; model_s = lambda x: 0.8 + 0.15 * x
real_ret = lambda x: 10 * true_s(x) - 0.5 * x
grid = np.linspace(0, 1, 101)
def kernel_fit(xs, hits, ns, bw=0.12):
    xs, hits, ns = map(np.array, (xs, hits, ns))
    def corr(x):
        w = np.exp(-0.5 * ((x[:, None] - xs[None]) / bw) ** 2) * ns[None]
        resid = hits / ns - model_s(xs)
        return (w * resid[None]).sum(1) / (w.sum(1) + 1.0)   # +1: shrink to the model where there is no data
    return corr
def replan(rounds=8, trials=30, seed=5, seeds=200):
    out = np.zeros(rounds + 1); xs_ = np.zeros(rounds + 1)
    for s in range(seeds):
        rng = np.random.default_rng(seed * 1000 + s); X, H, Ns = [], [], []
        for r in range(rounds + 1):
            if X: c = kernel_fit(X, H, Ns); ps = np.clip(model_s(grid) + c(grid), 0, 1)
            else: ps = model_s(grid)
            xstar = grid[np.argmax(10 * ps - 0.5 * grid)]
            out[r] += real_ret(xstar); xs_[r] += xstar
            X.append(xstar); H.append(rng.binomial(trials, true_s(xstar))); Ns.append(trials)
    return [{'round': r, 'mean_plan_x': float(xs_[r] / seeds), 'real_return': float(out[r] / seeds)} for r in range(rounds + 1)]
R['replan'] = replan()
def pessimistic(lam, seeds=200, trials=30):
    rets, xsel = [], []
    for s in range(seeds):
        rng = np.random.default_rng(7000 + s); X = [0.0, 0.1, 0.2, 0.3]; H = [rng.binomial(trials, true_s(x)) for x in X]; c = kernel_fit(X, H, [trials] * 4)
        ps = np.clip(model_s(grid) + c(grid), 0, 1); dist = np.abs(grid[:, None] - np.array(X)[None]).min(1)
        xstar = grid[np.argmax(10 * ps - 0.5 * grid - lam * dist)]; rets.append(real_ret(xstar)); xsel.append(xstar)
    return {'lambda': lam, 'mean_x': float(np.mean(xsel)), 'real_return': float(np.mean(rets))}
R['pessimism'] = [pessimistic(l) for l in [0, 2, 5, 10, 20]]
R['pessimism_context'] = {'optimal_real_return': float(real_ret(0.0)), 'data_range': [0.0, 0.3]}

json.dump(R, open('results_fixes.json', 'w'), indent=1)
b = R['bound']; print('A raw quality', [round(v, 3) for v in b['raw']['quality']]); print('A early stop', b['early_stop_budget'], round(b['early_stop_quality'], 3))
print('A soft', {k: (round(v['quality'], 3), round(v['hack_rate'], 3)) for k, v in b['soft_N256'].items()})
print('B', {m: {k: (round(v['quality'], 3), round(v['hack_rate'], 3)) for k, v in d.items()} for m, d in R['ensemble'].items()})
for k, v in R['audit'].items(): print('C', k, [(d['gated_types'], round(d['quality'], 3)) for d in v])
for k, v in R['bandit'].items(): print('D', k, {a: round(x, 3) for a, x in v.items()})
print('E replan', [(d['round'], round(d['mean_plan_x'], 2), round(d['real_return'], 2)) for d in R['replan']]); print('E pess', [(d['lambda'], round(d['mean_x'], 2), round(d['real_return'], 2)) for d in R['pessimism']])
