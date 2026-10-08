"""Toy experiments shared by the reward hacking series: optimization pressure (tilt, tails, canaries), feedback leakage, spot checks and test wear-out. Results in results_pressure.json.
Synthetic, seeds fixed, no language model trained."""
import json, math
import numpy as np
R = {}

# ---------------------------------------------------------------- F. the tilt model: predicted vs measured exploit rate and quality, best of N with temperature
P, MU_Q, SQ, SE, RE = 0.02, 0.6, 0.15, 0.1, 1.2
SR = math.hypot(SQ, SE)
def tilt(beta, p=P, re=RE):
    x = 1 / beta; a = p * math.exp(re * x); h = (1 - p) * math.exp(MU_Q * x + SR ** 2 * x * x / 2)
    piE = a / (a + h); qh = MU_Q + SQ ** 2 * x   # tilting shifts the honest quality mean by cov(q, r) / beta
    return piE, (1 - piE) * min(qh, 1.0)
def tempered(beta, N, T=20000, seed=11):
    rng = np.random.default_rng(seed); ex = rng.random((T, N)) < P
    q = np.where(ex, 0.0, np.clip(rng.normal(MU_Q, SQ, (T, N)), 0, 1)); proxy = np.where(ex, RE, q) + rng.normal(0, SE, (T, N))
    k = (proxy / beta + rng.gumbel(size=proxy.shape)).argmax(1); idx = np.arange(T)
    return float(ex[idx, k].mean()), float(q[idx, k].mean())
betas = [1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05]
R['tilt'] = {'beta': betas, 'theory_hack': [tilt(b)[0] for b in betas], 'theory_quality': [tilt(b)[1] for b in betas],
             'measured_N256': [tempered(b, 256) for b in betas], 'measured_N2048': [tempered(b, 2048, T=4000) for b in betas]}
A = SR ** 2 / 2; B = RE - MU_Q; C = math.log((1 - P) / P); xs = (B - math.sqrt(B * B - 4 * A * C)) / (2 * A)
R['tilt']['beta_star'] = 1 / xs
grid = np.linspace(0.04, 1.0, 400); qv = [tilt(b)[1] for b in grid]; R['tilt']['theory_quality_peak_beta'] = float(grid[int(np.argmax(qv))]); R['tilt']['theory_quality_peak'] = float(max(qv))

# ---------------------------------------------------------------- the worst-case budget: binary KL d(eps || p0)
d = lambda e, p: e * math.log(e / p) + (1 - e) * math.log((1 - e) / (1 - p))
R['budget'] = {f'{p0:g}': {f'{e:g}': d(e, p0) for e in (0.01, 0.1, 0.5)} for p0 in (1e-2, 1e-4, 1e-6)}
R['budget']['best_of_4_kl'] = math.log(4) - 3 / 4
R['budget']['best_of_4_hack'] = 1 - (1 - P) ** 4
R['budget']['min_kl_for_that_hack'] = d(1 - (1 - P) ** 4, P)

# ---------------------------------------------------------------- G. tails: best of n with light-tailed vs heavy-tailed proxy errors
rng = np.random.default_rng(21); ns = [1, 4, 16, 64, 256, 1024, 4096]; T = 3000; out = {'n': ns, 'light': [], 'heavy': []}
for n in ns:
    for kind in ('light', 'heavy'):
        acc = []
        for _ in range(T // 500):
            q = rng.normal(0, 1, (500, n)); e = rng.normal(0, 1, (500, n)) if kind == 'light' else rng.standard_t(1.5, (500, n))
            k = (q + e).argmax(1); acc.append(q[np.arange(500), k].mean())
        out[kind].append(float(np.mean(acc)))
R['tails'] = out

# ---------------------------------------------------------------- H. a canary: a planted, perfectly detectable exploit that is more tempting than the real one
def tempered3(beta, with_canary, N=256, T=20000, seed=31):
    rng = np.random.default_rng(seed); u = rng.random((T, N))
    ex = u < P; can = (u >= P) & (u < P + 0.05) if with_canary else np.zeros_like(ex)
    q = np.where(ex | can, 0.0, np.clip(rng.normal(MU_Q, SQ, (T, N)), 0, 1)); proxy = np.where(ex, RE, np.where(can, 1.3, q)) + rng.normal(0, SE, (T, N))
    k = (proxy / beta + rng.gumbel(size=proxy.shape)).argmax(1); idx = np.arange(T)
    return float(can[idx, k].mean()), float(ex[idx, k].mean())
inv = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]
R['canary_selection'] = {'inv_beta': inv, 'canary_with': [tempered3(1 / x, True)[0] for x in inv], 'exploit_without': [tempered3(1 / x, False)[1] for x in inv]}
def train(actions_p, rewards, steps=3000, batch=64, lr=0.5, seed=0):
    rng = np.random.default_rng(seed); logits = np.log(np.array(actions_p)); hist = []
    for t in range(steps):
        p = np.exp(logits - logits.max()); p /= p.sum(); hist.append(p.copy()); a = rng.choice(len(p), size=batch, p=p); r = np.array(rewards)[a]
        adv = r - r.mean(); logits = logits + lr * np.array([(adv * ((a == j) - p[j])).mean() for j in range(len(p))])
    return np.array(hist)
first = lambda h, j, th=0.1: int(np.argmax(h[:, j] >= th)) if (h[:, j] >= th).any() else None
no_c = [first(train([.98, .02], [1.0, 1.2], seed=s), 1) for s in range(20)]
with_c = [first(train([.93, .02, .05], [1.0, 1.2, 1.3], seed=s), 2) for s in range(20)]
R['canary_training'] = {'exploit_10pct_step_without_canary': float(np.median(no_c)), 'canary_10pct_step': float(np.median(with_c))}

# ---------------------------------------------------------------- I. leakage: a fixed verifier with per-test feedback is an oracle an attacker can climb
rng = np.random.default_rng(41); D, V, K = 1000, 10, 10; f = rng.integers(0, V, D)
def attack(fixed, feedback, Q=3000):
    g = rng.integers(0, V, D); tests = rng.choice(D, K, replace=False); rew, acc = [], []
    for t in range(Q):
        T_ = tests if fixed else rng.choice(D, K, replace=False)
        passed = g[T_] == f[T_]
        if feedback == 'per_test':   # the attacker sees which inputs failed: try another output on one of them
            bad = T_[~passed]
            if len(bad): x = rng.choice(bad); g[x] = (g[x] + 1) % V
        else:                         # one aggregate bit: no direction to climb, random tweaks
            if not passed.all(): x = rng.choice(T_); g[x] = rng.integers(0, V)
        rew.append(float(passed.mean() if feedback == 'per_test' else passed.all())); acc.append(float((g == f).mean()))
    return rew, acc
out = {}
for name, fixed, fb in [('fixed_per_test', True, 'per_test'), ('fixed_one_bit', True, 'bit'), ('fresh_per_test', False, 'per_test')]:
    rw, ac = attack(fixed, fb); out[name] = {'reward_last200': float(np.mean(rw[-200:])), 'true_accuracy_end': ac[-1], 'reward_at_100': float(np.mean(rw[90:110])), 'acc_at_100': ac[100]}
R['leakage'] = out

json.dump(R, open('results_pressure.json', 'w'), indent=1)
t = R['tilt']; print('beta*', round(t['beta_star'], 3), 'theory quality peak at beta', round(t['theory_quality_peak_beta'], 3), round(t['theory_quality_peak'], 3))
for b, th, m, m2 in zip(t['beta'], t['theory_hack'], t['measured_N256'], t['measured_N2048']): print(f"beta {b:5}: theory hack {th:.2f}  N256 hack {m[0]:.2f} q {m[1]:.2f}  N2048 hack {m2[0]:.2f} q {m2[1]:.2f}  theoryQ {tilt(b)[1]:.2f}")
print('budget', {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3)) for k, v in R['budget'].items()})
print('tails', {k: [round(x, 2) for x in v] if k != 'n' else v for k, v in R['tails'].items()})
print('canary sel', [(x, round(a, 2), round(b, 2)) for x, a, b in zip(inv, R['canary_selection']['canary_with'], R['canary_selection']['exploit_without'])])
print('canary train', R['canary_training']); print('leakage', R['leakage'])

# ---------------------------------------------------------------- J. tailoring: the test (input and expected output) is visible before the answer
rng = np.random.default_rng(51); D, V, K = 1000, 10, 10; f = rng.integers(0, V, D)
rew_v, acc_v, rew_c = [], [], []
for ep in range(2000):
    g = rng.integers(0, V, D)                     # an agent with no skill
    T_ = rng.choice(D, K, replace=False)
    gv = g.copy(); gv[T_] = f[T_]                 # visible test: hard-code the expected outputs it can read
    rew_v.append(float((gv[T_] == f[T_]).mean())); acc_v.append(float((gv == f).mean()))
    T2 = rng.choice(D, K, replace=False)          # commit first: the challenge is drawn after the answer is fixed
    rew_c.append(float((g[T2] == f[T2]).mean()))
R['tailoring'] = {'visible_reward': float(np.mean(rew_v)), 'visible_true_accuracy': float(np.mean(acc_v)), 'commit_first_reward': float(np.mean(rew_c))}

# ---------------------------------------------------------------- K. spot checks: a trace of 200 steps with 10% faked; check k random steps
rng = np.random.default_rng(61); out = {}
for k in (5, 20, 50):
    caught = []
    for _ in range(20000):
        fake = rng.random(200) < 0.10; idx = rng.choice(200, k, replace=False); caught.append(fake[idx].any())
    out[str(k)] = {'slip_through': 1 - float(np.mean(caught)), 'analytic': 0.9 ** k}
R['spot_checks'] = out

# ---------------------------------------------------------------- L. wearing out a hidden pool: 100 secret tests reused across episodes, 10 drawn per episode, per-test feedback
def pool_attack(rotate_every=None, Q=4000, seed=71):
    rng = np.random.default_rng(seed); f = rng.integers(0, V, D); g = rng.integers(0, V, D)
    holdout = rng.choice(D, 200, replace=False); rest = np.setdiff1d(np.arange(D), holdout)   # the holdout is never queried
    pool = rng.choice(rest, 100, replace=False); rew, hold = [], []
    for t in range(Q):
        if rotate_every and t % rotate_every == 0 and t: pool = rng.choice(rest, 100, replace=False)
        T_ = rng.choice(pool, K, replace=False); passed = g[T_] == f[T_]; bad = T_[~passed]
        if len(bad): x = rng.choice(bad); g[x] = (g[x] + 1) % V
        rew.append(float(passed.mean())); hold.append(float((g[holdout] == f[holdout]).mean()))
    return {'reward_last200': float(np.mean(rew[-200:])), 'holdout_accuracy_end': hold[-1], 'reward_at_1000': float(np.mean(rew[900:1100]))}
R['wear_out'] = {'reused_pool': pool_attack(None), 'rotated_every_200': pool_attack(200)}
json.dump(R, open('results_pressure.json', 'w'), indent=1)
print('tailoring', R['tailoring']); print('spot', R['spot_checks']); print('wear', R['wear_out'])
