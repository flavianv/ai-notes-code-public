"""Every number in the hidden-challenges film. Results in results.json. Synthetic toys, fixed seeds, no language model trained.
The task in every toy: a program must map 1,000 inputs to the right one of 10 outputs (a correct table f).
A test is one input; the program passes it if its output equals f's."""
import json, math
import numpy as np
R = {}
D, V = 1000, 10

# 1. a cheat wrong on a fraction p of inputs, checked with n hidden tests drawn after it is committed
p = 0.05; rng = np.random.default_rng(1); wrong = np.zeros(D, bool); wrong[rng.choice(D, int(p * D), replace=False)] = True; p_real = float(wrong.mean())
surv = {}
for n in (1, 4, 20, 50, 90):
    caught = np.array([wrong[rng.integers(0, D, n)].any() for _ in range(40000)])
    surv[str(n)] = {'simulated': 1 - float(caught.mean()), 'formula': (1 - p_real) ** n, 'bound': math.exp(-p_real * n)}
R['survive'] = {'p': p_real, 'n': surv, 'n_for_1pct': math.ceil(math.log(100) / p), 'n_for_1pct_exact': math.ceil(math.log(0.01) / math.log(1 - p))}

# 2. when cheating stops paying: honest 1; cheat 2 if missed, -4 if caught (illustrative payoffs)
q_star = 1 / 6
R['payoff'] = {'q_star': q_star, 'n_needed_p5': math.ceil(math.log(1 - q_star) / math.log(1 - p)), 'q_at_4': 1 - (1 - p) ** 4, 'value_at_4': 2 - 6 * (1 - (1 - p) ** 4)}

# 3. commit first: a program with no skill that can read the 10 test inputs and expected outputs before answering
rng = np.random.default_rng(2); f = rng.integers(0, V, D); rv, av, rc = [], [], []
for _ in range(4000):
    g = rng.integers(0, V, D); T = rng.choice(D, 10, replace=False)
    gv = g.copy(); gv[T] = f[T]; rv.append((gv[T] == f[T]).mean()); av.append((gv == f).mean())
    T2 = rng.choice(D, 10, replace=False); rc.append((g[T2] == f[T2]).mean())
R['commit'] = {'visible_reward': float(np.mean(rv)), 'visible_true_acc': float(np.mean(av)), 'commit_reward': float(np.mean(rc))}

# 4. wearing out a hidden pool: 10 tests per episode drawn from a pool; the trainee sees which tests failed and changes one output
def climb(pool_size=None, rotate=None, Q=4000, seed=3):
    rng = np.random.default_rng(seed); f = rng.integers(0, V, D); g = rng.integers(0, V, D)
    hold = rng.choice(D, 200, replace=False); rest = np.setdiff1d(np.arange(D), hold)   # never-queried inputs measure real skill
    pool = rest if pool_size is None else rng.choice(rest, pool_size, replace=False); rew, acc = [], []
    for t in range(Q):
        if rotate and t and t % rotate == 0: pool = rng.choice(rest, pool_size, replace=False)
        T = rng.choice(pool, 10, replace=False); ok = g[T] == f[T]; bad = T[~ok]
        if len(bad): x = rng.choice(bad); g[x] = (g[x] + 1) % V
        rew.append(ok.mean()); acc.append((g[hold] == f[hold]).mean())
    return {'reward_end': float(np.mean(rew[-200:])), 'reward_1000': float(np.mean(rew[900:1100])), 'holdout_end': float(acc[-1]), 'curve_reward': [float(np.mean(rew[i:i + 100])) for i in range(0, Q, 100)], 'curve_holdout': [float(acc[i]) for i in range(0, Q, 100)]}
R['wear'] = {'fixed_pool_100': climb(100), 'rotated_200': climb(100, 200)}
json.dump(R, open('results.json', 'w'), indent=1)
for k, v in R.items(): print(k, {a: b for a, b in v.items() if 'curve' not in a} if k != 'wear' else {a: {c: d for c, d in b.items() if 'curve' not in c} for a, b in v.items()})
