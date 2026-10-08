"""Every number in the uncertain-scores film. Results in results.json. Synthetic toy, fixed seeds, no language model trained.
Toy: best of N = 256 candidates scored by an ensemble of K reward models.
  usual honest answers (97%): true quality ~ N(0.5, 0.1); like the training data, so each model's error is small (sd 0.05).
  exploits (2%): true quality 0; unlike the training data, so each model errs a lot (sd 1.0).
  novel good answers (1%): true quality 0.85; also unlike the training data (sd 0.3).
An answer's error splits into a part every model shares (fraction rho of the variance, 0.3 by default) and a part each model has alone."""
import json, math
import numpy as np
R = {}
rng0 = np.random.default_rng(0)
emax = lambda n, T=400000: float(rng0.standard_normal((T // n + 1000, n)).max(1).mean())
R['curse'] = {str(n): emax(n) for n in (1, 2, 4, 5, 8, 16, 256)}   # E[max of n standard normals]

def run(K=5, lam=0.0, rho=0.3, agg='lcb', N=256, T=20000, seed=1):
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=(T, N), p=[.97, .02, .01])              # 0 usual, 1 exploit, 2 novel good
    q = np.where(kind == 0, rng.normal(.5, .1, (T, N)), np.where(kind == 1, 0.0, .85))
    sd = np.where(kind == 0, .05, np.where(kind == 1, 1.0, .3))
    shared = rng.standard_normal((T, N, 1)); own = rng.standard_normal((T, N, K))
    s = q[..., None] + sd[..., None] * (math.sqrt(rho) * shared + math.sqrt(1 - rho) * own)
    if agg == 'min': score = s.min(2)
    else: score = s.mean(2) - lam * (s.std(2, ddof=1) if K > 1 else 0)
    k = score.argmax(1); i = np.arange(T); kk = kind[i, k]
    return {'quality': float(q[i, k].mean()), 'exploit': float((kk == 1).mean()), 'novel': float((kk == 2).mean())}
R['single'] = run(K=1)
R['mean5'] = run(K=5, lam=0)
R['lam'] = {str(l): run(K=5, lam=l) for l in (0, .5, 1, 1.5, 2, 3, 5, 8)}
R['min'] = {str(K): run(K=K, agg='min') for K in (2, 4, 8)}
R['rho'] = {str(r): run(K=5, lam=1.5, rho=r) for r in (0, .3, .6, .9, 1.0)}
R['honest_only_best'] = 0.5 + 0.1 * R['curse']['256']
json.dump(R, open('results.json', 'w'), indent=1)
for k, v in R.items(): print(k, json.dumps(v)[:600])
