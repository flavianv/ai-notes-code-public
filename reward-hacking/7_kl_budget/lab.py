"""Numbers for the KL-budget film. Results in results.json. Synthetic toy, fixed seed, no language model trained."""
import json, math
import numpy as np
d = lambda e, p: e * math.log(e / p) + (1 - e) * math.log((1 - e) / (1 - p))   # KL between two coins
kl = lambda q, p: sum(a * math.log(a / b) for a, b in zip(q, p) if a > 0)
def emax(K, p):   # largest exploit rate whose coin KL fits in budget K
    lo, hi = p, 1 - 1e-12
    for _ in range(200):
        m = (lo + hi) / 2; lo, hi = (m, hi) if d(m, p) <= K else (lo, m)
    return lo
R = {}
# the two coins: 1 in 10,000 -> 10%
R['coins'] = {'exploit_term': 0.1 * math.log(0.1 / 1e-4), 'honest_term': 0.9 * math.log(0.9 / 0.9999), 'log_ratio_honest': math.log(0.9 / 0.9999), 'total': d(0.1, 1e-4)}
# three outputs: scale the groups vs move all exploit mass onto A
p = [0.01, 0.01, 0.98]
R['split'] = {'scale': kl([0.05, 0.05, 0.90], p), 'shift': kl([0.10, 0.0, 0.90], p), 'coin': d(0.1, 0.02)}
# budgets for an exploit rate of at most 10%
R['budget'] = {f'{p0:g}': d(0.1, p0) for p0 in (1e-2, 1e-4, 1e-6)}
# best of N on the toy: 2% exploits always score 1.2; honest outputs score their quality ~ N(0.6, 0.15) plus noise 0.1
P, T, rng = 0.02, 200000, np.random.default_rng(7)
NS = [2, 4, 8, 16, 32, 64]; R['bon'] = {'N': NS, 'kl': [], 'bound': [], 'measured': [], 'min_kl': []}
for N in NS:
    ex = rng.random((T, N)) < P; q = np.clip(rng.normal(0.6, 0.15, (T, N)), 0, 1)
    proxy = np.where(ex, 1.2, q) + rng.normal(0, 0.1, (T, N)); pick = ex[np.arange(T), proxy.argmax(1)].mean()
    K = math.log(N) - (N - 1) / N
    R['bon']['kl'].append(K); R['bon']['bound'].append(emax(K, P)); R['bon']['measured'].append(float(pick)); R['bon']['min_kl'].append(d(float(pick), P))
R['quiz1'] = d(0.1, 0.01)
R['abc'] = kl([0.3, 0.3, 0.4], [0.5, 0.3, 0.2])
R['log_growth'] = 0.1 * math.log(100)
R['h01'] = -(0.1 * math.log(0.1) + 0.9 * math.log(0.9))
P3 = json.load(open('../shared/results_pressure.json'))   # the tilt, tails and canary toys (best of 256 with temperature beta; 2% exploits scoring 1.2)
t = P3['tilt']; m = dict(zip(t['beta'], t['measured_N256']))
R['tilt'] = {'beta': t['beta'], 'measured_hack': [m[b][0] for b in t['beta']], 'measured_quality': [m[b][1] for b in t['beta']], 'theory_hack': t['theory_hack'], 'theory_quality': t['theory_quality'],
             'beta_star': t['beta_star'], 'quality_peak_beta': t['theory_quality_peak_beta'], 'quality_peak': t['theory_quality_peak']}
R['tails'] = P3['tails']; R['canary_selection'] = P3['canary_selection']; R['canary_training'] = P3['canary_training']
json.dump(R, open('results.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
