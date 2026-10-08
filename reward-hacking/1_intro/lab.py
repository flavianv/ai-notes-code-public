"""Every toy number in the intro film. Results in results.json. Synthetic toys, fixed seeds, no language model trained.
1. training dynamics: a policy over three actions (honest: reward 1.0, quality 1; hack A and hack B: reward 1.2, quality 0),
   starting at 96% / 2% / 2%, trained by policy gradient (REINFORCE with a mean baseline). Run twice: as is, and with both hacks scored 0.
2. the tilt picture: reused from ../shared/results_pressure.json (best of 256 with temperature beta; 2% exploits scoring 1.2)."""
import json
import numpy as np
R = {}
def train(rewards, p0=(.96, .02, .02), steps=3000, batch=64, lr=.5, seed=0):
    rng = np.random.default_rng(seed); z = np.log(np.array(p0)); r = np.array(rewards); hist = []
    for t in range(steps):
        p = np.exp(z - z.max()); p /= p.sum(); a = rng.choice(3, size=batch, p=p); adv = r[a] - r[a].mean()
        z = z + lr * np.array([(adv * ((a == j) - p[j])).mean() for j in range(3)])
        if t % 100 == 0 or t == steps - 1: hist.append(p.tolist())
    p = np.exp(z - z.max()); p /= p.sum(); return {'final': p.tolist(), 'quality': float(p[0]), 'hist': hist}
R['no_check'] = train([1.0, 1.2, 1.2])
R['fixed_reward'] = train([1.0, 0.0, 0.0])
P3 = json.load(open('../shared/results_pressure.json'))['tilt']
m = dict(zip(P3['beta'], P3['measured_N256'])); th = dict(zip(P3['beta'], P3['theory_hack']))
R['tilt'] = {'beta': P3['beta'], 'measured_hack': [m[b][0] for b in P3['beta']], 'measured_quality': [m[b][1] for b in P3['beta']], 'theory_hack': P3['theory_hack'],
             'beta_star': P3['beta_star'], 'quality_peak_beta': P3['theory_quality_peak_beta'], 'quality_peak': P3['theory_quality_peak']}
json.dump(R, open('results.json', 'w'), indent=1)
print('no check', [round(x, 4) for x in R['no_check']['final']], 'fixed', [round(x, 4) for x in R['fixed_reward']['final']])
print({k: (v if not isinstance(v, list) else [round(x, 3) for x in v]) for k, v in R['tilt'].items()})
