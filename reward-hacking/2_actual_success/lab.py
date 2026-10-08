"""Every number in the actual-success film. Results in results.json. Synthetic toy, fixed seeds, no language model trained.
Toy: a coding task, best of N candidate programs. 20% are correct. 5% are confident fakes: wrong, polished, and their summary
claims every test passed. The other 75% are plain wrong. Half of the non-fake candidates look polished, independently of correctness.
  appearance grader: score = 0.3 * correct + 1.0 * polished + 0.5 * claims_success + noise(sd 0.3)   (it reads the code and the summary)
  outcome checker:   runs the program on the full specification; accepts if and only if it is correct (sound, delta = 0)"""
import json, math
import numpy as np
R = {}
def run(N, grader, T=20000, seed=1):
    rng = np.random.default_rng(seed); u = rng.random((T, N)); ok = u < .2; fake = (u >= .2) & (u < .25); pol = (rng.random((T, N)) < .5) | fake
    if grader == 'appearance': s = .3 * ok + 1.0 * pol + .5 * fake + rng.normal(0, .3, (T, N))
    else: s = ok + rng.random((T, N)) * 1e-3          # accept iff correct; ties broken at random
    k = s.argmax(1); i = np.arange(T); acc = ok[i, k] if grader == 'checker' else np.ones(T, bool)
    return {'correct': float(ok[i, k].mean()), 'false_accept': float(((~ok[i, k]) & acc).mean()), 'fake': float(fake[i, k].mean())}
NS = [1, 4, 16, 64, 256]
R['N'] = NS
R['appearance'] = [run(n, 'appearance') for n in NS]
R['checker'] = [run(n, 'checker') for n in NS]
R['checker_formula'] = [1 - .8 ** n for n in NS]      # P(at least one correct among N)
R['union'] = {'delta': 1e-6, 'T': 10000, 'bound': 1e-6 * 10000, 'exact_independent': 1 - (1 - 1e-6) ** 10000}
json.dump(R, open('results.json', 'w'), indent=1)
for n, a, c, f in zip(NS, R['appearance'], R['checker'], R['checker_formula']): print(n, a, c, round(f, 4))
print(R['union'])
