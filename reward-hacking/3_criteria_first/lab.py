"""Every number in the criteria-first film. Results in results.json. Synthetic toy, fixed seeds, no language model trained.
Toy: a judge grades open answers pass or fail. There are m = 10 reasonable ways to frame the criteria for a question
(which aspects count, how strictly). A correct answer passes under every framing. A wrong answer passes a given framing
with probability r = 0.1, independently across framings (a different framing rewards different surface features).
  criteria after the answer: the judge reads the answer, and the answer's framing becomes the criteria; in effect the
                             wrong answer is graded under the framing that suits it best: it passes if any framing passes.
  criteria in advance:       one framing is fixed from the question alone, before any answer is seen.
Best of N: among the N candidates (30% correct), the optimizer submits one that passes, chosen at random among those that pass."""
import json, math
import numpy as np
R = {}
r, m, pc = .1, 10, .3
R['pass_one'] = {'fixed': r, 'adaptive_formula': 1 - (1 - r) ** m, 'm': m, 'r': r}
def run(N, adaptive, m=m, T=40000, seed=1):
    rng = np.random.default_rng(seed); ok = rng.random((T, N)) < pc
    fr = rng.random((T, N, m)) < r
    passes = ok | (fr.any(2) if adaptive else fr[:, :, 0])
    key = np.where(passes, rng.random((T, N)), -1.0); k = key.argmax(1); i = np.arange(T)
    acc = passes[i, k]; return {'wrong_accepted': float((acc & ~ok[i, k]).mean()), 'correct': float(ok[i, k].mean()), 'accepted': float(acc.mean())}
NS = [1, 4, 16, 64]
R['N'] = NS; R['adaptive'] = [run(n, True) for n in NS]; R['fixed'] = [run(n, False) for n in NS]
R['by_m'] = {str(mm): run(16, True, m=mm) for mm in (1, 2, 3, 5, 10)}
# share of accepted answers that are wrong, closed form at large N: (1-pc) a / (pc + (1-pc) a)
share = lambda a: (1 - pc) * a / (pc + (1 - pc) * a)
R['share_formula'] = {'fixed': share(r), 'adaptive': share(1 - (1 - r) ** m)}
json.dump(R, open('results.json', 'w'), indent=1)
print(R['pass_one']); print('adaptive', R['adaptive']); print('fixed', R['fixed']); print('by m', R['by_m']); print(R['share_formula'])
