"""Every number in the attempted-cheats film. Results in results.json. Synthetic toy, fixed seeds, no language model trained.
Toy: each episode the optimizer draws N candidate answers and keeps the one the evaluator scores highest (best of N).
Honest answers have quality ~ N(0.6, 0.15) and score = quality + noise. Five kinds of cheat ("holes") have quality 0
but fool the evaluator: each has a base rate (how often an untrained model produces it) and a score.
The audit: a trusted check reads A selected answers per round; every hole it sees is patched (the evaluator now scores it -inf)."""
import json, math
import numpy as np
R = {}
HOLES = [  # name, base rate per candidate, evaluator score
    ('A', 0.010, 1.20), ('B', 0.004, 1.15), ('C', 0.002, 1.10), ('D', 0.0005, 1.05), ('E', 0.000002, 1.30)]
R['holes'] = [{'name': n, 'p': p, 'score': s} for n, p, s in HOLES]
P = np.array([h[1] for h in HOLES]); S = np.array([h[2] for h in HOLES])

def episodes(rng, T, N, patched):
    probs = np.concatenate([[1 - P.sum()], P]); typ = rng.choice(6, size=(T, N), p=probs)      # 0 honest, 1..5 holes
    q = np.where(typ == 0, np.clip(rng.normal(.6, .15, (T, N)), 0, 1), 0.0)
    sc = np.where(typ == 0, q, S[np.maximum(typ - 1, 0)]) + rng.normal(0, .1, (T, N))
    for h in patched: sc = np.where(typ == h, -np.inf, sc)
    k = sc.argmax(1); i = np.arange(T); return typ[i, k], q[i, k]

def loop(N, A=50, rounds=8, T=20000, seed=1, patched0=()):
    rng = np.random.default_rng(seed); patched = set(patched0); out = []
    for r in range(rounds + 1):
        t, q = episodes(rng, T, N, patched)
        use = {HOLES[h - 1][0]: float((t == h).mean()) for h in range(1, 6)}
        out.append({'round': r, 'patched': sorted(HOLES[h - 1][0] for h in patched), 'hack_rate': float((t > 0).mean()), 'quality': float(q.mean()), 'use': use})
        audit = t[:A]; patched |= set(int(x) for x in audit[audit > 0])
    return out
R['loop_N256'] = loop(256, A=5)
R['after_N4096'] = loop(4096, A=50, rounds=6, T=4000, patched0=(1, 2, 3, 4), seed=2)
# step 1: how often best of N surfaces a hole
R['surface'] = {'pA_N1': 0.01, 'pA_N256': 1 - (1 - 0.01) ** 256, 'pE_N256': 1 - (1 - 2e-6) ** 256, 'pE_N4096': 1 - (1 - 2e-6) ** 4096, 'pD_N256': 1 - (1 - 5e-4) ** 256}
# step 2: chance an audit of A outputs sees a hole used at rate u
au = lambda u, A=50: 1 - (1 - u) ** A
R['audit_find'] = {'u_0p5': au(0.5), 'u_0p001': au(0.001), 'E_N256': au(R['surface']['pE_N256']), 'E_N4096': au(R['surface']['pE_N4096'])}
uD = float(np.mean([r['use']['D'] for r in R['loop_N256'][2:7]]))
R['rounds'] = {'uD': uD, 'D_find': au(uD, 5), 'D_expected_rounds': 1 / au(uD, 5), 'E256_find': au(R['surface']['pE_N256'], 5), 'E256_expected_rounds': 1 / au(R['surface']['pE_N256'], 5),
               'E4096_find': au(R['surface']['pE_N4096'], 50), 'E4096_expected_rounds': 1 / au(R['surface']['pE_N4096'], 50), 'A_find_round0': au(R['loop_N256'][0]['use']['A'], 5)}
json.dump(R, open('results.json', 'w'), indent=1)
for r in R['loop_N256']: print('N256', r['round'], r['patched'], round(r['hack_rate'], 4), round(r['quality'], 3), {k: round(v, 4) for k, v in r['use'].items() if v > 0})
for r in R['after_N4096']: print('N4096', r['round'], r['patched'], round(r['hack_rate'], 4), round(r['quality'], 3), {k: round(v, 4) for k, v in r['use'].items() if v > 0})
print(R['surface']); print(R['audit_find'])
