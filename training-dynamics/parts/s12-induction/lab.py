"""Induction heads, layer by layer. Reads the measurements of the Understanding Transformers A1 experiment
(ai-notes-code/advanced-transformers, tools/run_a1.py, 3 seeds, torch; saved as results/a1/induction_s*.json and the exported
example in understanding-transformers/a1-circuits/data.json) and writes results.json with the numbers used in the film.
Nothing is retrained here: no torch on this machine."""
import json, glob, numpy as np
import pathlib
R = str(pathlib.Path(__file__).resolve().parents[3] / 'advanced-transformers' / 'results' / 'a1') + '/'
EX = {'example': json.load(open('results.json'))['example']}   # the exported example (ids, predictions, attention patterns) is kept in results.json
S = [json.load(open(f)) for f in sorted(glob.glob(R + 'induction_s*.json'))]; assert len(S) == 3
out = dict(seeds=[], example=dict(ids=EX['example']['ids'], pred=EX['example']['pred'], pats=EX['example']['pats']))
for s in S:
    l1, l2 = s['L1'], s['L2']; c2 = np.array(l2['curve']); steps = c2[:, 0]; rep = c2[:, 2]
    # step at which the repeat loss first falls below half of its plateau value
    plateau = float(np.median(rep[:6])); first = int(steps[np.argmax(rep < plateau / 2)])
    prev_l0 = {k: v for k, v in l2['prev'].items() if k.startswith('0.')}; ind_l1 = {k: v for k, v in l2['ind'].items() if k.startswith('1.')}
    out['seeds'].append(dict(seed=s['seed'], acc1=l1['acc']['20'], acc2=l2['acc']['20'], acc2_len50=l2['acc']['50'], drop_step=first, plateau=plateau,
        prev_l0=prev_l0, ind_l1=ind_l1, prev_l1={k: v for k, v in l2['prev'].items() if k.startswith('1.')}, ind_l0={k: v for k, v in l2['ind'].items() if k.startswith('0.')},
        curve1=l1['curve'], curve2=l2['curve'], group=l2['group_ablate'], ablate=l2['ablate']))
    print(s['seed'], 'acc1', round(l1['acc']['20'], 3), 'acc2', round(l2['acc']['20'], 4), 'drop', first, 'len50', round(l2['acc']['50'], 3), 'maxprev', round(max(prev_l0.values()), 3), 'ind', [round(v, 2) for v in ind_l1.values()])
    g = l2['group_ablate']; print('   group', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()}, 'ablate', {k: round(v, 3) if isinstance(v, float) else v for k, v in l2['ablate'].items()})
json.dump(out, open('results.json', 'w'))
print('example ids', out['example']['ids'])
