# Builds narration.json (chapter -> subtitle-sized sentences) from script.md and data.js from results.json, and checks every spoken number.
import json, re, math
txt = open("script.md").read()
KEYS = ['question', 'noise', 'grok', 'bias', 'wrap', 'refs']
blocks = [b for b in re.split(r"\n## ", txt)[1:] if b[:1].isdigit()]
assert len(blocks) == len(KEYS), len(blocks)
def sentences(p):
    ss = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", p); out = []
    for s in ss:
        if out and len(out[-1].split()) + len(s.split()) <= 30 and len(s.split()) < 5: out[-1] += " " + s
        else: out.append(s)
    return out
N = {}
for key, block in zip(KEYS, blocks):
    vo = block.split("**Voice-over:**")[1].split("**Reader:**")[0].replace("---", "").strip()
    paras = [p.replace("\n", " ").replace("**", "").strip() for p in vo.split("\n\n")]
    N[key] = [s for p in paras if p for s in sentences(p)]
K = json.load(open("results.json"))
J = lambda k: " ".join(N.get(k, []))
r = lambda v, n=2: round(v, n)
def has(k, *ss):
    if k not in N: return
    for s in ss: assert s in J(k), (k, s)
M = K['match']; assert round(M['gauge_max_diff'] * 1e16) == 7 and round(M['bilinear_max_diff'] * 1e16) == 3 and round(M['perm_max_diff'] * 1e16) == 2 and M['rank_QK'] == 4 and M['dM'] == 8 and M['dK'] == 4
has('match', '8 by 8')
AC = K['accum']; assert r(AC['entropy']) == .31 and r(AC['eff_rank'], 1) == 1.4 and r(AC['interference_adjacent']) == .31 and r(AC['interference_far']) == .81 and AC['outer']['sum_error'] < 1e-12
has('accumulate', '0.31', '1.4', '0.81', '72 degrees')
A1 = K['a1']; assert r(A1['prev_head_score']['0'], 2) == .96 and [round(v * 100) for k, v in A1['ind_scores_seed0'].items() if k.startswith('1.')] == [90, 87, 89, 88] and r(A1['acc']['0'] * 100, 1) == 99.7 and round(A1['prev_off']['0'] * 100) == 7 and round(A1['ind_off']['0'] * 100) == 2
has('induction', '96 percent', '87 to 90 percent', '99.7 percent', '7 percent', '2 percent', '4 heads' if False else 'four heads')
c2 = {x[0]: x for x in A1['curve_2layer']['0']}; c1 = {x[0]: x for x in A1['curve_1layer']['0']}
assert r(c2[400][2], 1) == 3.5 and r(c2[500][2], 1) == 1.6 and r(c2[600][2], 1) == .6 and r(c2[2950][2]) == .05 and r(math.log(49), 1) == 3.9
avg400 = (c2[400][1] + c2[400][2]) / 2; avg600 = (c2[600][1] + c2[600][2]) / 2; assert r(avg400, 1) == 3.8 and r(avg600, 1) == 2.4, (avg400, avg600)
ones = [c1[s][2] for s in range(300, 800, 50) if s in c1]; assert r(min(ones), 1) >= 3.5 and r(max(ones), 1) <= 3.8 and all(3.9 <= c2[s][1] <= 4.4 for s in (300, 400, 500, 600))
has('hidden', '3.5 at step 400', '1.6 at step 500', '0.6 at step 600', '0.05 by step 3,000', 'from 3.8 to 2.4', 'between 3.5 and 3.8', '3.9')
G = K['grok']; runs = {(x['cond'], x['seed']): x for x in G['runs']}; n_tr = int(G['train_fraction'] * G['pairs']); assert G['pairs'] == 9409 and n_tr == 2822 and G['pairs'] - n_tr == 6587 and G['p'] == 97
rnd = [runs[('random', s)] for s in range(3)]; assert all(x['train_99'] == 100 for x in rnd) and [r(x['final_test'] * 100) for x in rnd] == [1.05, 1.02, 1.05] and r(100 / 97) == 1.03
has('noise', '9,409', '2,822', '6,587', '1.0 to 1.05 percent', '1.03 percent', 'step 100')
gk = [runs[('grok', s)] for s in range(3)]; nd = [runs[('no_decay', s)] for s in range(3)]
assert [x['test_99'] for x in gk] == [11600, 10600, 12000] and all(x['train_99'] == 100 for x in gk) and min(x['test_99'] for x in gk) / 100 > 100
assert [round(x['log'][0][3]) for x in gk] == [113, 113, 112] and [round(x['log'][-1][3]) for x in gk] == [64, 62, 63] and all(x['final_test'] == 0 for x in nd)
assert [round(x['log'][1][3]) for x in nd] == [124, 124, 123] and [round(x['log'][-1][3]) for x in nd] == [286, 290, 287] and all(max(l[2] for l in x['log'][10:]) <= .0005 for x in nd)
has('grok', 'step 11,600', '10,600', '12,000', '113', 'between 62 and 64', '0.05 percent', '124', 'between 286 and 291', '20,000')
IM = K['implicit']; assert round(IM['residual'] * 1e16) == 7 and r(IM['dist_to_min_norm'] * 1e15, 1) == 1.3 and r(IM['norm_gd']) == .62 and r(IM['other_fit_norm']) == 3.06 and round(IM['other_fit_norm'] / IM['norm_min']) == 5 and IM['m'] == 50 and IM['n'] == 200
D = K['dd']; ix = D['features'].index; assert r(D['test_mse'][ix(12)]) == .29 and D['peak_at'] == 40 and round(D['peak_value']) == 47 and r(D['value_at_160']) == .63 and D['n_train'] == 40
has('bias', '0.62', '3.06', '0.29', '47', '0.63', '50 equations and 200 unknowns', '40 noisy points')
W = K['width']; assert W['widths'] == [256, 512, 1024, 2048] and all(r(s, 1) == 1.0 for s in W['init_output_std']) and [r(c, 1) for c in W['one_step_change']] == [2.6, 5.0, 10.2, 20.4]
has('scale', '2.6, 5.1, 10 and 20', '256, 512, 1,024 and 2,048', '0.01', '20 tokens per parameter')
LN_ = K['linear']; assert round(LN_['pairwise_vs_recurrent'] * 1e16) == 2 and LN_['cost']['pairwise_madds'] == 2147483648 and round(LN_['cost']['recurrent_madds'] / 1e6) == 34 and LN_['cost']['ratio'] == 64 and LN_['state_entries'] == 72
has('linear', '2.1 billion', '34 million', '64 times fewer', '72 numbers', '4,096')
DL = K['delta']; assert round(DL['overwrite_error'] * 1e16) == 9 and DL['gradient_step_error'] < 1e-12 and r(DL['chunk_vs_recurrent'] * 1e15, 1) == 1.8 and DL['transition_norm_max'] < 1 + 1e-12 and DL['channel_gate_rows_kept'] == 0 and DL['channel_gate_rows_halved'] < 1e-12
cap = {c['pairs']: c for c in DL['capacity']}; assert [r(cap[16][k]) for k in ('additive', 'delta')] == [.47, .33] and [r(cap[64][k]) for k in ('additive', 'delta')] == [.98, .72] and [r(cap[128][k]) for k in ('additive', 'delta')] == [1.41, .99] and r(cap[96]['gated_delta']) == .87 and r(cap[96]['delta']) == .89 and cap[64]['gated_delta'] > cap[64]['delta']
has('delta', '0.47', '0.33', '0.98', '0.72', '1.41', '0.99', '0.87 against 0.89', '64-wide', 'rows 0 to 7')
has('frontier', '93 layers', '744 billion', '15.5 trillion', 'Muon', 'a thousand')
LR = K['lora']; assert LR['full'] == 16777216 and LR['lora'] == 131072 and LR['ratio'] == 128 and LR['vera_trainable'] == 4112 and r(LR['init']['A_random_B_zero']['grad_B_norm'], 1) == 17.6 and LR['init']['A_random_B_zero']['grad_A_norm'] == 0 and LR['init']['both_zero']['grad_B_norm'] == 0
e = LR['rank_error']['rel_error']; assert [r(e[0]), r(e[3]), r(e[7])] == [.85, .47, .21] and r(LR['quant']['rel_mse_quantile4'], 4) == .0085 and r(LR['quant']['rel_mse_uniform4'], 4) == .0100 and LR['quant']['n_levels'] == 16 and LR['dora']['col_norms_equal_m'] < 1e-12
has('adapt', '131,072', '16,777,216', '128 times', '17.6', '0.85', '0.47', '0.21', '0.0085', '0.0100', '4,112')
C = K['checks']; assert [r(v) for v in C['f1']['qk_scale_variance']] == [.99, 1.0, 1.0, .99] and round(C['f2']['attn_fd'] * 1e10, 1) == 5.8 and round(C['f2']['shampoo'] * 1e14) == 2 and round(C['f2']['clip_qk'] * 1e14) == 4 and r(C['f3']['delta_chunk'] * 1e15, 1) == 1.8
has('falsify', '5.8 times ten to the minus 10', '2 times ten to the minus 14', '4 times ten to the minus 14', '1.8 times ten to the minus 15')
json.dump(N, open("narration.json", "w"), indent=1)
open("data.js", "w").write("// generated by make_narration.py\nwindow.RES = {};\nwindow.RES0 = {};\nwindow.RES3 = {};\nwindow.LAB = " + json.dumps(K) + ";\n")
for k, v in N.items(): print(f"{k:10s} {len(v):2d} sentences {sum(len(s.split()) for s in v):4d} words")
tot = sum(len(s.split()) for v in N.values() for s in v); print("total", tot, "words, about", round(tot / 2.75 / 60, 1), "min")
