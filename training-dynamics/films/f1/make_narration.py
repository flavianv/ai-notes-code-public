# Builds narration.json (chapter -> subtitle-sized sentences) from script.md and data.js from results.json, and checks every spoken number.
import json, re, math
txt = open("script.md").read()
KEYS = ['courier', 'route', 'mean', 'sqrt', 'matrix', 'fwdbwd', 'attn', 'training', 'softmax', 'range', 'depth', 'norm', 'mlp', 'acts', 'recap', 'refs']
blocks = [b for b in re.split(r"\n## ", txt)[1:] if b[:1].isdigit()]
assert len(blocks) == len(KEYS), len(blocks)
def sentences(p):   # subtitle-sized pieces: merge short sentences, keep each under ~30 words
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
J = lambda k: " ".join(N[k])
r = lambda v, n=2: round(v, n)
def has(k, *ss):
    for s in ss: assert s in J(k), (k, s)
C = K['courier']; assert r(C['linear_small'][-1], 3) == .004 and round(C['linear_big'][-1]) == 186 and r(1 - math.sqrt(.8), 1) == .1 and r(math.sqrt(1.25) - 1, 2) == .12
has('courier', '0.004', '186', '10 percent', '12 percent', '48 layers')
S = K['signs']; assert r(S['var']) == 1.0 and r(S['var_of_3Z']) == 9.0; has('mean', 'nine')
U = K['sums']; assert r(U['std']) == 10.03 and round(K['corr']['at_0.1_formula']) == 1090 and round(K['corr']['at_0.1_measured']) == 1094 and round(math.sqrt(1090)) == 33
has('sqrt', '1,090', '33')
M = K['matrix']; assert round(M['x_norm2']) == 536 and r(M['s2_1']['var'], 1) == 539.5 and r(M['unit_var_expected']) == 1.05 and r(M['s2_1_over_d']['var']) == 1.06 and round(math.sqrt(M['x_norm2'])) == 23
has('matrix', '536', '1.05')
F = K['fwdbwd']; assert r(F['one_over_din']['fwd']) == 1.0 and r(F['one_over_din']['bwd']) == 3.99 and r(F['one_over_dout']['bwd']) == 1.0 and r(F['one_over_dout']['fwd']) == .25
assert r(F['xavier']['s2'], 5) == .00078 and r(F['xavier']['fwd']) == .4 and r(F['xavier']['bwd']) == 1.6
RH = K['relu_half']; assert r(RH['second_moment_in']) == 1.0 and r(RH['second_moment_out']) == .5
RN = K['relu_net']; assert r(RN['one_over_d'][-1], 3) == .001 and r(RN['he'][-1]) == 1.25
has('fwdbwd', '0.00078', '0.4 and the gradient at 1.6', '2,048', '512')
AT = {t['dk']: t for t in K['attn_scale']}
assert [r(AT[d]['raw'], 1) for d in (16, 64, 256, 1024)] == [15.9, 63.9, 255.5, 1014.7] and [r(AT[d]['sqrt']) for d in (16, 64, 256, 1024)] == [.99, 1.0, 1.0, .99] and r(AT[1024]['full'], 3) == .001
RW = K['row']; assert round(RW['raw_max'][2] * 100) == 79 and round(RW['sqrt_max'][2] * 100) == 11 and round(RW['raw_max'][4] * 100) == 95
has('attn', '16, 64, 256 or 1,024', '79 percent', '11 percent', '95 percent')
DP = K['dependence']; assert r(DP['mean'][0]) == 0.0 and r(DP['var'][0]) == 1.0 and r(DP['mean'][-1]) == 7.99
SR = K['softmax_row']; assert [r(v) for v in SR['p']] == [.15, .51, .08, .25] and [r(v) for v in SR['p_x3']] == [.02, .87, .0, .11] and r(SR['entropy']) == 1.19 and r(SR['entropy_x3']) == .47 and SR['shift_max_diff'] < 1e-15
has('training', 'leaves 8', '0.15, 0.51, 0.08 and 0.25', '0.02, 0.87, 0.004 and 0.11', '1.19', '0.47')
ST = K['sat']; assert ST['at_half'] == .25 and r(ST['gap_at_7'], 3) == .001 and round(ST['ce_example']['p'][0] * 100) == 96 and r(ST['ce_example']['grad'][0]) == .96
has('softmax', 'one quarter', '0.999', '0.001', '96 percent', '0.96', '1, 3', '0.25, minus 0.25')
JC = K['jac']; assert round(JC['up48']) == 97 and r(JC['down48'], 3) == .006
RS = K['resid']; assert [r(v, 1) for v in RS['alpha1']] == [12.9, 25.0, 49.4] and [r(v) for v in RS['alpha_isqrt']] == [2.03, 2.01, 1.99]
has('depth', '97', '0.006', '13, 25 and 49', 'The total is 2')
NM = K['norm']; E = NM['example']; assert E['mean'] == 1.5 and E['var'] == 4.25 and [r(v) for v in E['ln']] == [.73, -1.21, -.73, 1.21] and r(E['rms']) == 2.55 and [r(v) for v in E['rmsnorm']] == [1.18, -.39, 0.0, 1.57]
assert r(NM['post_w2'][0], 3) == .097 and r(NM['post_w2'][-1], 3) == .095 and r(NM['pre_w2'][0]) == .72 and r(NM['pre_w2'][-1]) == .17
has('norm', '1.5', '4.25', '0.73, minus 1.21, minus 0.73 and 1.21', '2.55', '1.18, minus 0.39, 0 and 1.57', '0.097', '0.095', '0.72', '0.17', '24 residual blocks')
ML = K['mlp']; assert ML['attn_params'] == 2359296 and ML['mlp_params'] == 4718592 and r(ML['fire_fraction'] * 100, 1) == 50.0 and ML['hidden'] == 128 and ML['dead_units'] == 74 and ML['dead_grad_max'] == 0.0 and ML['kv_error'] < 1e-15 and round(ML['kv_error'] / 1e-16) == 4
has('mlp', '2.36 million', '4.72 million', '50.0 percent', '74', '128')
A = K['act']; assert A['dsigmoid_max'] == .25 and 1e-29 < A['sigmoid_025_48'] < 1.3e-29 and A['ffn_gated_width_768'] == 2048 and A['ffn_check'] and A['squared_relu']['value'] == [.25, 1.0, 4.0]
import numpy as _np
_sil = lambda z: z / (1 + math.exp(-z))
assert r(_sil(1.5)) == 1.23 and r(_sil(1.5) * 2) == 2.45 and r(_sil(-2)) == -.24 and r(_sil(-2) * 3) == -.72
has('acts', 'one quarter', '2,048', 'eight thirds', '0.25', 'SiLU of 1.5 is 1.23', 'which is 2.45', 'minus 0.24', 'minus 0.72')
_xk = _np.array([1., 2., 0.]); _K = _np.array([[2, 0, 0], [0, -.5, 0], [.5, 0, 0], [1, 1, 0]]); _V = _np.array([[1, 0], [0, 1], [1, 1], [0, 2]]); _sc = _K @ _xk; _out = _np.maximum(_sc, 0) @ _V
assert list(_sc) == [2, -1, .5, 3] and list(_out) == [2.5, 6.5]
has('mlp', '2, minus 1, 0.5 and 3', '2.5 and 6.5', 'four entries')
Q = K['qk']; assert r(Q['raw_max'], 1) == 4.1 and round(Q['raw_max_10x']) == 410 and r(Q['norm_max'], 1) == 5.3 and r(Q['cap_at_200']) == 49.97 and r(Q['cap_grad_at_100']) == .07
has('range', 'about 400', '49.97', '0.07')
json.dump(N, open("narration.json", "w"), indent=1)
open("data.js", "w").write("// generated by make_narration.py\nwindow.RES = {};\nwindow.RES0 = {};\nwindow.RES3 = {};\nwindow.LAB = " + json.dumps(K) + ";\n")
for k, v in N.items(): print(f"{k:10s} {len(v):2d} sentences {sum(len(s.split()) for s in v):4d} words")
tot = sum(len(s.split()) for v in N.values() for s in v); print("total", tot, "words, about", round(tot / 2.75 / 60, 1), "min")
