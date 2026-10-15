# Builds narration.json from script.md and data.js from results.json, and checks every spoken number.
import json, re, math
import numpy as np
txt = open("script.md").read()
KEYS = ['open', 'step', 'momentum', 'rms', 'adam', 'guards', 'prec', 'sched', 'batch', 'matrix', 'muon', 'shampoo', 'qk', 'loop', 'refs']
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
K = json.load(open("results.json")); J = lambda k: " ".join(N[k]); r = lambda v, n=2: round(v, n)
def has(k, *ss):
    for s in ss: assert s in J(k), (k, s)
P = K['parab']; assert [P[k]['eta_h'] for k in ('0.25', '0.75', '1.1')] == [.5, 1.5, 2.2] and P['0.25']['path'][:4] == [1, .5, .25, .125] and [r(v) for v in P['1.1']['path'][:3]] == [1, -1.2, 1.44] and r(P['1.1']['path'][3]) == -1.73
has('step', 'eta h is 1.5', 'eta h is 2.2', '1, 0.5, 0.25, 0.125', 'minus 1.2', 'minus 1.73', 'below two over h')
E_ = K['eos']; assert E_['eta'] == .4 and E_['threshold'] == 5 and r(E_['start_sharpness']) == 2.77 and r(E_['late_mean_sharpness']) == 4.95 and E_['loss_up_events'] == 11
has('step', 'learning rate of 0.4', 'which is 5', '2.77', '4.95', '11 sudden jumps')
M = K['momentum']; assert [r(v) for v in M['steady']] == [1, 1.9, 2.71, 3.44, 4.1, 4.69] and [r(v) for v in M['noisy']] == [1, -.1, .91, -.18, .84, -.25] and r(M['limit']) == 10
has('momentum', '1.9', '2.71', '3.44, 4.10, 4.69', 'which is 10', 'minus 0.1', '0.91', 'minus 0.18, 0.84, minus 0.25')
R = K['rms_steady']; assert R['v'] == [10000, 1e-4] and R['sqrt_v'] == [100, .01] and R['step_over_eta'] == [1, 1] and R['sgd_ratio'] == 10000
has('rms', '10,000 and 0.0001', '100 and 0.01', 'ten thousand times apart')
O = K['optim']['at_120']; assert r(O['sgd'], 1) == 4.5 and r(O['momentum'], 3) == .003 and r(O['adam'], 4) == .0002
has('rms', 'below 0.02', 'still 4.5', 'reaches 0.003', '0.0002')
A = K['adam1']; assert r(A['rmsprop_first_step_over_eta'], 1) == 31.6 and A['adam_first_step_over_eta'] == 1
S = K['adam_steps']; assert r(S[0]['m'], 1) == .1 and r(S[0]['v'], 3) == .001 and r(S[1]['m']) == .39 and r(S[1]['v']) == .01 and r(S[1]['mhat']) == 2.05 and r(S[1]['vhat'], 1) == 5.0 and r(S[1]['step_over_eta']) == .92
has('adam', '3.16', '31.6 times too big', '0.39', '0.01', '0.19', '0.002', '2.05', '5.0', '0.92 eta', '0.9999')
W = K['adamw']; assert W['shrink'] == .9999
W_ = K['warmup']; assert W_['lr'][10] == .2 and W_['lr'][50] == 1
C = K['clip']; assert C['norm'] == 5 and [r(v, 1) for v in C['clipped']] == [.6, .8]
U = K['update_ratio']; assert r(U['W_fro'], 1) == 22.6 and r(U['adam_step_fro'], 3) == .154 and r(U['ratio'], 4) == .0068
has('guards', '0.2 of the maximum', 'which is 5', '0.6 and 0.8', '22.6', '0.154', '0.0068', '262,144')
F = K['fp16']; assert F['max'] == 65504 and F['scale'] == 1024 and r(F['scaled_1e-8'] * 1e5, 3) == 1.025
B = K['bf16']; assert r(B['spacing_at_1'], 4) == .0078 and B['mantissa_bits'] == 7
SO = K['scaling_order']; assert SO['true_norm'] == .5 and SO['scaled_norm'] == 512
has('prec', '65,504', '70,000', '1,024', '1.025 times 10 to the minus 5', '0.0078', '0.5', '512')
Sx = K['sched_ex']; assert Sx['cos_200'] == .1 and r(Sx['cos_300'], 3) == .029 and Sx['wsd_360'] == .1
SC = K['sched']; assert r(SC['final_const'], 3) == .053 and r(SC['predicted_const'], 3) == .056 and r(SC['final_cosine'], 4) == .0053 and r(SC['final_wsd'], 3) == .011
has('sched', 'which is 0.1', '0.029', '0.053', '0.056', '0.0053', '0.011', 'ten-fold')
Bt = K['batch']; assert [r(v, 3) for v in Bt['floor'][:3]] == [.056, .014, .003] and Bt['steps_to_target'] == [None, 11, 8, 8, 8]
assert r(Bt['floor'][2], 4) == .0035
has('batch', '0.0556 over B', '0.056', '0.014', '0.0035', '11 steps', 'always 8')
Sp = K['spectral']; assert Sp['sv'] == [100, 10, .1] and r(Sp['fro'], 1) == 100.5 and Sp['spec'] == 100 and Sp['sgd_ratio'] == [1, .1, .001]
has('matrix', '100.5', '1, 0.1 and 0.001', 'factor of one thousand')
Mu = K['muon']; assert Mu['eta'] == .01 and r(Mu['change_sgd'], 3) == -1.01 and r(Mu['change_ideal'], 3) == -1.101 and Mu['ns']['coeffs'] == [3.4445, -4.775, 2.0315]
n1 = K['ns_first']; assert r(n1['0.995']) == .70 and r(n1['0.1']) == .34 and r(n1['0.001'], 4) == .0034
t5 = Mu['ns']['traj'][5]; assert [r(v) for v in t5] == [.70, .71, .47] and round(t5[2] / 0.0009950366976) == 471
has('muon', '1.010', '1.101', '3.4445', '4.7750', '2.0315', '0.995, 0.1 and 0.001', '0.70', '0.34', '0.0034', '0.70, 0.71 and 0.47', '470')
Sh = K['shampoo2']; assert Sh['L'] == [[4, 0], [0, 1]] and r(Sh['Linv4'][0][0]) == .71 and [r(v) for v in K['shampoo']['sv_G']] == [9.64, 2.46, .31]
has('shampoo', '0.707', '9.64, 2.46 and 0.31', 'the identity')
Q = K['clip_qk']; assert r(Q['S_max']) == 1279.32 and r(Q['gamma'], 3) == .078 and r(Q['sqrt_gamma'], 3) == .280 and r(Q['S_max_after'], 1) == 100
Ml = K['clip_mla']; assert r(Ml['wrong_shared_key_error_head1'], 1) == 40.3 and r(Ml['rot_scale'], 1) == 72.9
has('qk', '1,279', '0.078', '0.280', '100.0', '40.3', '72.9', '15.5 trillion')
json.dump(N, open("narration.json", "w"), indent=1)
open("data.js", "w").write("// generated by make_narration.py\nwindow.RES = {};\nwindow.RES0 = {};\nwindow.RES3 = {};\nwindow.LAB = " + json.dumps(K) + ";\n")
for k, v in N.items(): print(f"{k:10s} {len(v):2d} sentences {sum(len(s.split()) for s in v):4d} words")
tot = sum(len(s.split()) for v in N.values() for s in v); print("total", tot, "words, about", round(tot / 2.75 / 60, 1), "min")
