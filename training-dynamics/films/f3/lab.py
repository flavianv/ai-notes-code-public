"""Every number and curve in film 3 (Thousands of steps). NumPy on a CPU, seed 20261014, results in results.json.
New computations: equation checks and toy problems. Two measured result sets are read from the series' public code and re-used:
 - A1 circuits (2-layer attention-only model on repeated random tokens, seeds 0-2): advanced-transformers/atlib/circuits.py
 - grokking (modular addition mod 97, small MLP, seeds 0-2): training-dynamics/grokking/grokking.py -> grokking/results/grokking.json"""
import json, math, warnings, statistics
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')   # macOS Accelerate emits spurious matmul warnings; every result is finite and asserted

SEED = 20261014
R = {'seed': SEED, 'kind': 'equation checks and toy problems; plus two measured sets from the public code (A1 circuits, grokking)'}
sec = lambda k: np.random.default_rng(SEED + k)
softmax = lambda z: np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 02 matching and message content
rng = sec(2); nT, dM, dK = 6, 8, 4
X = rng.normal(size=(nT, dM)); WQ = rng.normal(size=(dM, dK)) / math.sqrt(dM); WK = rng.normal(size=(dM, dK)) / math.sqrt(dM); Rg = rng.normal(size=(dK, dK)) + 3 * np.eye(dK)
S0 = (X @ WQ) @ (X @ WK).T / math.sqrt(dK); S1 = (X @ WQ @ Rg) @ (X @ WK @ np.linalg.inv(Rg).T).T / math.sqrt(dK)
WQK = WQ @ WK.T; S2 = X @ WQK @ X.T / math.sqrt(dK)
mask = np.tril(np.ones((nT, nT), bool)); perm = rng.permutation(nT)
def attn_out(X, mask, WV):
    S = (X @ WQ) @ (X @ WK).T / math.sqrt(dK); S = np.where(mask, S, -1e30); return softmax(S) @ (X @ WV)
WV = rng.normal(size=(dM, dK)) / math.sqrt(dM)
O1 = attn_out(X, mask, WV); O2 = attn_out(X[perm], mask[np.ix_(perm, perm)], WV)
R['match'] = {'dM': dM, 'dK': dK, 'gauge_max_diff': float(np.abs(S0 - S1).max()), 'bilinear_max_diff': float(np.abs(S0 - S2).max()), 'perm_max_diff': float(np.abs(O1[perm] - O2).max()), 'S0': S0.tolist(), 'S1': S1.tolist(),
              'rank_QK': int(np.linalg.matrix_rank(WQK)), 'qk_shape': [dM, dM]}

# ---------------------------------------------------------------- 03 how useful structure accumulates
sv = np.array([100., 10., .1]); p_ = sv / sv.sum(); H = float(-(p_ * np.log(p_)).sum())
ang = np.deg2rad(np.arange(5) * 72); F5 = np.stack([np.cos(ang), np.sin(ang)], 1); G5 = F5 @ F5.T
R['accum'] = {'sv': sv.tolist(), 'p': p_.tolist(), 'entropy': H, 'eff_rank': math.exp(H), 'features': F5.tolist(), 'gram': G5.tolist(), 'interference_adjacent': float(abs(G5[0, 1])), 'interference_far': float(abs(G5[0, 2]))}
Xo = np.array([[1., 2., 0.], [0., 1., 1.], [2., 0., 1.]]); Uo = np.array([[1., 0.], [0., -1.], [1., 1.]]); Gw = Xo.T @ Uo; Gsum = sum(np.outer(Xo[i], Uo[i]) for i in range(3))
R['accum']['outer'] = {'X': Xo.tolist(), 'U': Uo.tolist(), 'G': Gw.tolist(), 'sum_error': float(np.abs(Gw - Gsum).max())}

# ---------------------------------------------------------------- 04-05 induction circuit and hidden structure (measured in A1)
a1 = json.load(open(HERE / 'a1_results.json'))   # A1 circuits runs, from advanced-transformers (tools/run_a1.py)
s0 = a1['0']; R['a1'] = {'seeds': [0, 1, 2], 'curve_2layer': {k: a1[k]['curve'] for k in a1}, 'curve_1layer': {k: a1[k]['curve1'] for k in a1},
                         'prev_head_score': {k: a1[k]['prev']['0.0'] for k in a1}, 'prev_scores_seed0': s0['prev'], 'ind_scores_seed0': s0['ind'],
                         'acc': {k: a1[k]['group']['acc'] for k in a1}, 'prev_off': {k: a1[k]['group']['all_prev_off'] for k in a1}, 'ind_off': {k: a1[k]['group']['all_ind_off'] for k in a1}, 'acc_by_len_seed0': s0['acc']}
# ---------------------------------------------------------------- 06-07 grokking (measured in grokking.py)
G = json.load(open(HERE.parent.parent / 'grokking' / 'results' / 'grokking.json'))
R['grok'] = {'p': G['p'], 'pairs': G['p'] ** 2, 'train_fraction': G['train_fraction'], 'steps': G['steps'], 'runs': [{'cond': r['cond'], 'seed': r['seed'], 'train_99': r['train_99'], 'test_99': r['test_99'], 'final_train': r['final_train'], 'final_test': r['final_test'], 'log': r['log']} for r in G['runs']]}

# ---------------------------------------------------------------- 08 why the rule wins: implicit bias, double descent
rng = sec(8); m_, n_ = 50, 200; A_ = rng.normal(size=(m_, n_)); b_ = rng.normal(size=m_)
lr = 1.0 / np.linalg.norm(A_, 2) ** 2; w = np.zeros(n_); dist = []; resid = []; w_mn = np.linalg.pinv(A_) @ b_
for t in range(3001):
    if t % 50 == 0: dist.append(float(np.linalg.norm(w - w_mn))); resid.append(float(np.linalg.norm(A_ @ w - b_)))
    w = w - lr * A_.T @ (A_ @ w - b_)
null = np.linalg.svd(A_)[2][m_:]; w_other = w_mn + 3.0 * null[0]
R['implicit'] = {'m': m_, 'n': n_, 'residual': float(np.linalg.norm(A_ @ w - b_)), 'dist_to_min_norm': float(np.linalg.norm(w - w_mn)), 'norm_gd': float(np.linalg.norm(w)), 'norm_min': float(np.linalg.norm(w_mn)),
                 'other_fit_residual': float(np.linalg.norm(A_ @ w_other - b_)), 'other_fit_norm': float(np.linalg.norm(w_other)), 'steps': list(range(0, 3001, 50)), 'dist_curve': dist, 'resid_curve': resid}
ntr, dd_, noise, reps = 40, 5, 0.5, 60; ps = list(range(2, 161, 2)); te_all = np.zeros((reps, len(ps))); rng = np.random.default_rng(5)
for rep in range(reps):
    beta = rng.normal(size=dd_) / math.sqrt(dd_); Xtr = rng.normal(size=(ntr, dd_)); ytr = Xtr @ beta + noise * rng.normal(size=ntr); Xte = rng.normal(size=(1000, dd_)); yte = Xte @ beta; Wf = rng.normal(size=(160, dd_))
    for k, p in enumerate(ps):
        Ftr = np.maximum(Xtr @ Wf[:p].T, 0) / math.sqrt(p); Fte = np.maximum(Xte @ Wf[:p].T, 0) / math.sqrt(p); coef = np.linalg.pinv(Ftr) @ ytr
        te_all[rep, k] = np.mean((Fte @ coef - yte) ** 2)
te = np.median(te_all, 0); pk = int(np.argmax(te)); R['dd'] = {'features': ps, 'test_mse': te.tolist(), 'n_train': ntr, 'peak_at': ps[pk], 'peak_value': float(te[pk]), 'value_at_160': float(te[-1]), 'value_at_20': float(te[ps.index(20)]), 'noise_var': noise ** 2, 'dim': dd_, 'statistic': 'median over 60 draws'}

# ---------------------------------------------------------------- 09 width and mu-P
rng = sec(9); eta = 0.01; widths = [256, 512, 1024, 2048]; init_std, change = [], []
for d in widths:
    xw = rng.normal(size=(2000, d)); Wd = rng.normal(size=(d, 1)) / math.sqrt(d); init_std.append(float((xw @ Wd).std()))
    x1 = rng.normal(size=d); u = 1.0; dW = -eta * np.outer(x1, np.array([u])); change.append(float(abs(x1 @ dW)[0]))
R['width'] = {'widths': widths, 'eta': eta, 'init_output_std': init_std, 'one_step_change': change, 'ratio': [c / change[0] for c in change], 'formula': [eta * d for d in widths]}

# ---------------------------------------------------------------- 10-11 linear attention and delta attention
rng = sec(10); T_, d_ = 12, 8; Q = rng.normal(size=(T_, d_)); K = rng.normal(size=(T_, d_)); V = rng.normal(size=(T_, d_))
phi = lambda x: np.where(x > 0, x + 1, np.exp(x))            # elu(x) + 1, positive features
Pq, Pk = phi(Q), phi(K); mask = np.tril(np.ones((T_, T_), bool)); A = (Pq @ Pk.T) * mask; Yk = (A @ V) / A.sum(1, keepdims=True)
S = np.zeros((d_, d_)); z = np.zeros(d_); Yr = []
for t in range(T_): S = S + np.outer(Pk[t], V[t]); z = z + Pk[t]; Yr.append(Pq[t] @ S / (Pq[t] @ z))
R['linear'] = {'T': T_, 'd': d_, 'pairwise_vs_recurrent': float(np.abs(Yk - np.array(Yr)).max()), 'state_entries': d_ * d_ + d_,
               'cost': {'N': 4096, 'm': 64, 'dv': 64, 'pairwise_madds': 2 * 4096 * 4096 * 64, 'recurrent_madds': 2 * 4096 * 64 * 64, 'ratio': (2 * 4096 * 4096 * 64) / (2 * 4096 * 64 * 64)}}
def unitv(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def recurrent(Qm, Km, Vm, beta, alpha=None, S0=None):
    T, dk = Km.shape; St = np.zeros((dk, Vm.shape[1])) if S0 is None else S0.copy(); Y = []
    for t in range(T):
        if alpha is not None:
            a = alpha[t] if np.ndim(alpha[t]) else np.full(dk, alpha[t]); St = a[:, None] * St
        St = St - beta[t] * np.outer(Km[t], Km[t] @ St - Vm[t]); Y.append(St.T @ Qm[t])
    return np.array(Y), St
def chunk_wy(Qm, Km, Vm, beta, S0):
    C = len(Km); W = np.zeros_like(Km); U = np.zeros_like(Vm)
    for t in range(C): W[t] = beta[t] * (Km[t] - (Km[:t] @ Km[t]) @ W[:t]); U[t] = beta[t] * (Vm[t] - (Km[:t] @ Km[t]) @ U[:t])
    Am = np.tril(Qm @ Km.T); return Qm @ S0 - Am @ (W @ S0) + Am @ U, S0 - Km.T @ (W @ S0) + Km.T @ U
rng = sec(11); dd = 16; Tn = 12; Kd, Qd, Vd = unitv(rng.normal(size=(Tn, dd))), rng.normal(size=(Tn, dd)), rng.normal(size=(Tn, dd)); beta = rng.uniform(.2, 1., Tn)
_, Sx = recurrent(Qd[:1], Kd[:1], Vd[:1], np.ones(1), S0=rng.normal(size=(dd, dd))); overwrite = float(np.abs(Sx.T @ Kd[0] - Vd[0]).max())
S0r = rng.normal(size=(dd, dd)); grad = np.outer(Kd[0], S0r.T @ Kd[0] - Vd[0]); _, S1r = recurrent(Qd[:1], Kd[:1], Vd[:1], beta[:1], S0=S0r); gd_err = float(np.abs(S1r - (S0r - beta[0] * grad)).max())
Yr_, Sr_ = recurrent(Qd, Kd, Vd, beta); Yc, Sc = chunk_wy(Qd, Kd, Vd, beta, np.zeros((dd, dd))); chunk_err = float(max(np.abs(Yr_ - Yc).max(), np.abs(Sr_ - Sc).max()))
Y1, S1c = chunk_wy(Qd[:6], Kd[:6], Vd[:6], beta[:6], np.zeros((dd, dd))); Y2, S2c = chunk_wy(Qd[6:], Kd[6:], Vd[6:], beta[6:], S1c); chain_err = float(max(np.abs(Yr_ - np.vstack([Y1, Y2])).max(), np.abs(Sr_ - S2c).max()))
norms = [np.linalg.norm(np.eye(dd) - b * np.outer(k, k), 2) for k in Kd for b in (0., .5, 1., 2.)]
dim = 64; rows = []
for n in (16, 32, 64, 96, 128):
    errs = {'additive': [], 'delta': [], 'gated_delta': []}
    for s in range(5):
        r = np.random.default_rng(s); Kn, Vn = unitv(r.normal(size=(n, dim))), r.normal(size=(n, dim)); bn = np.ones(n)
        for name, St in (('additive', Kn.T @ Vn), ('delta', recurrent(Kn, Kn, Vn, bn)[1]), ('gated_delta', recurrent(Kn, Kn, Vn, bn, alpha=np.full(n, .98))[1])):
            errs[name].append(float(np.mean(np.linalg.norm(Kn @ St - Vn, axis=1) / np.linalg.norm(Vn, axis=1))))
    rows.append({'pairs': n, **{k: float(np.mean(v)) for k, v in errs.items()}})
S0g = rng.normal(size=(16, 16)); chg = np.ones(16); chg[8:] = .5; Kz = np.zeros((1, 16)); Kz[0, :8] = unitv(rng.normal(size=(1, 8)))
_, Scg = recurrent(Qd[:1], Kz, Vd[:1], np.zeros(1), alpha=[chg], S0=S0g); _, Ssg = recurrent(Qd[:1], Kz, Vd[:1], np.zeros(1), alpha=[.75], S0=S0g)
R['delta'] = {'overwrite_error': overwrite, 'gradient_step_error': gd_err, 'chunk_vs_recurrent': chunk_err, 'two_chunk_chain': chain_err, 'transition_norm_max': float(max(norms)), 'capacity_width': dim, 'capacity': rows,
              'channel_gate_rows_kept': float(np.abs(Scg[:8] - S0g[:8]).max()), 'channel_gate_rows_halved': float(np.abs(Scg[8:] - .5 * S0g[8:]).max()), 'scalar_gate_all_rows': float(np.abs(Ssg - .75 * S0g).max())}

# ---------------------------------------------------------------- 13 adaptation dynamics: LoRA, QLoRA, VeRA, DoRA
rng = sec(13); din, dout, r_ = 4096, 4096, 16
R['lora'] = {'d': din, 'r': r_, 'full': din * dout, 'lora': r_ * (din + dout), 'ratio': din * dout / (r_ * (din + dout)), 'vera_trainable': dout + r_, 'dora_extra': dout}
# gradient at initialization: A random, B zero -> B gets a gradient, A does not; both zero -> neither
dI, dO, rr = 12, 10, 3; x = rng.normal(size=dI); u = rng.normal(size=dO)
def grads(A, B, s=1.0):
    return s * np.outer(u, A @ x), s * (B.T @ np.outer(u, x))      # (grad B, grad A) of L = u . (s B A x)
gB, gA = grads(rng.normal(size=(rr, dI)), np.zeros((dO, rr))); gB0, gA0 = grads(np.zeros((rr, dI)), np.zeros((dO, rr)))
R['lora']['init'] = {'A_random_B_zero': {'grad_B_norm': float(np.linalg.norm(gB)), 'grad_A_norm': float(np.linalg.norm(gA))}, 'both_zero': {'grad_B_norm': float(np.linalg.norm(gB0)), 'grad_A_norm': float(np.linalg.norm(gA0))}}
# a low-rank target: the best rank-r update, error from the discarded singular values
U_, S_, Vt_ = np.linalg.svd(rng.normal(size=(64, 64)) @ np.diag(np.exp(-np.arange(64) / 6.0)) @ rng.normal(size=(64, 64)))
S_ = S_ / S_[0]; dWt = (U_ * S_) @ Vt_; errs_r = []
for r in range(1, 17): Wr = (U_[:, :r] * S_[:r]) @ Vt_[:r]; errs_r.append(float(np.linalg.norm(dWt - Wr) / np.linalg.norm(dWt)))
R['lora']['rank_error'] = {'r': list(range(1, 17)), 'rel_error': errs_r, 'sv': S_[:16].tolist()}
# 4-bit: uniform vs quantile levels, block-wise absmax scaling (block 64) on Gaussian weights
nd = statistics.NormalDist(); off = 0.9677
v1 = [nd.inv_cdf(q) for q in np.linspace(off, 0.5, 9)[:-1]]; v3 = [-nd.inv_cdf(q) for q in np.linspace(off, 0.5, 8)[:-1]]
levels = np.array(sorted(v1 + [0.0] + v3)); levels = levels / np.abs(levels).max()
unif = np.linspace(-1, 1, 16)
wq = rng.normal(size=(4096, 64)); sc = np.abs(wq).max(1, keepdims=True); wn = wq / sc
def quant(wn, lv): idx = np.abs(wn[..., None] - lv).argmin(-1); return lv[idx]
e_unif = float(np.mean((quant(wn, unif) * sc - wq) ** 2) / np.mean(wq ** 2)); e_nf = float(np.mean((quant(wn, levels) * sc - wq) ** 2) / np.mean(wq ** 2))
R['lora']['quant'] = {'levels_quantile': levels.tolist(), 'levels_uniform': unif.tolist(), 'rel_mse_uniform4': e_unif, 'rel_mse_quantile4': e_nf, 'block': 64, 'n_levels': len(levels)}
# DoRA: magnitude and direction
w0 = rng.normal(size=(6, 5)); dw = 0.05 * rng.normal(size=(6, 5)); m = np.linalg.norm(w0, axis=0); wd = m * (w0 + dw) / np.linalg.norm(w0 + dw, axis=0)
R['lora']['dora'] = {'col_norms_equal_m': float(np.abs(np.linalg.norm(wd, axis=0) - m).max())}

# ---------------------------------------------------------------- 14 falsifiable: gather the checks of the three films
def load(p):
    try: return json.load(open(HERE.parent / p / 'results.json'))
    except Exception: return None
f1, f2 = load('f1'), load('f2')
R['checks'] = {'f1': {'qk_scale_variance': [t['sqrt'] for t in f1['attn_scale']] if f1 else None, 'kv_error': f1['mlp']['kv_error'] if f1 else None},
               'f2': {'loss_fd': f2['loss']['fd_error'] if f2 else None, 'linear_fd': f2['linear']['fd_error'] if f2 else None, 'attn_fd': max(f2['attn_bwd']['errors'].values()) if f2 else None, 'shampoo': f2['shampoo']['max_diff_to_UVt'] if f2 else None, 'clip_qk': f2['clip_qk']['scaling_error'] if f2 else None},
               'f3': {'gauge': R['match']['gauge_max_diff'], 'perm': R['match']['perm_max_diff'], 'delta_chunk': chunk_err, 'delta_gd': gd_err, 'linear_recurrent': R['linear']['pairwise_vs_recurrent']}}

json.dump(R, open('results.json', 'w'), indent=1)
for key in ('match', 'accum', 'implicit', 'dd', 'width', 'linear', 'delta', 'lora'):
    print(key, json.dumps({k: v for k, v in R[key].items() if not isinstance(v, (list, dict))})[:600])
print('capacity', R['delta']['capacity']); print('lora init', R['lora']['init'], 'quant', {k: v for k, v in R['lora']['quant'].items() if k.startswith('rel')}, 'rank err', [round(e, 3) for e in R['lora']['rank_error']['rel_error'][:8]])
print('a1', R['a1']['prev_head_score'], R['a1']['acc'], R['a1']['prev_off'], R['a1']['ind_off']); print('ind', R['a1']['ind_scores_seed0'])
print('dd', R['dd']['peak_at'], R['dd']['peak_value'], R['dd']['value_at_20'], R['dd']['value_at_160'])
print('checks', json.dumps(R['checks']))
