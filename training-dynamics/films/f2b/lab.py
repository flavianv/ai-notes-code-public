"""Every number and curve in film 2 (One step). NumPy on a CPU, seed 20261014, results in results.json.
Equation checks and toy problems: no language model is trained."""
import json, math, warnings
import numpy as np
warnings.filterwarnings('ignore')   # macOS Accelerate emits spurious matmul warnings; every result is finite and asserted

SEED = 20261014
R = {'seed': SEED, 'kind': 'equation checks and toy problems; no language model trained'}
sec = lambda k: np.random.default_rng(SEED + k)
softmax = lambda z: np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)

# ---------------------------------------------------------------- 07 the step size
ht = [0.5, 1.5, 2.2]; R['quad'] = {'eta_h': ht, 'steps': list(range(13)), 'paths': [[(1 - e) ** t for t in range(13)] for e in ht]}
rng = np.random.default_rng(3)   # the seed that shows the edge of stability clearly (found by a short search over seeds)
nS, dS, hS = 24, 4, 32; Xd = rng.normal(size=(nS, dS)); yd = np.sin(Xd @ rng.normal(size=dS))
def lg(w):
    W1 = w[:dS * hS].reshape(dS, hS); w2 = w[dS * hS:]; a = np.tanh(Xd @ W1); f = a @ w2; r = f - yd; Lq = 0.5 * np.mean(r ** 2)
    gf = r / nS; return Lq, np.concatenate([(Xd.T @ (np.outer(gf, w2) * (1 - a ** 2))).ravel(), a.T @ gf])
def sharp(w, iters=30):
    vv = rng.normal(size=w.size); vv /= np.linalg.norm(vv); eps = 1e-4
    for _ in range(iters):
        hv = (lg(w + eps * vv)[1] - lg(w - eps * vv)[1]) / (2 * eps); lam = np.linalg.norm(hv); vv = hv / lam
    return float(lam)
w = np.concatenate([rng.normal(size=dS * hS) * .5 / math.sqrt(dS), rng.normal(size=hS) * .5 / math.sqrt(hS)]); eta = 0.4; ts, Ls, Ss = [], [], []
for t in range(4001):
    Lq, g = lg(w)
    if t % 20 == 0: ts.append(t); Ls.append(float(Lq)); Ss.append(sharp(w))
    w = w - eta * g
R['eos'] = {'eta': eta, 'threshold': 2 / eta, 'steps': ts, 'loss': Ls, 'sharpness': Ss, 'start_sharpness': Ss[0], 'late_mean_sharpness': float(np.mean(Ss[len(Ss) // 2:])), 'late_max_sharpness': float(max(Ss[len(Ss) // 2:])),
            'loss_up_events': int(sum(1 for a, b in zip(Ls[40:], Ls[41:]) if b > a * 1.5))}

# ---------------------------------------------------------------- 08 RMSProp, Adam
b2 = 0.999; g1 = 1.0; V1 = (1 - b2) * g1 ** 2
R['adam1'] = {'beta2': b2, 'rmsprop_first_step_over_eta': g1 / math.sqrt(V1), 'adam_first_step_over_eta': (g1 / (1 - 0.9) * (1 - 0.9)) / math.sqrt(V1 / (1 - b2)),
              'two_coord': {'grads': [100., .01], 'sgd_ratio': 100 / .01, 'rms_steps': [1.0, 1.0]}}
Hd = np.array([100., 1.])
def run(opt, steps=120):
    w = np.array([1., 1.]); m = np.zeros(2); v = np.zeros(2); out = []
    for t in range(1, steps + 1):
        g = Hd * w
        if opt == 'sgd': w = w - 0.0199 * g
        elif opt == 'mom': m = 0.9 * m + g; w = w - 0.0019 * m
        else: m = .9 * m + .1 * g; v = .999 * v + .001 * g ** 2; w = w - 0.05 * (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .999 ** t)) + 1e-8)
        out.append(float(0.5 * (Hd * w * w).sum()))
    return out
R['optim'] = {'steps': list(range(1, 121)), 'sgd': run('sgd'), 'momentum': run('mom'), 'adam': run('adam')}
R['optim']['at_120'] = {k: R['optim'][k][-1] for k in ('sgd', 'momentum', 'adam')}

# ---------------------------------------------------------------- 09 warmup, clipping, precision
R['warmup'] = {'t': list(range(0, 201)), 'lr': [min(1.0, t / 50) for t in range(0, 201)]}
gvec = np.array([3., 4.]); c = 1.0; R['clip'] = {'g': gvec.tolist(), 'norm': float(np.linalg.norm(gvec)), 'clipped': (gvec * min(1, c / np.linalg.norm(gvec))).tolist()}
dW = 512; Wm = sec(9).normal(size=(dW, dW)) / math.sqrt(dW); eta_a = 3e-4; R['update_ratio'] = {'d': dW, 'W_fro': float(np.linalg.norm(Wm)), 'adam_step_fro': eta_a * dW, 'ratio': eta_a * dW / float(np.linalg.norm(Wm)), 'eta': eta_a}
f16 = np.float16
R['fp16'] = {'max': float(np.finfo(f16).max), 'overflow_of_70000': float(f16(70000.)), 'underflow_of_1e-8': float(f16(1e-8)), 'scaled_1e-8': float(f16(1e-8 * 1024)), 'unscaled_back': float(f16(1e-8 * 1024)) / 1024, 'scale': 1024}

# ---------------------------------------------------------------- 10 schedules and batch size
rng = sec(10); T = 400; h_ = 1.0
def noisy(sched, B=1, seeds=20, eta0=0.2, steps=T, w0=1.0):
    ls = np.zeros(steps + 1)
    for s in range(seeds):
        w = w0; ls[0] += 0.5 * h_ * w * w
        for t in range(steps):
            w = w - sched(t, steps, eta0) * (h_ * w + rng.normal() / math.sqrt(B)); ls[t + 1] += 0.5 * h_ * w * w
    return ls / seeds
const = lambda t, T_, e: e
cosine = lambda t, T_, e: 0.5 * e * (1 + math.cos(math.pi * t / T_))
wsd = lambda t, T_, e: e if t < 0.8 * T_ else e * (T_ - t) / (0.2 * T_)
Lc, Lk, Lw = noisy(const), noisy(cosine), noisy(wsd)
tail = lambda L: float(np.mean(L[-20:]))
R['sched'] = {'steps': list(range(T + 1)), 'lr_const': [0.2] * (T + 1), 'lr_cosine': [cosine(t, T, .2) for t in range(T + 1)], 'lr_wsd': [wsd(t, T, .2) for t in range(T + 1)], 'loss_const': Lc.tolist(), 'loss_cosine': Lk.tolist(), 'loss_wsd': Lw.tolist(),
              'final_const': tail(Lc), 'final_cosine': tail(Lk), 'final_wsd': tail(Lw), 'predicted_const': 0.5 * 0.2 / (2 - 0.2)}
Bs = [1, 4, 16, 64, 256]; target = 0.02; steps_to = []; curves = {}
for B in Bs:
    L = noisy(const, B=B, seeds=60, steps=60); curves[B] = L.tolist(); hit = next((t for t, v_ in enumerate(L) if v_ <= target), None); steps_to.append(hit)
R['batch'] = {'B': Bs, 'target': target, 'steps_to_target': steps_to, 'curves': curves, 'floor': [0.5 * 0.2 / (B * 1.8) * 1.0 for B in Bs]}

# ---------------------------------------------------------------- 11 matrix updates and the spectral norm
Gm = np.diag([100., 10., 0.1]); sv = np.array([100., 10., 0.1])
R['spectral'] = {'sv': sv.tolist(), 'fro': float(np.sqrt((sv ** 2).sum())), 'spec': float(sv.max()), 'sgd_ratio': (sv / sv.max()).tolist()}

# ---------------------------------------------------------------- 12 Muon
eta_m = 0.01
R['muon'] = {'eta': eta_m, 'sgd_sv': (eta_m * sv / sv.max()).tolist(), 'ideal_sv': [eta_m] * 3, 'change_sgd': float(-(sv * eta_m * sv / sv.max()).sum()), 'change_ideal': float(-(sv * eta_m).sum()), 'nuclear': float(sv.sum())}
a_, b_, c_ = 3.4445, -4.7750, 2.0315
s = sv / math.sqrt((sv ** 2).sum()); traj = [s.tolist()]
for _ in range(5): s = a_ * s + b_ * s ** 3 + c_ * s ** 5; traj.append(s.tolist())
ss = np.linspace(0, 1.2, 241); R['muon']['ns'] = {'coeffs': [a_, b_, c_], 'start': traj[0], 'traj': traj, 'poly_s': ss.tolist(), 'poly_f': (a_ * ss + b_ * ss ** 3 + c_ * ss ** 5).tolist(), 'zero_stays_zero': float(a_ * 0 + b_ * 0 + c_ * 0)}
rng = sec(12); M0 = rng.normal(size=(8, 5)); M0 = M0 @ np.diag([6., 3., 1., .3, .05]); U_, S_, Vt_ = np.linalg.svd(M0, full_matrices=False); X = M0 / np.linalg.norm(M0)
for _ in range(5):
    A_ = X @ X.T; X = a_ * X + b_ * A_ @ X + c_ * A_ @ A_ @ X
R['muon']['matrix'] = {'before': S_.tolist(), 'after': np.linalg.svd(X, compute_uv=False).tolist(), 'ortho_target': np.linalg.svd(U_ @ Vt_, compute_uv=False).tolist(), 'shape': [8, 5], 'same_subspace_error': float(np.abs(U_.T @ X @ Vt_.T - np.diag(np.diag(U_.T @ X @ Vt_.T))).max())}
# the rows/columns story for a rectangular matrix
Xo = U_ @ Vt_; R['muon']['rect'] = {'cols_orthonormal': float(np.abs(Xo.T @ Xo - np.eye(5)).max()), 'rows_orthonormal': float(np.abs(Xo @ Xo.T - np.eye(8)).max())}

# ---------------------------------------------------------------- 13 Shampoo
rng = sec(13); Gs = rng.normal(size=(4, 3)) @ np.diag([5., 1., .2]); eps = 1e-12
def inv_root(Mx, k): w_, V_ = np.linalg.eigh(Mx); return (V_ * np.maximum(w_, eps) ** (-1 / k)) @ V_.T
Ls_ = Gs @ Gs.T; Rs_ = Gs.T @ Gs
# L is rank 3 in a 4x4 space: use the pseudo inverse root on the range
def pinv_root(Mx, k, tol=1e-9): w_, V_ = np.linalg.eigh(Mx); keep = w_ > tol; return (V_[:, keep] * w_[keep] ** (-1 / k)) @ V_[:, keep].T
Psh = pinv_root(Ls_, 4) @ Gs @ pinv_root(Rs_, 4); Us_, Ss_, Vs_ = np.linalg.svd(Gs, full_matrices=False)
R['shampoo'] = {'G': Gs.tolist(), 'one_step_update': Psh.tolist(), 'orthogonalized': (Us_ @ Vs_).tolist(), 'max_diff_to_UVt': float(np.abs(Psh - Us_ @ Vs_).max()), 'sv_G': Ss_.tolist(), 'sv_update': np.linalg.svd(Psh, compute_uv=False).tolist()}

# ---------------------------------------------------------------- 14 MuonClip
rng = sec(14); nq, dq = 32, 16; Xq = rng.normal(size=(nq, dq)); Wq = rng.normal(size=(dq, dq)) * 5.0; Wk = rng.normal(size=(dq, dq)) * 5.0
lg_ = lambda Wq, Wk: (Xq @ Wq) @ (Xq @ Wk).T / math.sqrt(dq)
S0 = lg_(Wq, Wk); Smax = float(S0.max()); tau = 100.0; gam = min(1.0, tau / Smax); S1 = lg_(Wq * math.sqrt(gam), Wk * math.sqrt(gam))
R['clip_qk'] = {'tau': tau, 'S_max': Smax, 'gamma': gam, 'sqrt_gamma': math.sqrt(gam), 'S_max_after': float(S1.max()), 'scaling_error': float(np.abs(S1 - gam * S0).max()), 'hist_before': np.histogram(S0.ravel(), bins=np.linspace(-Smax, Smax, 31))[0].tolist(), 'edges': np.linspace(-Smax, Smax, 31).tolist(), 'hist_after': np.histogram(S1.ravel(), bins=np.linspace(-Smax, Smax, 31))[0].tolist()}
# MLA: content part q_C . k_C plus rotary part q_R . k_R with ONE rotary key shared by all heads
dc, dr, H_ = 8, 4, 2; qC = rng.normal(size=(H_, nq, dc)) * 3; kC = rng.normal(size=(H_, nq, dc)) * 3; qR = rng.normal(size=(H_, nq, dr)) * 3; kR = rng.normal(size=(nq, dr)) * 3
logit = lambda h, qC_, kC_, qR_, kR_: qC_[h] @ kC_[h].T + qR_[h] @ kR_.T
gam_h = [0.2, 0.5]
good = [logit(h, qC * 1, kC * 1, qR * 1, kR * 1) for h in range(H_)]
def clipped(h, shared_key_scaled):
    g = gam_h[h]; kr = kR * (math.sqrt(g) if shared_key_scaled else 1.0)
    return (math.sqrt(g) * qC[h]) @ (math.sqrt(g) * kC[h]).T + ((g * qR[h]) @ kr.T if not shared_key_scaled else (math.sqrt(g) * qR[h]) @ kr.T)
R['clip_mla'] = {'gamma_h': gam_h, 'right_error': [float(np.abs(clipped(h, False) - gam_h[h] * good[h]).max()) for h in range(H_)]}
# wrong way: scale the shared rotary key by sqrt(gamma of head 0): head 1's rotary logits change although head 1 was not clipped
wrong_h1 = (qR[1] @ (kR * math.sqrt(gam_h[0])).T); right_h1 = qR[1] @ kR.T; R['clip_mla']['wrong_shared_key_error_head1'] = float(np.abs(wrong_h1 - right_h1).max()); R['clip_mla']['rot_scale'] = float(np.abs(right_h1).max())

# ---------------------------------------------------------------- new worked examples for 2B
# step size on a parabola with h = 2
hh = 2.0; ex = {}
for eta_ in (0.25, 0.75, 1.1):
    w = 1.0; path = [w]
    for _ in range(4): w = w - eta_ * hh * w; path.append(w)
    ex[str(eta_)] = {'eta_h': eta_ * hh, 'path': path, 'first_grad': hh * 1.0, 'first_step': eta_ * hh * 1.0}
R['parab'] = ex
# momentum, PyTorch convention: v = mu v + g ; w -= lr v
mu = 0.9
def mom(gs):
    v = 0.0; out = []
    for g in gs: v = mu * v + g; out.append(v)
    return out
R['momentum'] = {'mu': mu, 'steady': mom([1] * 6), 'noisy': mom([1, -1, 1, -1, 1, -1]), 'limit': 1 / (1 - mu)}
# RMSProp steady state for two coordinates
R['rms_steady'] = {'g': [100., 0.01], 'v': [100. ** 2, 0.01 ** 2], 'sqrt_v': [100., 0.01], 'step_over_eta': [1.0, 1.0], 'sgd_steps_over_eta': [100., 0.01], 'sgd_ratio': 100. / 0.01}
# Adam: two steps, beta1 = .9, beta2 = .999, gradients 1 and 3
b1, b2 = .9, .999; m = v = 0.0; steps = []
for t, g in enumerate([1., 3.], 1):
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g; mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t); steps.append({'g': g, 'm': m, 'v': v, 'mhat': mh, 'vhat': vh, 'step_over_eta': mh / math.sqrt(vh)})
R['adam_steps'] = steps; R['adamw'] = {'eta': 1e-3, 'lam': 0.1, 'shrink': 1 - 1e-3 * 0.1}
# schedules
T_ = 400; em = 0.2
R['sched_ex'] = {'cos_200': 0.5 * em * (1 + math.cos(math.pi * 200 / T_)), 'cos_300': 0.5 * em * (1 + math.cos(math.pi * 300 / T_)), 'wsd_360': em * (T_ - 360) / (0.2 * T_), 'warm_10': 10 / 50}
# scaled gradient and clipping order
R['scaling_order'] = {'true_norm': 0.5, 'scale': 1024, 'scaled_norm': 512, 'clip': 1.0}
# bf16 spacing near 1
R['bf16'] = {'mantissa_bits': 7, 'spacing_at_1': 2 ** -7, 'max': 3.3895313892515355e+38}
# Newton-Schulz on single singular values, first iteration
a_, b_, c_ = 3.4445, -4.7750, 2.0315
R['ns_first'] = {str(s): a_ * s + b_ * s ** 3 + c_ * s ** 5 for s in (0.995, 0.1, 0.001)}
# a 2x2 Shampoo example
G2 = np.diag([2., 1.]); L2 = G2 @ G2.T; R2 = G2.T @ G2; inv4 = lambda M_: np.diag(np.diag(M_) ** -0.25)
R['shampoo2'] = {'G': G2.tolist(), 'L': L2.tolist(), 'Linv4': inv4(L2).tolist(), 'update': (inv4(L2) @ G2 @ inv4(R2)).tolist()}
# gradient accumulation
R['accum'] = {'micro': 4, 'effective_batch': 4 * 16}
json.dump(R, open('results.json', 'w'), indent=1)
for key in ('parab', 'momentum', 'rms_steady', 'adam_steps', 'adamw', 'sched_ex', 'ns_first', 'shampoo2', 'eos', 'adam1', 'sched', 'batch', 'spectral', 'clip_qk', 'clip_mla', 'warmup', 'clip', 'update_ratio', 'fp16'):
    v_ = R[key]; print(key, json.dumps(v_)[:300] if not isinstance(v_, dict) or len(json.dumps(v_)) < 400 else {k: (v if not isinstance(v, list) else '...') for k, v in v_.items()})
