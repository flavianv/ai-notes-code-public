"""Every number and curve in film 1 (Trainable at step zero). NumPy on a CPU, seed 20261014, results in results.json.
These are equation checks on random matrices, not trained language models."""
import json, math, warnings
import numpy as np
warnings.filterwarnings('ignore')   # macOS Accelerate emits spurious matmul warnings; results are finite and checked below

SEED = 20261014
rng = np.random.default_rng(SEED)
R = {'seed': SEED, 'kind': 'equation checks on random matrices; no trained model'}

# ---------------------------------------------------------------- 01 the courier through 48 layers
rng = np.random.default_rng(SEED + 1)
D, L = 256, 48
x0 = rng.normal(size=(64, D))                       # 64 couriers of unit size
def stack(sd2_times_d, act):
    x, sizes = x0.copy(), []
    for _ in range(L):
        W = rng.normal(size=(D, D)) * math.sqrt(sd2_times_d / D)
        x = x @ W
        if act == 'relu': x = np.maximum(x, 0)
        sizes.append(float(np.sqrt((x ** 2).mean())))
    return sizes
R['courier'] = {'layers': list(range(1, L + 1)),
                'linear_small': stack(0.8, 'lin'), 'linear_right': stack(1.0, 'lin'), 'linear_big': stack(1.25, 'lin'),
                'relu_quarter': stack(1.0, 'relu'), 'relu_he': stack(2.0, 'relu'), 'relu_big': stack(3.0, 'relu')}
R['courier']['linear_small_48'] = 0.8 ** (L / 2)
R['courier']['linear_big_48'] = 1.25 ** (L / 2)
R['courier']['relu_quarter_48'] = 0.5 ** (L / 2)

# ---------------------------------------------------------------- 02 mean is not magnitude
rng = np.random.default_rng(SEED + 2)
Z = rng.choice([-1., 1.], size=200000)
R['signs'] = {'mean': float(Z.mean()), 'var': float(Z.var()), 'var_of_3Z': float((3 * Z).var()), 'abs_min': float(np.abs(Z).min())}

# ---------------------------------------------------------------- 03 why square roots appear
rng = np.random.default_rng(SEED + 3)
d = 100
S = rng.choice([-1., 1.], size=(20000, d)).sum(1)
edges = np.arange(-40, 41, 4)
hist = np.histogram(S, bins=edges)[0]
R['sums'] = {'d': d, 'n': 20000, 'std': float(S.std()), 'var': float(S.var()), 'aligned': d, 'hist_edges': edges.tolist(), 'hist': hist.tolist()}
rhos = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
meas = []
for rho in rhos:
    g = rng.normal(size=(20000, 1)); e = rng.normal(size=(20000, d))
    Zc = math.sqrt(rho) * g + math.sqrt(1 - rho) * e
    meas.append(float(Zc.sum(1).var()))
R['corr'] = {'rho': rhos, 'measured': meas, 'formula': [d * (1 + (d - 1) * r) for r in rhos], 'at_0.1_measured': meas[4], 'at_0.1_formula': d * (1 + (d - 1) * 0.1)}
walks = rng.choice([-1., 1.], size=(5, 100)).cumsum(1)
R['walks'] = {'random': walks.tolist(), 'aligned': list(range(1, 101))}

# ---------------------------------------------------------------- 04 what a random matrix does
rng = np.random.default_rng(SEED + 4)
d = 512
x = rng.normal(size=d)
R['matrix'] = {'d': d, 'x_norm2': float((x ** 2).sum())}
for name, s2 in (('s2_1_over_d', 1 / d), ('s2_1', 1.0)):
    Y = np.array([x @ (rng.normal(size=(d, 64)) * math.sqrt(s2)) for _ in range(400)]).ravel()
    R['matrix'][name] = {'var': float(Y.var()), 'mean': float(Y.mean()), 'predicted': float(s2 * (x ** 2).sum()),
                         'hist_edges': np.linspace(-6, 6, 25).tolist(), 'hist': np.histogram(Y / math.sqrt(s2 * (x ** 2).sum()), bins=np.linspace(-6, 6, 25))[0].tolist()}
R['matrix']['unit_var_expected'] = float((x ** 2).sum() / d)

# ---------------------------------------------------------------- 05 forward and backward scale
rng = np.random.default_rng(SEED + 5)
din, dout = 512, 2048
def fb(s2):
    W = rng.normal(size=(din, dout)) * math.sqrt(s2)
    xin = rng.normal(size=(4096, din)); gy = rng.normal(size=(4096, dout))
    return float(((xin @ W) ** 2).mean()), float(((gy @ W.T) ** 2).mean())
f1, b1 = fb(1 / din); f2, b2 = fb(2 / (din + dout)); f3, b3 = fb(1 / dout)
R['fwdbwd'] = {'din': din, 'dout': dout, 'one_over_din': {'fwd': f1, 'bwd': b1}, 'xavier': {'fwd': f2, 'bwd': b2, 's2': 2 / (din + dout)}, 'one_over_dout': {'fwd': f3, 'bwd': b3}}
z = rng.normal(size=2000000)
R['relu_half'] = {'second_moment_in': float((z ** 2).mean()), 'second_moment_out': float((np.maximum(z, 0) ** 2).mean())}
# a 20-layer ReLU network: 1/d against He, forward size per layer
def relu_net(s2d, depth=20, dd=512):
    xx = rng.normal(size=(64, dd)); out = []
    for _ in range(depth):
        xx = np.maximum(xx @ (rng.normal(size=(dd, dd)) * math.sqrt(s2d / dd)), 0); out.append(float(np.sqrt((xx ** 2).mean())))
    return out
R['relu_net'] = {'layers': list(range(1, 21)), 'one_over_d': relu_net(1.0), 'he': relu_net(2.0)}

# ---------------------------------------------------------------- 07 the attention scaling
rng = np.random.default_rng(SEED + 7)
dks = [16, 64, 256, 1024]; tab = []
for dk in dks:
    q = rng.normal(size=(20000, dk)); k = rng.normal(size=(20000, dk)); z = (q * k).sum(1)
    tab.append({'dk': dk, 'raw': float(z.var()), 'sqrt': float((z / math.sqrt(dk)).var()), 'full': float((z / dk).var())})
R['attn_scale'] = tab
# a softmax row over 64 keys: how peaked is it when scores are raw, scaled by 1/sqrt(dk), scaled by 1/dk
def row_stats(dk, scale, nk=64, reps=200):
    q = rng.normal(size=(reps, 1, dk)); K = rng.normal(size=(reps, nk, dk)); s = (q * K).sum(-1) * scale
    e = np.exp(s - s.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    return float(p.max(1).mean()), float((-(p * np.log(p + 1e-300)).sum(1)).mean())
dk_list = [4, 16, 64, 256, 1024]
R['row'] = {'dk': dk_list, 'nkeys': 64, 'uniform_entropy': math.log(64),
            'raw_max': [], 'raw_entropy': [], 'sqrt_max': [], 'sqrt_entropy': [], 'full_max': [], 'full_entropy': []}
for dk in dk_list:
    for name, sc in (('raw', 1.0), ('sqrt', 1 / math.sqrt(dk)), ('full', 1 / dk)):
        mx, en = row_stats(dk, sc); R['row'][name + '_max'].append(mx); R['row'][name + '_entropy'].append(en)

# ---------------------------------------------------------------- 08 what training changes
rng = np.random.default_rng(SEED + 8)
zrow = np.array([1.0, 2.2, 0.4, 1.5])
sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
ent = lambda p: float(-(p * np.log(p)).sum())
R['softmax_row'] = {'z': zrow.tolist(), 'p': sm(zrow).tolist(), 'p_shift5': sm(zrow + 5).tolist(), 'shift_max_diff': float(np.abs(sm(zrow) - sm(zrow + 5)).max()),
                    'p_x3': sm(3 * zrow).tolist(), 'entropy': ent(sm(zrow)), 'entropy_x3': ent(sm(3 * zrow))}
R['dot'] = {'random_std': 1.0, 'cos_theta': [0.0, 0.5, 0.9], 'norms_grow': [1, 3], 'aligned_score_at_norm3': 3 * 3 * 0.9}
# score variance with shared structure: q = k + noise
def score_var(mix, dk=64):
    q = rng.normal(size=(20000, dk)); k = mix * q + math.sqrt(1 - mix ** 2) * rng.normal(size=(20000, dk)); return float(((q * k).sum(1) / math.sqrt(dk)).mean()), float(((q * k).sum(1) / math.sqrt(dk)).var())
R['dependence'] = {'mix': [0.0, 0.25, 0.5, 0.75, 1.0], 'mean': [], 'var': []}
for m in R['dependence']['mix']:
    a, b = score_var(m); R['dependence']['mean'].append(a); R['dependence']['var'].append(b)

# ---------------------------------------------------------------- 09 softmax and saturation
rng = np.random.default_rng(SEED + 9)
ps = np.linspace(0.001, 0.999, 200)
R['sat'] = {'p': ps.tolist(), 'sens': (ps * (1 - ps)).tolist(), 'at_half': 0.25, 'at_999': 0.999 * 0.001, 'cross_entropy_grad_wrong': float(0.9 - 0.0)}
logits = np.array([4.0, 0.0, 0.0]); pr = sm(logits)
R['sat']['ce_example'] = {'logits': logits.tolist(), 'p': pr.tolist(), 'target': 1, 'grad': (pr - np.array([0, 1, 0])).tolist()}
# two keys: score gap vs sensitivity
gaps = np.linspace(0, 12, 100); p1 = 1 / (1 + np.exp(-gaps))
R['sat']['gap'] = gaps.tolist(); R['sat']['gap_sens'] = (p1 * (1 - p1)).tolist()
R['sat']['gap_at_7'] = float((1 / (1 + math.exp(-6.9))) * (1 - 1 / (1 + math.exp(-6.9))))   # p = 0.999

# ---------------------------------------------------------------- 10 depth and residuals
rng = np.random.default_rng(SEED + 10)
R['jac'] = {'layers': list(range(0, 49)), 'up': [1.1 ** k for k in range(49)], 'down': [0.9 ** k for k in range(49)], 'up48': 1.1 ** 48, 'down48': 0.9 ** 48}
def resid_var(Ld, alpha, dd=256, reps=64):
    x = rng.normal(size=(reps, dd)); v = [float(x.var())]
    for _ in range(Ld):
        x = x + alpha * rng.normal(size=(reps, dd)); v.append(float(x.var()))
    return v
R['resid'] = {'L': [12, 24, 48], 'alpha1': [resid_var(Ld, 1.0)[-1] for Ld in (12, 24, 48)], 'alpha_isqrt': [resid_var(Ld, 1 / math.sqrt(Ld))[-1] for Ld in (12, 24, 48)],
              'curve_alpha1': resid_var(48, 1.0), 'curve_isqrt': resid_var(48, 1 / math.sqrt(48))}

# ---------------------------------------------------------------- 11 where normalization acts
rng = np.random.default_rng(SEED + 11)
def ln(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True); r = 1 / np.sqrt(var + eps); return (x - mu) * r, (r, (x - mu) * r)
def ln_back(g, cache):
    r, xh = cache; dd = g.shape[-1]; return r * (g - g.mean(-1, keepdims=True) - xh * (g * xh).mean(-1, keepdims=True))
def block_fwd(x, W1, W2):
    h = x @ W1; a = np.maximum(h, 0); return a @ W2, (x, h)
def block_bwd(g, cache, W1, W2):
    x, h = cache; ga = g @ W2.T; gh = ga * (h > 0); return gh @ W1.T
def grad_profile(mode, Ln=24, dd=128, reps=32):
    Ws = [(rng.normal(size=(dd, 4 * dd)) * math.sqrt(2 / dd), rng.normal(size=(4 * dd, dd)) * math.sqrt(2 / (4 * dd))) for _ in range(Ln)]
    x = rng.normal(size=(reps, dd)); caches = []
    for W1, W2 in Ws:
        if mode == 'pre':
            n, lc = ln(x); f, bc = block_fwd(n, W1, W2); caches.append((lc, bc)); x = x + f
        else:
            f, bc = block_fwd(x, W1, W2); s = x + f; y, lc = ln(s); caches.append((lc, bc)); x = y
    g = rng.normal(size=x.shape); g = g / np.sqrt((g ** 2).mean()); out = [float(np.sqrt((g ** 2).mean()))]; pw = []
    for (W1, W2), (lc, bc) in zip(reversed(Ws), reversed(caches)):
        if mode == 'pre':
            gf = g; gx = g + ln_back(block_bwd(g, bc, W1, W2), lc)
        else:
            gs = ln_back(g, lc); gf = gs; gx = gs + block_bwd(gs, bc, W1, W2)
        a = np.maximum(bc[1], 0); pw.append(float(np.sqrt(((a.T @ gf / reps) ** 2).mean())))   # gradient of this block's W2, RMS per entry
        g = gx; out.append(float(np.sqrt((g ** 2).mean())))
    return out[::-1], pw[::-1]    # hidden-state gradient from input to output; W2 gradient per block, first block first
PP = [grad_profile('pre') for _ in range(8)]; QQ = [grad_profile('post') for _ in range(8)]
pre, post = np.mean([a for a, _ in PP], 0), np.mean([a for a, _ in QQ], 0); prew, postw = np.mean([b for _, b in PP], 0), np.mean([b for _, b in QQ], 0)
R['norm'] = {'layers': list(range(0, 25)), 'pre': pre.tolist(), 'post': post.tolist(), 'pre_ratio': float(pre[0] / pre[-1]), 'post_ratio': float(post[0] / post[-1]),
             'blocks': list(range(1, 25)), 'pre_w2': prew.tolist(), 'post_w2': postw.tolist(), 'pre_w2_ratio': float(prew[-1] / prew[0]), 'post_w2_ratio': float(postw[-1] / postw[0])}
# the layer norm and RMS norm formulas on one example vector
v = np.array([3.0, -1.0, 0.0, 4.0]); mu, var = v.mean(), v.var(); R['norm']['example'] = {'v': v.tolist(), 'mean': float(mu), 'var': float(var), 'ln': ((v - mu) / np.sqrt(var + 1e-5)).tolist(), 'rms': float(np.sqrt((v ** 2).mean())), 'rmsnorm': (v / np.sqrt((v ** 2).mean() + 1e-5)).tolist()}

# ---------------------------------------------------------------- 12 the MLP as key-value memory
rng = np.random.default_rng(SEED + 12)
dm = 768
R['mlp'] = {'d': dm, 'attn_params': 4 * dm * dm, 'mlp_params': 8 * dm * dm, 'mlp_share': 8 / 12}
dd, hh = 32, 128
W1 = rng.normal(size=(dd, hh)) / math.sqrt(dd); W2 = rng.normal(size=(hh, dd)) / math.sqrt(hh); xx = rng.normal(size=dd)
mat = np.maximum(xx @ W1, 0) @ W2; kv = sum(max(xx @ W1[:, i], 0) * W2[i] for i in range(hh))
R['mlp']['kv_error'] = float(np.abs(mat - kv).max())
Wbig = rng.normal(size=(dm, 4 * dm)) / math.sqrt(dm); xb = rng.normal(size=(512, dm)); fire = float(((xb @ Wbig) > 0).mean())
R['mlp']['fire_fraction'] = fire
u = rng.normal(size=dd); h = xx @ W1; gk = np.outer(xx, (W2 @ u) * (h > 0)); dead = ~(h > 0)
R['mlp']['dead_units'] = int(dead.sum()); R['mlp']['dead_grad_max'] = float(np.abs(gk[:, dead]).max()) if dead.any() else 0.0; R['mlp']['hidden'] = hh
R['mlp']['unit_fires'] = (h > 0).astype(int).tolist()

# ---------------------------------------------------------------- 13 activation functions
rng = np.random.default_rng(SEED + 13)
from math import erf, exp, sqrt, pi
zz = np.linspace(-4, 4, 161)
sg = 1 / (1 + np.exp(-zz)); gelu = zz * 0.5 * (1 + np.vectorize(erf)(zz / sqrt(2))); silu = zz * sg
Phi = 0.5 * (1 + np.vectorize(erf)(zz / sqrt(2)))
R['act'] = {'z': zz.tolist(), 'sigmoid': sg.tolist(), 'dsigmoid': (sg * (1 - sg)).tolist(), 'tanh': np.tanh(zz).tolist(), 'dtanh': (1 - np.tanh(zz) ** 2).tolist(),
            'relu': np.maximum(zz, 0).tolist(), 'drelu': (zz > 0).astype(float).tolist(), 'gelu': gelu.tolist(), 'dgelu': (Phi + zz * np.exp(-zz ** 2 / 2) / sqrt(2 * pi)).tolist(),
            'silu': silu.tolist(), 'dsilu': (sg + zz * sg * (1 - sg)).tolist(), 'dsigmoid_max': float((sg * (1 - sg)).max()), 'sigmoid_025_48': 0.25 ** 48,
            'ffn_gated_width_768': 8 * 768 // 3, 'ffn_check': 3 * 768 * (8 * 768 // 3) == 2 * 768 * (4 * 768)}
fd = lambda f, z0: (f(z0 + 1e-6) - f(z0 - 1e-6)) / 2e-6
sgm = lambda z0: 1 / (1 + exp(-z0))
R['act']['fd_error'] = max(abs(fd(lambda z0: z0 * sgm(z0), a) - (sgm(a) + a * sgm(a) * (1 - sgm(a)))) for a in (-2., -1., .5, 2.))
R['act']['best_case_grad'] = {'layers': list(range(0, 25)), 'sigmoid': [0.25 ** k for k in range(25)], 'tanh_at0': [1.0] * 25, 'relu_active': [1.0] * 25}
R['act']['squared_relu'] = {'z': [0.5, 1.0, 2.0], 'value': [0.25, 1.0, 4.0], 'grad': [1.0, 2.0, 4.0]}

# ---------------------------------------------------------------- 14 attention scores that stay in range
rng = np.random.default_rng(SEED + 14)
dq = 64
q = rng.normal(size=(256, dq)); k = rng.normal(size=(256, dq))
raw = lambda s: float(((s * q) @ (s * k).T / math.sqrt(dq)).max())
R['qk'] = {'tokens': 256, 'raw_max': raw(1.0), 'raw_max_10x': raw(10.0)}
qn = q / np.linalg.norm(q, axis=1, keepdims=True); kn = k / np.linalg.norm(k, axis=1, keepdims=True)
R['qk']['norm_max'] = float(10 * (qn @ kn.T).max()); R['qk']['norm_max_10x'] = R['qk']['norm_max']
sc = np.linspace(-150, 150, 301); cap = 50
R['qk']['cap'] = cap; R['qk']['s'] = sc.tolist(); R['qk']['capped'] = (cap * np.tanh(sc / cap)).tolist(); R['qk']['cap_grad'] = (1 - np.tanh(sc / cap) ** 2).tolist()
R['qk']['cap_at_200'] = float(cap * math.tanh(200 / cap)); R['qk']['cap_grad_at_100'] = float(1 - math.tanh(100 / cap) ** 2)
# max |score| during a toy "norm growth" curve: raw grows with the product of norms, QK-norm stays at g
growth = np.linspace(1, 10, 28)
R['qk']['curve'] = {'growth': growth.tolist(), 'raw': [float(((g * q) @ (g * k).T / math.sqrt(dq)).max()) for g in growth], 'qknorm': [R['qk']['norm_max']] * len(growth)}

json.dump(R, open('results.json', 'w'), indent=1)
print('ok', {k: (v if not isinstance(v, (dict, list)) else '...') for k, v in R.items()})
for key in ('sums', 'matrix', 'fwdbwd', 'relu_half', 'softmax_row', 'resid', 'norm', 'mlp', 'qk'):
    print(key, json.dumps({k: v for k, v in R[key].items() if not isinstance(v, list) or len(v) < 6})[:600])
print('attn_scale', R['attn_scale']); print('courier ends', {k: round(v[-1], 6) for k, v in R['courier'].items() if isinstance(v, list)})
print('row', {k: [round(a, 3) for a in v] for k, v in R['row'].items() if isinstance(v, list) and k != 'dk'})
print('norm pre/post', R['norm']['pre_ratio'], R['norm']['post_ratio'])
