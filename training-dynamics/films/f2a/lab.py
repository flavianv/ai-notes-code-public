"""Every number in film 2A (Gradients, module by module). Small worked examples with exact arithmetic (NumPy, CPU).
Each gradient is also checked against central finite differences (kept for the checks table, not narrated)."""
import json, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
R = {'kind': 'worked examples; no trained model'}
softmax = lambda z: np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)
def fd(f, A, eps=1e-6):
    g = np.zeros_like(A, dtype=float)
    for idx in np.ndindex(*A.shape):
        E = np.zeros_like(A, dtype=float); E[idx] = eps; g[idx] = (f(A + E) - f(A - E)) / (2 * eps)
    return g
FD = {}

# ---------------------------------------------------------------- 1 the loss
z = np.array([2., 1., 0., -1.]); p = softmax(z); y = np.array([1., 0, 0, 0]); words = ['bank', 'water', 'loan', 'steep']
R['loss'] = {'words': words, 'z': z.tolist(), 'exp': np.exp(z).tolist(), 'sum_exp': float(np.exp(z).sum()), 'p': p.tolist(), 'loss': float(-math.log(p[0])), 'grad': (p - y).tolist(),
             'N': 6, 'grad_over_N': ((p - y) / 6).tolist(), 'three_positions_p': [.64, .5, .25], 'three_losses': [-math.log(v) for v in (.64, .5, .25)], 'mean_loss': float(np.mean([-math.log(v) for v in (.64, .5, .25)])),
             'curve_p': np.linspace(0.02, 1, 60).tolist(), 'curve_loss': (-np.log(np.linspace(0.02, 1, 60))).tolist()}
FD['loss'] = float(np.abs(fd(lambda zz: -math.log(softmax(zz)[0]), z) - (p - y)).max())

# ---------------------------------------------------------------- 2 one linear layer
x = np.array([1., 2., -1.]); W = np.array([[.5, -1.], [1., .5], [-.5, 2.]]); gy = np.array([.5, -1.]); eta = 0.1
yv = x @ W; GW = np.outer(x, gy); GX = gy @ W.T; W2 = W - eta * GW; y2 = x @ W2
R['linear'] = {'x': x.tolist(), 'W': W.tolist(), 'y': yv.tolist(), 'gy': gy.tolist(), 'GW': GW.tolist(), 'GX': GX.tolist(), 'eta': eta, 'W_new': W2.tolist(), 'y_new': y2.tolist(),
               'lin_before': float(gy @ yv), 'lin_after': float(gy @ y2), 'drop': float(gy @ yv - gy @ y2), 'fro2': float((GW ** 2).sum())}
FD['linear_W'] = float(np.abs(fd(lambda Wm: (x @ Wm) @ gy, W) - GW).max()); FD['linear_x'] = float(np.abs(fd(lambda xv: (xv @ W) @ gy, x) - GX).max())

# ---------------------------------------------------------------- 3 one matrix, many tokens
tokens = ['the', 'river', 'bank', 'steep']; X = np.array([[1., 0, 0], [0, 1., 1.], [1., 1., 0], [0, 0, 1.]]); G = np.array([[.1, -.2], [.3, .1], [-.4, .2], [.2, .2]]); eta3 = 0.5
parts = [np.outer(X[i], G[i]) for i in range(4)]; GW3 = sum(parts)
R['many'] = {'tokens': tokens, 'X': X.tolist(), 'G': G.tolist(), 'parts': [m.tolist() for m in parts], 'GW': GW3.tolist(), 'W': W.tolist(), 'eta': eta3, 'W_new': (W - eta3 * GW3).tolist(), 'check': float(np.abs(GW3 - X.T @ G).max())}

# ---------------------------------------------------------------- 4 embeddings
vocab = ['the', 'bank', 'river', 'loan', 'by']; sent = ['the', 'bank', 'by', 'the', 'river']; Gx = np.array([[.1, -.2, .3], [-.1, .2, 0.], [.2, 0., -.1], [.3, .1, 0.], [0., -.2, .1]])
GE = np.zeros((5, 3))
for t, w in enumerate(sent): GE[vocab.index(w)] += Gx[t]
R['embed'] = {'vocab': vocab, 'sentence': sent, 'Gx': Gx.tolist(), 'GE': GE.tolist(), 'rows_touched': int((np.abs(GE).sum(1) > 0).sum()), 'the_row': GE[0].tolist()}

# ---------------------------------------------------------------- 5 residual connections
Jf = np.array([[.1, .3], [-.2, .1]]); gout = np.array([1., 1.]); branch = Jf.T @ gout; gres = gout + branch
R['resid'] = {'Jf': Jf.tolist(), 'gout': gout.tolist(), 'branch': branch.tolist(), 'gx': gres.tolist(), 'layers': list(range(0, 13)), 'plain': [0.1 ** k for k in range(13)], 'residual': [1.1 ** k for k in range(13)], 'plain12': 0.1 ** 12, 'residual12': 1.1 ** 12}

# ---------------------------------------------------------------- 6 norm layers
xn = np.array([3., -1., 0., 4.]); dN = 4; rms = math.sqrt((xn ** 2).mean()); yn = xn / rms; gN = np.array([1., 0., 0., 1.])
gxn = (gN - yn * (yn @ gN) / dN) / rms
FD['rmsnorm'] = float(np.abs(fd(lambda v: (v / math.sqrt((v ** 2).mean())) @ gN, xn) - gxn).max())
R['norm'] = {'x': xn.tolist(), 'rms': rms, 'y': yn.tolist(), 'g': gN.tolist(), 'y_dot_g': float(yn @ gN), 'gx': gxn.tolist(), 'gx_dot_x': float(gxn @ xn), 'sq_mean': float((xn ** 2).mean()), 'scale_test_y': (2 * xn / math.sqrt(((2 * xn) ** 2).mean())).tolist()}

# ---------------------------------------------------------------- 7 the MLP
xm = np.array([1., 2., 0.]); K = np.array([[2, 0, 0], [0, -.5, 0], [.5, 0, 0], [1, 1, 0]], float); V = np.array([[1, 0], [0, 1], [1, 1], [0, 2]], float); u = np.array([1., -1.])
s = K @ xm; a = np.maximum(s, 0); out = a @ V; gv = np.outer(a, u); ga = V @ u; gs = ga * (s > 0); gk = np.outer(gs, xm); gxm = gs @ K
FD['mlp_x'] = float(np.abs(fd(lambda v: (np.maximum(K @ v, 0) @ V) @ u, xm) - gxm).max())
R['mlp'] = {'x': xm.tolist(), 'K': K.tolist(), 'V': V.tolist(), 'scores': s.tolist(), 'act': a.tolist(), 'out': out.tolist(), 'u': u.tolist(), 'grad_v': gv.tolist(), 'g_act': ga.tolist(), 'g_pre': gs.tolist(), 'grad_k': gk.tolist(), 'gx': gxm.tolist()}

# ---------------------------------------------------------------- 8 attention
Q = np.array([[1., 0], [0, 1.], [1., 1.]]); Kk = np.array([[1., 0], [0, 1.], [0, 1.]]); Vv = np.array([[1., 0], [0, 1.], [1., 1.]]); U = np.array([[1., 0], [0, 1.], [1., -1.]]); dk = 2; mask = np.tril(np.ones((3, 3), bool))
def attn(Q, Kk, Vv):
    S = Q @ Kk.T / math.sqrt(dk); S = np.where(mask, S, -1e30); A = softmax(S); return A, S
A, S = attn(Q, Kk, Vv); GV = A.T @ U; GA = U @ Vv.T; GS = A * (GA - (A * GA).sum(1, keepdims=True)); GQ = GS @ Kk / math.sqrt(dk); GK = GS.T @ Q / math.sqrt(dk)
L_ = lambda Qm, Km, Vm: (attn(Qm, Km, Vm)[0] @ Vm * U).sum()
FD['attn_Q'] = float(np.abs(fd(lambda m: L_(m, Kk, Vv), Q) - GQ).max()); FD['attn_K'] = float(np.abs(fd(lambda m: L_(Q, m, Vv), Kk) - GK).max()); FD['attn_V'] = float(np.abs(fd(lambda m: L_(Q, Kk, m), Vv) - GV).max())
R['attn'] = {'Q': Q.tolist(), 'K': Kk.tolist(), 'V': Vv.tolist(), 'U': U.tolist(), 'A': A.tolist(), 'GV': GV.tolist(), 'GA': GA.tolist(), 'GS': GS.tolist(), 'GQ': GQ.tolist(), 'GK': GK.tolist(), 'dk': dk, 'sqrt_dk': math.sqrt(dk), 'row2_mean': float((A[2] * GA[2]).sum())}

# ---------------------------------------------------------------- 9 routes
v = np.array([0., 2.]); yv2 = 2.0
def route(zz): a = softmax(zz); h = float(a @ v); uu = h - yv2; return a, h, 0.5 * (h - yv2) ** 2, a * uu * (v - h)
a0, h0, l0, gs0 = route(np.zeros(2)); a1, h1, l1, _ = route(-0.1 * gs0); zz = np.zeros(2); A_t = [float(softmax(zz)[1])]; L_t = [l0]
for _ in range(80): _, _, _, g = route(zz); zz = zz - 0.1 * g; A_t.append(float(softmax(zz)[1])); L_t.append(float(route(zz)[2]))
a_eq = softmax(np.zeros(2)); vv = np.array([1., 1.]); hh = float(a_eq @ vv); eqg = a_eq * (hh - yv2) * (vv - hh)
R['route'] = {'a0': a0.tolist(), 'h0': h0, 'loss0': l0, 'gs': gs0.tolist(), 'a1': a1.tolist(), 'loss1': l1, 'equal_value_grad': eqg.tolist(), 'steps': list(range(81)), 'attn2': A_t, 'loss': L_t}
FD['route'] = float(np.abs(fd(lambda zz_: 0.5 * (softmax(zz_) @ v - yv2) ** 2, np.zeros(2)) - gs0).max())

# ---------------------------------------------------------------- 10 one block, who gets what
R['block'] = {'rows': [['unembedding W_U', 'logits gradient (p - y)', 'a^T (p - y)'], ['final norm', 'gradient through the projection', 'gain: x_hat . g'], ['MLP W_2, W_1', 'g_act = V u, slope of the activation', 'a^T g, x^T g_pre'], ['attention W_V, W_K, W_Q, W_O', 'softmax Jacobian, routes', 'X^T G_V, X^T G_K, X^T G_Q'], ['residual +', 'identity: g splits, both copies', 'none (no weights)'], ['embedding E', 'sparse rows, sums repeats', 'one row per token']]}
R['fd'] = FD
json.dump(R, open('results.json', 'w'), indent=1)
print({k: round(v, 12) for k, v in FD.items()})
print('linear', R['linear']['y'], R['linear']['GW'], R['linear']['GX'], R['linear']['drop'], R['linear']['fro2'])
print('many GW', R['many']['GW']); print('embed', R['embed']['GE']); print('resid', R['resid']['gx'], R['resid']['plain12'], R['resid']['residual12'])
print('norm', R['norm']['gx'], R['norm']['gx_dot_x'], R['norm']['rms'], R['norm']['y']); print('mlp', R['mlp']['g_pre'], R['mlp']['grad_k'], R['mlp']['gx'])
print('attn A', np.round(A, 4).tolist()); print('GS', np.round(GS, 4).tolist(), 'GQ', np.round(GQ, 4).tolist(), 'GK', np.round(GK, 4).tolist(), 'GV', np.round(GV, 4).tolist(), 'GA', GA.tolist())
