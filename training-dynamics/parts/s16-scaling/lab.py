"""Scaling-law toy: train tiny character-level MLP language models of several sizes N on several data sizes D, fit
L(N,D) = E + A/N^alpha + B/D^beta, derive the compute-optimal split under C = 6 N D, and compare with the constants
published by Hoffmann et al. (2022). Pure NumPy. Run: python3 lab.py  -> results.json"""
import json, time, numpy as np
rng0 = np.random.default_rng(0)
text = open('shakespeare.txt').read()
chars = sorted(set(text)); V = len(chars); idx = {c: i for i, c in enumerate(chars)}
data = np.array([idx[c] for c in text], dtype=np.int64)
K = 6                                              # context length (characters)
def windows(arr):                                  # (n, K) contexts and (n,) targets
    n = len(arr) - K; X = np.stack([arr[i:i + n] for i in range(K)], 1); return X, arr[K:K + n]
val_X, val_y = windows(data[-60000:])
train_pool = data[:-60000]
CONFIGS = [(4, 8), (6, 16), (8, 32), (12, 64), (16, 128), (24, 256)]      # (embedding dim, hidden width)
DS = [12800, 51200, 204800, 819200]                  # tokens seen: one pass over D unique training positions
BATCH, LR = 128, 3e-3
def init(e, h, rng):
    return {'E': rng.normal(0, .3, (V, e)), 'W1': rng.normal(0, 1 / np.sqrt(K * e), (K * e, h)), 'b1': np.zeros(h), 'W2': rng.normal(0, 1 / np.sqrt(h), (h, V)), 'b2': np.zeros(V)}
def nparams(P): return sum(v.size for v in P.values())
def forward(P, X):
    emb = P['E'][X].reshape(len(X), -1); a = np.tanh(emb @ P['W1'] + P['b1']); z = a @ P['W2'] + P['b2']
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True); return emb, a, p
def loss_of(P, X, y):
    _, _, p = forward(P, X); return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
def train(e, h, D, seed):
    rng = np.random.default_rng(seed); P = init(e, h, rng); Xtr, ytr = windows(train_pool[:D + K]); n = len(ytr); perm = rng.permutation(n); STEPS = n // BATCH; warm = max(10, STEPS // 20)
    M = {k: np.zeros_like(v) for k, v in P.items()}; S = {k: np.zeros_like(v) for k, v in P.items()}
    for t in range(1, STEPS + 1):
        b = perm[(t - 1) * BATCH: t * BATCH]; X, y = Xtr[b], ytr[b]; emb, a, p = forward(P, X)
        dz = p.copy(); dz[np.arange(BATCH), y] -= 1; dz /= BATCH
        G = {'W2': a.T @ dz, 'b2': dz.sum(0)}; da = dz @ P['W2'].T; dpre = da * (1 - a * a)
        G['W1'] = emb.T @ dpre; G['b1'] = dpre.sum(0); demb = dpre @ P['W1'].T; G['E'] = np.zeros_like(P['E']); np.add.at(G['E'], X.reshape(-1), demb.reshape(-1, P['E'].shape[1]))
        lr = LR * min(1, t / warm) * (0.5 * (1 + np.cos(np.pi * t / STEPS)) * .9 + .1)
        for k in P:
            M[k] = .9 * M[k] + .1 * G[k]; S[k] = .999 * S[k] + .001 * G[k] ** 2
            P[k] -= lr * (M[k] / (1 - .9 ** t)) / (np.sqrt(S[k] / (1 - .999 ** t)) + 1e-8)
    return nparams(P), loss_of(P, val_X, val_y)
t0 = time.time(); Ns = []; L = []
for (e, h) in CONFIGS:
    row = []
    for D in DS:
        N, l = train(e, h, D, seed=e * 1000 + D % 997); row.append(l)
    Ns.append(N); L.append(row); print(f'N={N:6d}', ['%.3f' % v for v in row], f'{time.time() - t0:.0f}s', flush=True)
Ns = np.array(Ns, float); Ds = np.array(DS, float); L = np.array(L)
# fit L = E + A N^-alpha + B D^-beta: for fixed (alpha, beta) linear in (E, A, B); grid over the exponents
NN, DD = np.meshgrid(Ns, Ds, indexing='ij'); y = L.ravel(); best = None
for al in np.arange(.05, 1.6, .01):
    for be in np.arange(.05, 1.6, .01):
        X = np.stack([np.ones(y.size), (NN ** -al).ravel(), (DD ** -be).ravel()], 1); c, *_ = np.linalg.lstsq(X, y, rcond=None)
        if c[0] < 0 or c[1] < 0 or c[2] < 0: continue
        r = float(np.sqrt(np.mean((X @ c - y) ** 2)))
        if best is None or r < best[0]: best = (r, al, be, c)
rmse, al, be, c = best; Efit, Afit, Bfit = [float(v) for v in c]
def optimal(E, A, B, alpha, beta, C):              # minimise E + A N^-a + B D^-b subject to C = 6 N D
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta)); N = G * (C / 6) ** (beta / (alpha + beta)); D = C / (6 * N); return float(N), float(D)
H = dict(E=1.69, A=406.4, B=410.7, alpha=.34, beta=.28)    # Hoffmann et al. (2022), parametric fit as reproduced in Besiroglu et al. (2024) Table
Cs = [1e19, 1e21, 5.76e23, 1e25]
opt_h = [dict(C=C, N=optimal(H['E'], H['A'], H['B'], H['alpha'], H['beta'], C)[0], D=optimal(H['E'], H['A'], H['B'], H['alpha'], H['beta'], C)[1]) for C in Cs]
for o in opt_h: o['tokens_per_param'] = o['D'] / o['N']
EP = dict(E=1.8172, A=482.01, B=2085.43, alpha=.3478, beta=.3658)    # Epoch AI replication (Besiroglu et al. 2024)
opt_e = [dict(C=C, N=optimal(EP['E'], EP['A'], EP['B'], EP['alpha'], EP['beta'], C)[0], D=optimal(EP['E'], EP['A'], EP['B'], EP['alpha'], EP['beta'], C)[1]) for C in Cs]
for o in opt_e: o['tokens_per_param'] = o['D'] / o['N']
Ctoy = [1e10, 1e11, 1e12, 1e13]
opt_t = [dict(C=C, N=optimal(Efit, Afit, Bfit, al, be, C)[0], D=optimal(Efit, Afit, Bfit, al, be, C)[1]) for C in Ctoy]
for o in opt_t: o['tokens_per_param'] = o['D'] / o['N']
# numeric check of the closed form: brute-force minimum on a grid of N for one budget
def loss_hoff(N, D): return H['E'] + H['A'] * N ** -H['alpha'] + H['B'] * D ** -H['beta']
C0 = 5.76e23; grid = np.logspace(9, 12.5, 4000); Lg = loss_hoff(grid, C0 / (6 * grid)); Nbrute = float(grid[Lg.argmin()])
frontier = dict(C=list(np.logspace(19, 26, 29)), N=[], D=[], ratio=[])
for C in frontier['C']: n, d = optimal(H['E'], H['A'], H['B'], H['alpha'], H['beta'], C); frontier['N'].append(n); frontier['D'].append(d); frontier['ratio'].append(d / n)
frontier['ratio_epoch'] = [optimal(EP['E'], EP['A'], EP['B'], EP['alpha'], EP['beta'], C)[1] / optimal(EP['E'], EP['A'], EP['B'], EP['alpha'], EP['beta'], C)[0] for C in frontier['C']]
frontier['C'] = [float(v) for v in frontier['C']]
# power-law slope in N at the largest D (log-log)
sl = float(np.polyfit(np.log(Ns), np.log(np.maximum(L[:, -1] - Efit, 1e-6)), 1)[0])
out = dict(V=V, K=K, configs=CONFIGS, N=Ns.tolist(), D=Ds.tolist(), L=L.tolist(), batch=BATCH,
           fit=dict(E=Efit, A=Afit, B=Bfit, alpha=float(al), beta=float(be), rmse=rmse), pred=(Efit + Afit * NN ** -al + Bfit * DD ** -be).tolist(),
           hoffmann=H, optimal_hoffmann=opt_h, epoch=EP, optimal_epoch=opt_e, brute_N=Nbrute, optimal_toy=opt_t, frontier=frontier, slope_N=float(sl),
           ratio_exponent=float((H['alpha'] - H['beta']) / (H['alpha'] + H['beta'])))
json.dump(out, open('results.json', 'w'), indent=1); print('fit', out['fit']); print('hoffmann opt', opt_h); print('toy opt', opt_t); print('brute N', Nbrute)
