"""mu-P: why a learning rate tuned on a narrow model breaks on a wide one. Pure NumPy. Run: python3 lab.py -> results.json
A. one SGD step on a wide linear layer: output change = eta * |x|^2 * g  (exact), growing with the width
B. a one-hidden-layer tanh network f(x) = v . tanh(W x), trained with SGD on a fixed regression task, at widths 64..2048:
   standard parametrization (SP): W ~ N(0, 1/d_in), v ~ N(0, 1/n), one learning rate
   mu-P (SGD, Yang & Hu 2021, Tensor Programs V, Table 3): W ~ N(0, 1/d_in) with lr eta*n, v ~ N(0, 1/n^2) with lr eta/n
   For each width, scan the learning rate and record the final loss; the best learning rate per width shows whether it transfers."""
import json, numpy as np
R = {}
d_in = 16; WIDTHS = [64, 128, 256, 512, 1024, 2048]
# ---------------------------------------------------------------- A. exact one-step change of the output
rngA = np.random.default_rng(0); xA = np.sign(rngA.normal(size=d_in * 128))[: d_in * 128]       # input with coordinates of order one
A = []
for n in WIDTHS:
    x = np.sign(np.random.default_rng(n).normal(size=n)); g = 1.0; eta = .01
    W = np.random.default_rng(n + 1).normal(0, 1 / np.sqrt(n), (n, 1)); y0 = float(x @ W[:, 0]); Wn = W - eta * np.outer(x, [g]); dy = float(x @ Wn[:, 0]) - y0
    A.append(dict(n=n, init_std=float(np.std([float(np.sign(np.random.default_rng(s).normal(size=n)) @ np.random.default_rng(s + 99).normal(0, 1 / np.sqrt(n), n)) for s in range(200)])), delta=dy, predicted=-eta * float(x @ x) * g))
R['one_step'] = A
# ---------------------------------------------------------------- B. a tiny network, two parametrizations
rng = np.random.default_rng(1); Xtr = rng.normal(size=(256, d_in)); wstar = rng.normal(size=d_in) / np.sqrt(d_in); ytr = np.sin(Xtr @ wstar * 2.0) + .3 * np.sin(Xtr[:, 0] * 3)
Xte = rng.normal(size=(1024, d_in)); yte = np.sin(Xte @ wstar * 2.0) + .3 * np.sin(Xte[:, 0] * 3)
def run(n, eta, mode, steps=300, seed=0):
    r = np.random.default_rng(seed); W = r.normal(0, 1 / np.sqrt(d_in), (n, d_in)); v = r.normal(0, (1 / n) if mode == 'mup' else 1 / np.sqrt(n), n)
    lrW = eta * n if mode == 'mup' else eta; lrv = eta / n if mode == 'mup' else eta
    for t in range(steps):
        H = np.tanh(Xtr @ W.T); f = H @ v; e = f - ytr; loss = .5 * np.mean(e ** 2)
        if not np.isfinite(loss) or loss > 1e6: return float('nan')
        gv = H.T @ e / len(e); gH = np.outer(e, v) * (1 - H ** 2); gW = gH.T @ Xtr / len(e); W -= lrW * gW; v -= lrv * gv
    H = np.tanh(Xtr @ W.T); return float(.5 * np.mean((H @ v - ytr) ** 2))
LRS = [2.0 ** k for k in range(-8, 5)]
res = {}
for mode in ('sp', 'mup'):
    res[mode] = {}
    for n in WIDTHS:
        res[mode][n] = [float(np.mean([run(n, lr, mode, seed=s) for s in range(3)])) for lr in LRS]
        print(mode, n, ['%.3f' % v if np.isfinite(v) else 'nan' for v in res[mode][n]], flush=True)
def best(curve): c = [v if np.isfinite(v) else 1e9 for v in curve]; return LRS[int(np.argmin(c))]
R['lrs'] = LRS; R['losses'] = {m: {str(n): res[m][n] for n in WIDTHS} for m in res}; R['best_lr'] = {m: {str(n): best(res[m][n]) for n in WIDTHS} for m in res}; R['best_loss'] = {m: {str(n): float(np.nanmin(res[m][n])) for n in WIDTHS} for m in res}
R['widths'] = WIDTHS; R['d_in'] = d_in
json.dump(R, open('results.json', 'w'), indent=1); print('best', R['best_lr']); print('A', A)
