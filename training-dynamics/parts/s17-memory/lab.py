"""Attention as memory: softmax (keep every pair), linear (additive state), delta (write the error), gated delta (forget).
One test for all: write n random key-value pairs into a memory of width 64, read every key back, report the relative error.
Run: python3 lab.py -> results.json"""
import json, numpy as np
def unitv(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def recurrent(Km, Vm, beta, alpha=None):
    T, dk = Km.shape; St = np.zeros((dk, Vm.shape[1]))
    for t in range(T):
        if alpha is not None: St = alpha[t] * St
        St = St - beta[t] * np.outer(Km[t], Km[t] @ St - Vm[t])
    return St
R = {}
# worked example: the same key written twice with different values (2-dimensional)
k = np.array([1., 0.]); v1 = np.array([1., 0.]); v2 = np.array([0., 1.])
S_add = np.outer(k, v1) + np.outer(k, v2); S1 = np.outer(k, v1); read_before = S1.T @ k; S_del = S1 + 1.0 * np.outer(k, v2 - read_before)
R['collision'] = dict(additive_read=(S_add.T @ k).tolist(), delta_read_before=read_before.tolist(), delta_error=(v2 - read_before).tolist(), delta_read_after=(S_del.T @ k).tolist())
# linear attention = the same function as pairwise kernel attention, computed in another order (T=12 tokens, d=8)
rng = np.random.default_rng(10); T_, d_ = 12, 8; Q = rng.normal(size=(T_, d_)); K = rng.normal(size=(T_, d_)); V = rng.normal(size=(T_, d_))
phi = lambda x: np.where(x > 0, x + 1, np.exp(x)); Pq, Pk = phi(Q), phi(K); mask = np.tril(np.ones((T_, T_), bool)); A = (Pq @ Pk.T) * mask; Yk = (A @ V) / A.sum(1, keepdims=True)
S = np.zeros((d_, d_)); z = np.zeros(d_); Yr = []
for t in range(T_): S = S + np.outer(Pk[t], V[t]); z = z + Pk[t]; Yr.append(Pq[t] @ S / (Pq[t] @ z))
R['linear'] = dict(T=T_, d=d_, pairwise_vs_recurrent=float(np.abs(Yk - np.array(Yr)).max()), state_entries=d_ * d_ + d_,
                   cost=dict(N=4096, m=64, pairwise_madds=2 * 4096 * 4096 * 64, recurrent_madds=2 * 4096 * 64 * 64, ratio=(2 * 4096 * 4096 * 64) / (2 * 4096 * 64 * 64)))
# delta transition never grows the state: ||I - beta k k^T||_2 <= 1 for unit k and beta in [0, 1]
rng = np.random.default_rng(3); dd = 16; Kd = unitv(rng.normal(size=(20, dd))); R['delta_norm_max'] = float(max(np.linalg.norm(np.eye(dd) - b * np.outer(kk, kk), 2) for kk in Kd for b in (0., .25, .5, .75, 1.)))
# the common test: write n pairs, read all back
dim = 64; BETA_SOFTMAX = 30.; rows = []
for n in (16, 32, 64, 96, 128):
    e = {k: [] for k in ('softmax', 'additive', 'delta', 'gated_delta')}
    for s in range(5):
        r = np.random.default_rng(s); Kn, Vn = unitv(r.normal(size=(n, dim))), r.normal(size=(n, dim)); bn = np.ones(n)
        W = np.exp(BETA_SOFTMAX * (Kn @ Kn.T)); W /= W.sum(1, keepdims=True)
        for name, Y in (('softmax', W @ Vn), ('additive', Kn @ (Kn.T @ Vn)), ('delta', Kn @ recurrent(Kn, Vn, bn)), ('gated_delta', Kn @ recurrent(Kn, Vn, bn, alpha=np.full(n, .98)))):
            e[name].append(float(np.mean(np.linalg.norm(Y - Vn, axis=1) / np.linalg.norm(Vn, axis=1))))
    rows.append(dict(pairs=n, **{k: float(np.mean(v)) for k, v in e.items()}))
R['capacity'] = dict(width=dim, beta_softmax=BETA_SOFTMAX, rows=rows)
R['state'] = dict(fixed=dim * dim, kv_cache={str(n): n * 2 * dim for n in (16, 64, 128, 4096)})
F3 = json.load(open('../../films/f3/results.json'))['delta']; R['delta_checks'] = {k: F3[k] for k in ('overwrite_error', 'gradient_step_error', 'chunk_vs_recurrent', 'two_chunk_chain')}   # computed by ../f3/lab.py
json.dump(R, open('results.json', 'w'), indent=1)
print(R['collision']); print(R['linear']['pairwise_vs_recurrent'], R['delta_norm_max']); [print(r) for r in rows]; print(R['state'])
