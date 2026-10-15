"""Relations and outer-product memory. Pure NumPy. Run: python3 lab.py -> results.json
A. an outer-product memory W = sum_i v_i k_i^T: exact read-out y = sum_i v_i (k_i . k_j); measured crosstalk against the closed form sqrt((n-1)/d)
B. a toy world of 8 entities with three binary attributes (gender, age, royalty): embeddings trained to predict attribute words; analogies a - b + c
C. a small MLP that must apply a relation (flip gender / age / royalty) to an entity: held-out relation pairs, and linear probes of every layer"""
import json, itertools, numpy as np
R = {}
rng = np.random.default_rng(0)
# ---------------------------------------------------------------- A. outer-product memory
k1 = np.array([1., 0., 0.]); k2 = np.array([.6, .8, 0.]); v1 = np.array([1., 0.]); v2 = np.array([0., 1.]); W = np.outer(v1, k1) + np.outer(v2, k2)
R['example'] = dict(read1=(W @ k1).tolist(), read2=(W @ k2).tolist(), dot=float(k1 @ k2), W=W.tolist())
d = 64; rows = []
for n in (8, 16, 32, 64, 96, 128):
    er, eo = [], []
    for s in range(20):
        r = np.random.default_rng(100 + s); K = r.normal(size=(n, d)); K /= np.linalg.norm(K, axis=1, keepdims=True); V = r.normal(size=(n, d)); V /= np.linalg.norm(V, axis=1, keepdims=True)
        Wm = V.T @ K; Y = K @ Wm.T; er.append(float(np.sqrt(np.mean(np.linalg.norm(Y - V, axis=1) ** 2))))
        if n <= d: Q, _ = np.linalg.qr(r.normal(size=(d, d))); Ko = Q[:, :n].T; eo.append(float(np.mean(np.linalg.norm(Ko @ (V.T @ Ko).T - V, axis=1))))
    rows.append(dict(n=n, measured=float(np.mean(er)), theory=float(np.sqrt((n - 1) / d)), orthonormal=float(np.mean(eo)) if eo else None))
R['memory'] = dict(d=d, rows=rows)
# ---------------------------------------------------------------- B. embeddings with attribute structure
names = {(0, 0, 0): 'man', (1, 0, 0): 'woman', (0, 1, 0): 'boy', (1, 1, 0): 'girl', (0, 0, 1): 'king', (1, 0, 1): 'queen', (0, 1, 1): 'prince', (1, 1, 1): 'princess'}   # (female, child, royal)
ents = list(names.keys()); words = ['he', 'she', 'works', 'pays', 'plays', 'learns', 'crown', 'throne', 'market', 'farm']
def target(a):
    t = np.zeros(len(words)); g, c, ry = a
    t[words.index('she' if g else 'he')] += 1 / 3
    for w in (('plays', 'learns') if c else ('works', 'pays')): t[words.index(w)] += 1 / 6
    for w in (('crown', 'throne') if ry else ('market', 'farm')): t[words.index(w)] += 1 / 6
    return t
T = np.array([target(a) for a in ents]); dim = 4
E = rng.normal(0, .3, (8, dim)); U = rng.normal(0, .3, (dim, len(words)))
for t in range(4000):
    Z = E @ U; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); dZ = (P - T) / 8
    gE, gU = dZ @ U.T, E.T @ dZ; E -= .5 * gE; U -= .5 * gU
Z = E @ U; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); R['skipgram'] = dict(loss=float(-(T * np.log(P + 1e-12)).sum(1).mean()), floor=float(-(T * np.log(T + 1e-12)).sum(1).mean()), emb=E.tolist(), names=[names[a] for a in ents])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
idx = {names[a]: i for i, a in enumerate(ents)}
def analogy(a, b, c):
    q = E[idx[b]] - E[idx[a]] + E[idx[c]]; sc = {n: cos(q, E[i]) for n, i in idx.items() if n not in (a, b, c)}; return max(sc, key=sc.get), sc
ans, sc = analogy('man', 'king', 'woman'); R['king_queen'] = dict(answer=ans, scores=sc)
qs = []
for attr in range(3):
    for combo in itertools.product((0, 1), repeat=2):
        for combo2 in itertools.product((0, 1), repeat=2):
            if combo == combo2: continue
            def ent(cb, val): a = list(cb); a.insert(attr, val); return names[tuple(a)]
            qs.append((ent(combo, 0), ent(combo, 1), ent(combo2, 0), ent(combo2, 1)))
ok = sum(analogy(a, b, c)[0] == d_ for a, b, c, d_ in qs); R['analogies'] = dict(total=len(qs), correct=int(ok))
# ---------------------------------------------------------------- C. applying a relation, held-out pairs, probes by layer
# a larger world: 32 entities = 5 binary attributes. Embeddings from the same objective as part B, then an MLP applies "flip attribute r".
na = 5; E5s = list(itertools.product((0, 1), repeat=na)); ne = len(E5s); W5 = [(a, v) for a in range(na) for v in (0, 1)]; T5 = np.zeros((ne, len(W5)))
for i, e in enumerate(E5s):
    for a in range(na): T5[i, W5.index((a, e[a]))] = 1 / na
r5 = np.random.default_rng(0); dim5 = 6; Em = r5.normal(0, .3, (ne, dim5)); U5 = r5.normal(0, .3, (dim5, len(W5)))
for t in range(3000):
    Z = Em @ U5; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); dZ = (P - T5) / ne; gE, gU = dZ @ U5.T, Em.T @ dZ; Em -= .8 * gE; U5 -= .8 * gU
EmN = Em / np.linalg.norm(Em, axis=1, keepdims=True)
pairs = [(e, r) for e in range(ne) for r in range(na)]; flip = lambda e, r: E5s.index(tuple(v ^ (1 if j == r else 0) for j, v in enumerate(E5s[e])))
Y = np.array([flip(e, r) for e, r in pairs]); BITS = np.array([E5s[y] for y in Y], float); REL = np.eye(na)[[r for e, r in pairs]]; ONEHOT = np.eye(ne)[[e for e, r in pairs]]; EMB = EmN[[e for e, r in pairs]]
def train_mlp(Xt, Bt, seed, h=64, steps=8000, lr=.2):
    r = np.random.default_rng(seed); din = Xt.shape[1]; Wb = r.normal(0, 1 / np.sqrt(din), (din, h)); Wc = r.normal(0, 1 / np.sqrt(h), (h, h)); Wd = r.normal(0, 1 / np.sqrt(h), (h, na)); n = len(Bt)
    for t in range(steps):
        a2 = np.tanh(Xt @ Wb); a3 = np.tanh(a2 @ Wc); z = a3 @ Wd; p = 1 / (1 + np.exp(-z)); dz = (p - Bt) / n; gWd = a3.T @ dz; da3 = dz @ Wd.T * (1 - a3 ** 2); gWc = a2.T @ da3; da2 = da3 @ Wc.T * (1 - a2 ** 2); gWb = Xt.T @ da2
        Wb -= lr * gWb; Wc -= lr * gWc; Wd -= lr * gWd
    return Wb, Wc, Wd
def layers(M, Xs):
    Wb, Wc, Wd = M; a2 = np.tanh(Xs @ Wb); a3 = np.tanh(a2 @ Wc); return a2, a3, a3 @ Wd
exact = lambda M, X, B: float(np.mean(((layers(M, X)[2] > 0) == (B > .5)).all(1)))
def probe_acc(F, rng_, reps=20):
    accs = []
    for _ in range(reps):
        perm = rng_.permutation(ne); trn, tst = perm[:24], perm[24:]
        for a in range(na):
            lab = np.array([E5s[e][a] for e in range(ne)]) * 2. - 1.; w = np.linalg.lstsq(np.c_[F[trn], np.ones(24)], lab[trn], rcond=1e-6)[0]; accs.append(float(np.mean(np.sign(np.c_[F[tst], np.ones(len(tst))] @ w) == lab[tst])))
    return float(np.mean(accs))
out = {}
for cond, ENT in (('one-hot ids', ONEHOT), ('learned embeddings', EMB)):
    X = np.c_[ENT, REL]; res = []; pr = {'input': [], 'hidden 1': [], 'hidden 2': []}
    for seed in range(10):
        r = np.random.default_rng(seed); perm = r.permutation(len(pairs)); tr, te = perm[:96], perm[96:]; M = train_mlp(X[tr], BITS[tr], seed)
        res.append((exact(M, X[tr], BITS[tr]), exact(M, X[te], BITS[te])))
        Xe = np.array([X[e * na] for e in range(ne)]); a2, a3, _ = layers(M, Xe)
        for name, F in (('input', Xe[:, :ENT.shape[1]]), ('hidden 1', a2), ('hidden 2', a3)): pr[name].append(probe_acc(F, np.random.default_rng(seed)))
    out[cond] = dict(train_acc=float(np.mean([a for a, _ in res])), test_acc=float(np.mean([b for _, b in res])), test_by_seed=[b for _, b in res], probes={k: float(np.mean(v)) for k, v in pr.items()})
R['relation'] = dict(world=dict(entities=ne, attributes=na, pairs=len(pairs), n_train=96, n_test=len(pairs) - 96, chance_exact=1 / ne), conditions=out)
# directions: are the four gender differences the same vector? and a 2-D view of the 8 embeddings
def vec(n): return E[idx[n]]
g = [vec('woman') - vec('man'), vec('girl') - vec('boy'), vec('queen') - vec('king'), vec('princess') - vec('prince')]
R['directions'] = dict(gender_cos=[[cos(x, y) for y in g] for x in g], gender_norms=[float(np.linalg.norm(x)) for x in g])
Ec = E - E.mean(0); _, _, Vt = np.linalg.svd(Ec, full_matrices=False); R['pca2'] = dict(coords=(Ec @ Vt[:2].T).tolist(), names=R['skipgram']['names'], var=[float(v) for v in (np.linalg.svd(Ec, compute_uv=False) ** 2 / (np.linalg.svd(Ec, compute_uv=False) ** 2).sum())])
# E = b + g*d_gender + c*d_age + r*d_royal ? fit it, and give the 8 embeddings in an orthonormal basis of the three directions
Abits = np.array(ents, float); Dm, res, *_ = np.linalg.lstsq(np.c_[np.ones(8), Abits], E, rcond=None); recon = np.c_[np.ones(8), Abits] @ Dm
Qb, _ = np.linalg.qr(Dm[1:].T); cube = (E - E.mean(0)) @ Qb
R['additive'] = dict(resid=float(np.linalg.norm(E - recon) / np.linalg.norm(E - E.mean(0))), dir_norms=[float(np.linalg.norm(v)) for v in Dm[1:]], dir_cos=[[cos(x, y) for y in Dm[1:]] for x in Dm[1:]], cube=cube.tolist(), names=R['skipgram']['names'])
json.dump(R, open('results.json', 'w'), indent=1)
print('A example', R['example']); [print(r) for r in rows]
print('B loss', R['skipgram']['loss'], 'floor', R['skipgram']['floor'], 'king_queen', R['king_queen']['answer'], {k: round(v, 3) for k, v in R['king_queen']['scores'].items()}, 'analogies', R['analogies'])
print('C world', R['relation']['world'])
for k, v in out.items(): print('C', k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != 'test_by_seed'}, [round(x, 2) for x in v['test_by_seed']])
