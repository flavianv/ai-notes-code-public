"""Transformer Training Dynamics (AI Notes): every lab number shown in the film. NumPy only, runs on a CPU.

These are simulations and equation checks, not trained-model benchmarks.
Run from any directory: python3 training-dynamics/lab.py   (writes results/lab-results.json)
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261006)

def softmax(z):
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(x, wq, wk, wv, mask):
    q, k, v = x @ wq, x @ wk, x @ wv
    a = softmax(np.where(mask, q @ k.T / np.sqrt(q.shape[1]), -np.inf))
    return a @ v, (q, k, v, a)

def backward(x, wq, wk, wv, mask, upstream):
    _, (q, k, v, a) = attention(x, wq, wk, wv, mask)
    gv = a.T @ upstream
    ga = upstream @ v.T
    gs = a * (ga - (a * ga).sum(axis=1, keepdims=True))
    gq = gs @ k / np.sqrt(q.shape[1])
    gk = gs.T @ q / np.sqrt(q.shape[1])
    gx = gq @ wq.T + gk @ wk.T + gv @ wv.T
    return gx, x.T @ gq, x.T @ gk, x.T @ gv, gs

def differences(fn, a, eps=1e-6):
    out = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old = a[idx]
        a[idx] = old + eps
        plus = fn()
        a[idx] = old - eps
        minus = fn()
        a[idx] = old
        out[idx] = (plus - minus) / (2 * eps)
    return out

def run():
    report = {'seed':20261006,'kind':'numerical equation checks, no trained model','variance':[]}
    for d in [16, 64, 256, 1024]:
        # Batch sampling limits memory while retaining independent observations.
        samples=[]
        for _ in range(20):
            q,k = RNG.normal(size=(2, 1000, d))
            samples.append(np.sum(q*k,axis=1))
        z=np.concatenate(samples)
        report['variance'].append(dict(d=d,raw=float(z.var()),scaled=float((z/np.sqrt(d)).var()),divided_by_d=float((z/d).var())))
        assert abs(z.var()/d-1)<.06
    x=RNG.normal(size=(4,5)); wq=RNG.normal(size=(5,3))*.3
    wk=RNG.normal(size=(5,3))*.3; wv=RNG.normal(size=(5,2))*.3
    mask=np.tril(np.ones((4,4),dtype=bool)); upstream=RNG.normal(size=(4,2))
    fn=lambda:float(np.sum(attention(x,wq,wk,wv,mask)[0]*upstream))
    grads=backward(x,wq,wk,wv,mask,upstream)
    report['gradient_max_abs_error']={}
    for name,param,grad in zip(['X','WQ','WK','WV'],[x,wq,wk,wv],grads[:4]):
        num=differences(fn,param)
        err=float(np.max(np.abs(num-grad)))
        report['gradient_max_abs_error'][name]=err
        np.testing.assert_allclose(num,grad,atol=2e-8,rtol=2e-6)
    assert np.all(grads[4][~mask]==0)
    np.testing.assert_allclose(grads[4].sum(axis=1),0,atol=1e-14)
    # Fixed causal mask is not arbitrarily permutation equivariant. Permute
    # the mask as well to check the algebra without changing allowed edges.
    p=np.array([2,0,3,1]); h=attention(x,wq,wk,wv,mask)[0]
    hp=attention(x[p],wq,wk,wv,mask[np.ix_(p,p)])[0]
    report['permutation_max_abs_error']=float(np.max(np.abs(hp-h[p])))
    np.testing.assert_allclose(hp,h[p],atol=1e-14)
    # Gauge transformation on an unrotated Q/K head.
    r=RNG.normal(size=(3,3))+3*np.eye(3)
    h2=attention(x,wq@r,wk@np.linalg.inv(r).T,wv,mask)[0]
    report['gauge_max_abs_error']=float(np.max(np.abs(h2-h)))
    np.testing.assert_allclose(h2,h,atol=1e-14)
    # Local routing example, a real gradient step on two logits.
    v=np.array([0.,2.]); z=np.zeros(2); y=2.
    a=softmax(z); h=float(a@v); u=h-y
    gs=a*u*(v-h); before=.5*(h-y)**2
    a_new=softmax(z-.1*gs); after=.5*(float(a_new@v)-y)**2
    np.testing.assert_allclose(gs,[.5,-.5]); assert after<before
    report['routing']=dict(gradient=gs.tolist(),attention_before=a.tolist(),attention_after=a_new.tolist(),loss_before=before,loss_after=after)
    # Spectral budget: same largest singular value for both candidates.
    g=np.diag([100.,10.,.1]); u,s,vt=np.linalg.svd(g,full_matrices=False)
    eta=.01; polar=-eta*u@vt; spectral_sgd=-eta*g/s[0]
    assert np.isclose(np.linalg.norm(polar,2),eta)
    assert np.isclose(np.sum(g*polar),-eta*s.sum())
    report['spectral_budget']=dict(eta=eta,gradient_singular_values=s.tolist(),sgd_singular_values=np.linalg.svd(spectral_sgd,compute_uv=False).tolist(),ideal_muon_singular_values=np.linalg.svd(polar,compute_uv=False).tolist(),linear_change_sgd=float(np.sum(g*spectral_sgd)),linear_change_ideal_muon=float(np.sum(g*polar)))
    report['extras']=extras()
    report['status']='PASS'
    return report

def extras():
    """Checks added for the three-film version. A separate generator, so the original numbers above are unchanged.
    On macOS with numpy 2 + Accelerate, matmul can print spurious divide/overflow warnings; every value here is finite (the tests check it)."""
    r = np.random.default_rng(20261007); out = {}
    z = r.choice([-1., 1.], 100_000); out['moments'] = dict(mean=float(z.mean()), var=float(z.var()), var_3z=float((3*z).var()))
    S = r.choice([-1., 1.], (20_000, 100)).sum(1)
    rho = .1; c = r.normal(size=(20_000, 1)); e = r.normal(size=(20_000, 100)); Sc = (np.sqrt(rho)*c + np.sqrt(1-rho)*e).sum(1)
    out['sums'] = dict(std_independent=float(S.std()), var_rho_0_1=float(Sc.var()), predicted_rho_0_1=100*(1+99*rho))
    d = 512; x = r.normal(size=d); ys = np.array([x @ (r.normal(size=(d, 4))/np.sqrt(d)) for _ in range(2000)]).ravel(); yu = np.array([x @ r.normal(size=(d, 4)) for _ in range(2000)]).ravel()
    out['linear'] = dict(d=d, var_scaled=float(ys.var()), var_unscaled=float(yu.var()), sq_norm_x=float(x @ x))
    di, do = 512, 2048; W = r.normal(size=(di, do))/np.sqrt(di); X = r.normal(size=(4000, di)); Gy = r.normal(size=(4000, do))
    out['init'] = dict(d_in=di, d_out=do, forward_var=float((X @ W).var()), backward_var=float((Gy @ W.T).var()))
    zz = np.array([1., 2.2, .4, 1.5]); p = softmax(zz); ent = lambda q: float(-(q*np.log(q)).sum())
    out['softmax'] = dict(sens_half=.25, sens_0999=.999*(1-.999), shift_diff=float(np.abs(softmax(zz+5)-p).max()), entropy=ent(p), entropy_3z=ent(softmax(3*zz)), weights=p.tolist(), weights_3z=softmax(3*zz).tolist())
    res = {}
    for L in [12, 24, 48]:
        for name, a in [('alpha_1', 1.), ('alpha_inv_sqrt_L', 1/np.sqrt(L))]:
            xs = r.normal(size=20_000) + a*r.normal(size=(L, 20_000)).sum(0); res[f'{L}:{name}'] = float(xs.var())
    out['residual'] = res
    lg = np.array([2., 1., 0., -1.]); pl = softmax(lg); out['objective'] = dict(logits=lg.tolist(), target=0, probs=pl.tolist(), loss=float(-np.log(pl[0])), grad=(pl - np.eye(4)[0]).tolist())
    Xs = r.normal(size=(6, 5)); Gq = r.normal(size=(6, 3)); out['shared_grad_error'] = float(np.abs(Xs.T @ Gq - sum(np.outer(Xs[i], Gq[i]) for i in range(6))).max())
    dm = 768; W1 = r.normal(size=(dm, 4*dm))/np.sqrt(dm); W2 = r.normal(size=(4*dm, dm))/np.sqrt(4*dm); xm = r.normal(size=(64, dm))
    pre = xm @ W1; h = np.maximum(pre, 0)
    xi = xm[0]; kv = sum(max(xi @ W1[:, i], 0.) * W2[i] for i in range(4*dm)); u = r.normal(size=dm); gv = np.outer(h[0], u); active = pre[0] > 0
    out['mlp'] = dict(d=dm, attn_weights=4*dm*dm, mlp_weights=8*dm*dm, mlp_share=8/12, kv_error=float(np.abs(kv - h[0] @ W2).max()), active_fraction=float((pre > 0).mean()), dead_value_grad_max=float(np.abs(gv[~active]).max()))
    from math import erf, exp, pi, sqrt
    gelu_p = lambda t: .5*(1+erf(t/sqrt(2))) + t*exp(-t*t/2)/sqrt(2*pi); sig = lambda t: 1/(1+exp(-t))
    hid = 8*dm//3; out['activations'] = dict(relu_prime_m1=0., gelu_prime_m1=gelu_p(-1.), silu_prime_m1=sig(-1.)*(1+(-1.)*(1-sig(-1.))), swiglu_hidden=hid, swiglu_weights=3*dm*hid, classic_weights=8*dm*dm)
    q = r.normal(size=(256, 64)); k = r.normal(size=(256, 64)); g = 10.; raw = lambda a, b: np.abs(a @ b.T).max(); nrm = lambda a: a/np.linalg.norm(a, axis=1, keepdims=True)
    out['qk_norm'] = dict(raw_max=float(raw(q, k)), raw_max_grown=float(raw(10*q, 10*k)), qknorm_max=float(g*raw(nrm(q), nrm(k))), qknorm_max_grown=float(g*raw(nrm(10*q), nrm(10*k))), g=g)
    out['rmsprop_first_step'] = dict(beta2=.999, rmsprop=float(100/np.sqrt(.001*100**2)), adam_corrected=float(100/np.sqrt(.001*100**2/(1-.999))))
    A = r.normal(size=(50, 200)); b = r.normal(size=50); w = np.zeros(200); lr = 1/np.linalg.norm(A, 2)**2
    for _ in range(20_000): w -= lr * A.T @ (A @ w - b)
    wp = np.linalg.pinv(A) @ b; null = np.linalg.svd(A)[2][-1]
    out['implicit_bias'] = dict(rows=50, cols=200, residual=float(np.linalg.norm(A @ w - b)), dist_to_pinv=float(np.linalg.norm(w - wp)), norm_gd=float(np.linalg.norm(w)), norm_other_fit=float(np.linalg.norm(wp + 2*null)))
    dr = {}
    for N in [16, 32, 64, 128]:
        K = r.normal(size=(N, 64)); K /= np.linalg.norm(K, axis=1, keepdims=True); Vv = r.normal(size=(N, 64))
        Sa = K.T @ Vv; Sd = np.zeros((64, 64))
        for kk, vv in zip(K, Vv): Sd = Sd - np.outer(kk, kk @ Sd - vv)
        err = lambda Sm: float(np.mean(np.linalg.norm(K @ Sm - Vv, axis=1) / np.linalg.norm(Vv, axis=1)))
        dr[str(N)] = dict(additive=err(Sa), delta=err(Sd))
    out['delta_rule'] = dr
    tau = 100.; Wq = r.normal(size=(4, 64, 16)) * 3; Wk = r.normal(size=(4, 64, 16)) * 3; Xh = r.normal(size=(32, 64))
    smax = np.array([np.abs((Xh @ Wq[h]) @ (Xh @ Wk[h]).T).max() for h in range(4)]); gam = np.minimum(1, tau/smax)
    after = np.array([np.abs((Xh @ (Wq[h]*np.sqrt(gam[h]))) @ (Xh @ (Wk[h]*np.sqrt(gam[h]))).T).max() for h in range(4)])
    out['qk_clip'] = dict(tau=tau, s_max_before=smax.tolist(), s_max_after=after.tolist())
    pis = np.array([.6, .25, .15]); Gs = np.array([[1., -.2], [-.3, -1.], [.4, .9]]); idx = r.choice(3, 100_000, p=pis)
    out['mixture'] = dict(pi=pis.tolist(), expected=(pis @ Gs).tolist(), sampled=Gs[idx].mean(0).tolist())
    # noisy quadratic: constant step vs warmup-stable-decay; then steps to a target loss vs batch size
    def sgd(etas, B, sigma=1., h=1., seed=0):
        rr = np.random.default_rng(seed); w = 3.; L = []
        for eta in etas: w -= eta * (h*w + sigma*rr.normal()/np.sqrt(B)); L.append(.5*h*w*w)
        return np.array(L)
    T = 4000; const = np.full(T, .2); wsd = np.concatenate([np.full(int(.8*T), .2), np.linspace(.2, .002, T - int(.8*T))])
    lc = np.mean([sgd(const, 1, seed=s)[-200:].mean() for s in range(20)]); lw = np.mean([sgd(wsd, 1, seed=s)[-200:].mean() for s in range(20)])
    steps = {}
    for B in [1, 4, 16, 64, 256, 1024]:
        st = []
        for s in range(20):
            Ls = sgd(np.full(3000, .2), B, seed=s); win = np.convolve(Ls, np.ones(20)/20, 'valid'); hit = np.nonzero(win < .005)[0]; st.append(int(hit[0]) if len(hit) else 3000)
        steps[str(B)] = float(np.median(st))
    out['schedules'] = dict(final_loss_constant=float(lc), final_loss_wsd=float(lw), steps_to_target_by_batch=steps)
    return out

def main():
    report=run()
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'lab-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
