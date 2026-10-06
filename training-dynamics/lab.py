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
    report['status']='PASS'
    return report

def main():
    report=run()
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/'lab-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
