# Transformer Training Dynamics

Code for the AI Notes film *Transformer Training Dynamics*: signals, gradients, updates and representations in a Transformer. Every lab number shown in the film is computed here.

```bash
python3 training-dynamics/lab.py        # NumPy only, a few seconds on a CPU; writes results/lab-results.json
pytest training-dynamics                # pins every on-screen number
```

What `lab.py` checks (fixed seed 20261006):

| Chapter | Check | Result |
|---|---|---|
| 08 Derive the attention scaling | variance of q·k at widths 16–1024, raw / ÷√d / ÷d, 20,000 samples each | raw ≈ d, ÷√d ≈ 1 (e.g. 259 → 1.01 at d = 256) |
| 12 Backpropagate through attention | hand-written Q, K, V, X gradients against central finite differences, causal mask | max error 2 × 10⁻¹⁰ |
| 07, 27 | Q/K gauge invariance; permutation equivariance with the mask permuted | 1.1 × 10⁻¹⁶, 5.6 × 10⁻¹⁷ |
| 13 Why a route gains attention | one gradient step on two logits, values 0 and 2, target 2 | attention 0.475 / 0.525, loss 0.5 → 0.451 |
| 20 Derive the idealized Muon direction | best step under a spectral-norm budget vs scaled SGD, σ(G) = (100, 10, 0.1) | linear loss change −1.101 vs −1.010 |

These are numerical checks of equations, not language-model benchmarks. The induction-circuit results in chapters 23 and 25 come from the A1 code in [`advanced-transformers/`](../advanced-transformers/README.md) (`tools/run_a1.py induction <seed>`).
