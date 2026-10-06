# Transformer Training Dynamics

Code for the AI Notes film *Transformer Training Dynamics*: signals, gradients, updates and representations in a Transformer. Every lab number shown in the film is computed here.

```bash
python3 training-dynamics/lab.py        # NumPy only, a few seconds on a CPU; writes results/lab-results.json
python3 training-dynamics/grokking.py   # PyTorch, a few minutes on a CPU; writes results/grokking.json
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

The lab also checks the newer chapters: residual variance with depth, the MLP as key–value units (and dead ReLU gradients), GELU/SiLU derivatives, QK-norm and QK-Clip bounds, RMSProp's first step, schedules and batch size on a noisy quadratic, implicit bias of gradient descent, and the delta rule's recall.

`grokking.py` (film 3): a + b mod 97, a small MLP, AdamW, seeds 0–2. With weight decay 1.0, train accuracy passes 99% at step 100 and test accuracy at steps 11,600, 10,600 and 12,000; without weight decay test accuracy ends at 0; with shuffled labels the network still fits the training set and the test stays at chance (≈1%).

These are numerical checks of equations, not language-model benchmarks. The induction-circuit results in chapters 23 and 25 come from the A1 code in [`advanced-transformers/`](../advanced-transformers/README.md) (`tools/run_a1.py induction <seed>`).
