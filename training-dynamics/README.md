# Transformer Training Dynamics

Code behind the AI Notes series *Transformer Training Dynamics*: four films (1, 2A, 2B, 3) and a set of short standalone parts. Every number shown on screen comes from a lab script here, and the narration builder asserts every spoken number against the saved results before the voice-over is generated.

| Folder | What |
|---|---|
| `films/f1` | Film 1, *Trainable at step zero*: variance, random matrices, attention scaling, softmax and its Jacobian, depth, norms, the MLP as a memory, activations |
| `films/f2a` | Film 2A, *Gradients, module by module*: the loss, a linear layer, shared matrices, embeddings, residuals, norms, MLP, attention, routes |
| `films/f2b` | Film 2B, *Optimization tricks*: step size and the edge of stability, momentum, RMSProp, Adam/AdamW, guards, precision, schedules, batch size, Muon, Shampoo, QK-Clip |
| `films/f2` | only `results.json` of the earlier single-film version of film 2, kept because film 3's lab gathers its derivation checks |
| `films/f3` | Film 3, *Thousands of steps*: what accumulates in training, circuits, grokking, scale, linear and delta attention |
| `grokking/` | the PyTorch grokking runs (modular addition mod 97) that film 3 reads |
| `parts/` | labs of the short parts that have their own experiments: 12 induction heads (reads the A1 results), 13 relations and outer-product memory, 15 mu-P, 16 scaling laws, 17 attention as memory, 18 frontier models (`facts.json` with sources) |

Each film or part folder has `lab.py` (computes the numbers), `results.json` (its output), `make_narration.py` (checks every spoken number against `results.json`), `script.md` (the voice-over) and `youtube.md`.

```bash
python3 training-dynamics/films/f2b/lab.py            # NumPy only; writes results.json next to it
cd training-dynamics/films/f2b && python3 make_narration.py   # asserts every spoken number
sh training-dynamics/run_checks.sh                    # runs every make_narration.py
python3 training-dynamics/grokking/grokking.py        # PyTorch, a few minutes on a CPU
pytest training-dynamics/grokking                     # pins the grokking numbers
```

Notes
- The other short parts of the series (courier, initialization, attention scaling, norms, MLP, learning rate, backprop, Adam, precision, Muon, grokking) reuse the lab results of the films above.
- `parts/s12-induction/lab.py` re-reads the induction runs saved in `advanced-transformers/results/a1/`; `parts/s16-scaling` trains tiny character models on `shakespeare.txt` (public domain, ~2.5 minutes).
- `parts/s18-frontier` is a dated snapshot (October 2026): every figure is in `facts.json` with its source.
- Film 3 reuses two measured sets: the A1 circuits runs (`films/f3/a1_results.json`) and `grokking/results/grokking.json`.
