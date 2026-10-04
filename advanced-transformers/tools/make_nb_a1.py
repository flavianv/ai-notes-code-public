# Builds advanced-transformers/notebooks/A1_circuits.ipynb (A1: induction heads, ablations, superposition, sparse autoencoder)
import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Understanding Transformers · A1: circuits
Companion notebook to the video. We train a tiny attention-only transformer, watch an induction circuit form, find its heads, switch them off, and then look at superposition and a sparse autoencoder. Everything runs on a laptop CPU in a few minutes.
""")
code("""
import sys, subprocess, os, pathlib
if "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "-q", "https://github.com/flavianv/ai-notes-code-public"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "ai-notes-code"], check=True)
    sys.path.insert(0, "ai-notes-code/advanced-transformers")
else:
    sys.path.insert(0, str(pathlib.Path.cwd().parent))                  # run from advanced-transformers/notebooks
import torch
from atlib.circuits import *
from atlib.automata import TRANSITIONS
torch.manual_seed(0)
STEPS = int(os.environ.get("STEPS", 1500))                               # the video used 3000; the circuit forms around step 300-600
""")
md("## 1. One attention read\nScores `[0, 6, 0, 0]` after softmax: one large score and the head reads a single position.")
code("""
print([round(v, 4) for v in torch.tensor([0., 6., 0., 0.]).softmax(-1).tolist()])
""")
md("""## 2. The task: random tokens, then the same tokens again
The first half is unpredictable. The repeat is predictable only by finding the earlier copy. Training varies the repeat length (8 to 32), so a head that just looks a fixed distance back cannot solve it.""")
code("""
print(repeat_batch(2, half=8, gen=torch.Generator().manual_seed(0)))
""")
md("## 3. One layer vs two layers\nWe log the loss on the random half and on the repeat during training. Watch the repeat loss for the two-layer model: a plateau, then a sudden drop.")
code("""
logs = {}
models = {}
for L in (1, 2):
    logs[L] = []
    models[L] = train_induction(n_layers=L, steps=STEPS, seed=0, log=logs[L])
    print(f"{L} layer(s): loss random half / repeat = {[round(v, 2) for v in halves_loss(models[L])]}, repeat accuracy = {repeat_accuracy(models[L]):.3f}")
print("\\ntwo layers, repeat loss over training:")
for step, _, rep in logs[2][::4]: print(f"  step {step:5d}  {rep:.2f}  " + "#" * int(rep * 10))
""")
md("## 4. Find the heads\nPrevious-token score: how much a layer-1 head attends to the position just before. Induction score: how much a layer-2 head attends to the token right after the earlier copy.")
code("""
m = models[2]
prev, ind = head_scores(m)
print("layer 1, previous-token score per head:", {f"{l}.{h}": round(v, 2) for (l, h), v in prev.items() if l == 0})
print("layer 2, induction score per head:     ", {f"{l}.{h}": round(v, 2) for (l, h), v in ind.items() if l == 1})
""")
md("## 5. Switch heads off\nOne head at a time can hide redundancy, so we also switch off whole groups.")
code("""
best_ind = max((k for k in ind if k[0] == 1), key=ind.get)
print("one induction head off:", round(repeat_accuracy(m, off={best_ind}), 3))
for k, v in group_ablation(m).items(): print(f"{k:16s}", v if not isinstance(v, float) else round(v, 3))
""")
md("## 6. Longer repeats than in training\nTraining used repeats of at most 32 tokens.")
code("""
for h in (8, 20, 32, 50): print(f"repeat length {h:2d}: accuracy {repeat_accuracy(m, half=h):.3f}")
""")
md("## 7. An MLP as pair detectors (constructed, not learned)\nNine ReLU units compute the 3-state machine's transition exactly.")
code("""
f = pair_detector_mlp(TRANSITIONS); oh = lambda i: torch.eye(3)[i]
for s in range(3):
    print(f"state {s}:", {a: int(f(oh(s), oh(ai)).argmax()) for ai, a in enumerate("ABI")}, " expected", {a: TRANSITIONS[a][s] for a in "ABI"})
""")
md("## 8. Superposition: 5 features through 2 numbers\nThe rarer the features, the more of them the model keeps.")
code("""
for S in (0.0, 0.5, 0.9):
    W, b = train_toy(sparsity=S)
    print(f"sparsity {S}: features represented = {represented(W)}, column lengths = {[round(v, 2) for v in W.norm(dim=0).tolist()]}")
""")
md("## 9. A sparse autoencoder recovers the hidden directions\nIt only sees the 2-number activations. For each true feature: the best cosine similarity with a learned dictionary direction.")
code("""
W, _ = train_toy(sparsity=0.9)
D = train_sae(W, sparsity=0.9)
print([round(v, 3) for v in recovery(W, D).tolist()])
""")
nb["cells"] = C
out = pathlib.Path(__file__).parent.parent / "notebooks" / "A1_circuits.ipynb"
out.parent.mkdir(exist_ok=True); nbf.write(nb, out); print("wrote", out)
