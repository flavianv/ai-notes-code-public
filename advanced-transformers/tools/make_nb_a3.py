# Builds advanced-transformers/notebooks/A3_what_training_learns.ipynb (A3: four ways to train on the 3-state machine, tested past the training length)
import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Understanding Transformers · A3: what does training learn?
Companion notebook to the video. Same machine, same model size, same training lengths (1 to 20 actions); only what the model is asked to write changes. Then we test far past the training length, and look at where the traces break.

Training here is shortened (`STEPS`, default 1500; the video used 3000 steps and 3 seeds per setup). Expect the same pattern, with somewhat lower numbers.
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
from atlib.automata import *
torch.manual_seed(0)
STEPS = int(os.environ.get("STEPS", 1500))
""")
md("## 1. Four formats for the same sequence\nThe tokens the model must write are the states (token ids 4, 5, 6 = states 0, 1, 2).")
code("""
for fmt in ("direct", "after", "interleaved", "looped"):
    toks, mask = encode("ABAIB", fmt)
    print(f"{fmt:12s}", toks)
""")
md("## 2. Train each setup on lengths 1 to 20, test up to 200\n`final` = final state right; `trace` = every state right; `first_error_median` = where failed traces first go wrong.")
code("""
results = {}
for fmt in ("direct", "after", "interleaved", "looped"):
    m = train(fmt, steps=STEPS, seed=0)
    results[fmt] = {n: evaluate(m, fmt, n, size=200) for n in (10, 20, 50, 100)}
    print(fmt, {n: round(r["final"], 2) for n, r in results[fmt].items()})
""")
code("""
for fmt in ("after", "interleaved"):
    print(fmt, {n: (r["trace"], r["first_error_median"]) for n, r in results[fmt].items()})
""")
md("""## 3. Why does the interleaved model break at a fixed step?
Three suspects, one variant each: random start positions (`offset`), no position encoding (`nope`), attention limited to the last 8 tokens (`window`).""")
code("""
for fmt in ("interleaved-offset", "interleaved-nope", "interleaved-window"):
    m = train(fmt, steps=STEPS, seed=0)
    print(fmt, {n: round(evaluate(m, fmt, n, size=200)["final"], 2) for n in (20, 50, 100, 200)})
""")
md("With the window, the model never sees more context than in training, and the same learned step runs as far as you like.")
nb["cells"] = C
out = pathlib.Path(__file__).parent.parent / "notebooks" / "A3_what_training_learns.ipynb"
out.parent.mkdir(exist_ok=True); nbf.write(nb, out); print("wrote", out)
