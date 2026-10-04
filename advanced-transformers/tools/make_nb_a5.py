# Builds advanced-transformers/notebooks/A5_universality_and_scaling.ipynb (A5: a tape as two stacks, a checked learned controller, an end-to-end contrast)
import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Understanding Transformers · A5: universality and scaling
Companion notebook to the video: a tape held as two stacks, a Turing machine that adds one, a learned controller checked on every possible input, and a transformer trained end to end for contrast.
""")
code("""
import sys, subprocess, os, pathlib, random
if "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "-q", "https://github.com/flavianv/ai-notes-code-public"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "ai-notes-code"], check=True)
    sys.path.insert(0, "ai-notes-code/advanced-transformers")
else:
    sys.path.insert(0, str(pathlib.Path.cwd().parent))
import torch
from atlib.tape import *
STEPS = int(os.environ.get("STEPS", 2000))                               # the video used 3000 steps and 3 runs
""")
md("## 1. A tape as two stacks\nLeft stack: cells left of the head, nearest on top. Right stack: the current cell and everything to its right, current on top.")
code("""
t = TwoStacks("abcde"); t.move("R"); t.move("R")
print("current:", t.read(), " left top:", t.left_top(), " L:", t.L, " R:", t.R)
t.write("X"); print("after writing X:", t.tape())
""")
md("## 2. The increment machine: six rules")
code("""
for k, v in INCREMENT.items(): print(k, "->", v)
print(run_tm("1011", table_controller))
rng = random.Random(0); s = "1" + "".join(rng.choice("01") for _ in range(999))
print("1,000 digits correct:", run_tm(s, table_controller)[0] == increment(s))
""")
md("## 3. A learned controller, checked on every input it can ever see")
code("""
ctrl, seen = train_controller()
print("inputs seen in training:", seen, "  exhaustive check:", exhaustive_check(ctrl))
print("1,000 digits correct:", run_tm(s, ctrl)[0] == increment(s))
""")
md("## 4. End to end: a transformer that writes the answer directly\nTrained on numbers up to 16 digits.")
code("""
m = train_e2e(steps=STEPS)
print({n: e2e_accuracy(m, n, size=200) for n in (8, 16, 24, 32, 64)})
""")
md("Same task. The controller only makes a local decision we can check completely; end to end, the network must do the bookkeeping for the whole computation.")
nb["cells"] = C
out = pathlib.Path(__file__).parent.parent / "notebooks" / "A5_universality_and_scaling.ipynb"
out.parent.mkdir(exist_ok=True); nbf.write(nb, out); print("wrote", out)
