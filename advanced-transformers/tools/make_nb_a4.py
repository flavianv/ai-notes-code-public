# Builds advanced-transformers/notebooks/A4_computation_as_a_system.ipynb (A4: the same trained model inside five systems)
import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Understanding Transformers · A4: computation as a system
Companion notebook to the video. We train the interleaved model from A3 once, then never change it: we only change the system around it.
Sizes are reduced so it runs on a laptop CPU in a few minutes (the video used 3000 training steps, 200 problems and 64 samples, 3 runs).
""")
code("""
import sys, subprocess, os, pathlib
if "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "-q", "https://github.com/flavianv/ai-notes-code-public"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "ai-notes-code"], check=True)
    sys.path.insert(0, "ai-notes-code/advanced-transformers")
else:
    sys.path.insert(0, str(pathlib.Path.cwd().parent))
import torch
from atlib.automata import train, evaluate
from atlib.systems import sampling, search, harness, workers, reliability
torch.manual_seed(0)
STEPS = int(os.environ.get("STEPS", 2000))
model = train("interleaved", steps=STEPS, seed=0)
print({n: round(evaluate(model, "interleaved", n, size=200)["final"], 2) for n in (20, 50, 100)})
""")
md("## 1. Sampling, then selecting\nAt length 50, 16 samples per problem: coverage vs what each selector actually picks.")
code("""
s = sampling(model, 50, N=16, size=100)
print("single attempt:", s["single"])
for r in s["N"]: print(r)
""")
md("## 2. Search with an exact checker\nThe checker always wins; the count shows how much work it did.")
code("print(search(model, 200, size=100))")
md("## 3. A memory harness: the model only ever sees 3 to 5 tokens")
code("print({n: harness(model, n, size=100)['final'] for n in (200, 1000)})")
md("## 4. Workers that pass whole tables, vs final-state-only workers, vs one model chunk after chunk")
code("print(workers(model, size=100))")
md("## 5. Reliability inside the training range")
code("print(reliability(model, size=500))")
nb["cells"] = C
out = pathlib.Path(__file__).parent.parent / "notebooks" / "A4_computation_as_a_system.ipynb"
out.parent.mkdir(exist_ok=True); nbf.write(nb, out); print("wrote", out)
