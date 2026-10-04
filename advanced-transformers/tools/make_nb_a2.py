# Builds advanced-transformers/notebooks/A2_what_can_it_compute.ipynb (A2: finite-state machines, the tree trick, depth vs length, counting vs state, probes)
import nbformat as nbf, pathlib
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Understanding Transformers · A2: what can one forward pass compute?
Companion notebook to the video: finite-state machines, the tree trick (adding numbers, then combining tables), and small experiments on depth, counting vs state tracking, and probes. Runs on a laptop CPU.
""")
code("""
import sys, subprocess, os, pathlib, random
if "google.colab" in sys.modules:
    subprocess.run(["git", "clone", "-q", "https://github.com/flavianv/ai-notes-code-public"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "ai-notes-code"], check=True)
    sys.path.insert(0, "ai-notes-code/advanced-transformers")
else:
    sys.path.insert(0, str(pathlib.Path.cwd().parent))                  # run from advanced-transformers/notebooks
import torch
from atlib.automata import *
from atlib.expressivity import *
torch.manual_seed(0)
STEPS = int(os.environ.get("STEPS", 800))                                # the video used 2000-3000 steps per model
""")
md("## 1. Two everyday finite-state machines")
code("""
print(turnstile(["coin", "push", "push"]), turnstile(["push", "coin", "coin"]))
line = 'say("hi") # "x"'
print(line); print("".join("i" if x else "o" for x in in_string(line)))
""")
md("## 2. Our machine: A rotates, B swaps 0 and 1, I does nothing\nOrder matters, and inserting I never changes the answer.")
code("""
print(run("ABAIB"))
print("AB ->", run("AB")[0], "  BA ->", run("BA")[0])
print("ABAIB with an extra I ->", run("ABAIIB")[0])
""")
md("## 3. The tree trick on numbers: 8 numbers, 3 rounds")
code("""
for r in tree_rounds([3, 1, 4, 1, 5, 9, 2, 6], lambda a, b: a + b): print(r)
print("rounds for 200 numbers:", len(tree_rounds(list(range(200)), lambda a, b: a + b)) - 1)
""")
md("## 4. The same trick on tables\nEach action is a table: where 0, 1 and 2 go. Two tables combine into one.")
code("""
print("A =", TRANSITIONS["A"], " B =", TRANSITIONS["B"], " A then B =", compose(TRANSITIONS["B"], TRANSITIONS["A"]))
for r in table_rounds("ABAIBAAB"): print(r)
print("start 0 ends in", table_rounds("ABAIBAAB")[-1][0][0], "; step by step:", run("ABAIBAAB")[0])
rng = random.Random(0)
assert all(summarize(s) == tuple(run(s, start=q)[0] for q in range(3)) for s in (random_actions(rng.randint(0, 300), rng) for _ in range(500)))
print("tree and step-by-step agree on 500 random sequences, from every start state")
""")
md("## 5. Depth vs length, one forward pass\nTrain at one fixed length with the state labelled at every position. At length 8 depth helps; try 32 and see training struggle.")
code("""
for n in (8, 32):
    for L in (1, 2):
        m = train_labelled("state", n_layers=L, lengths=(n, n), steps=STEPS)
        print(f"length {n:3d}, {L} layer(s): {labelled_accuracy(m, 'state', n)}")
""")
md("## 6. Counting vs state tracking\nSame 4-layer model, trained on lengths 1 to 20, tested longer.")
code("""
models = {}
for task in ("count", "state"):
    models[task] = train_labelled(task, n_layers=4, lengths=(1, 20), steps=STEPS)
    print(task, {n: round(labelled_accuracy(models[task], task, n)["last"], 2) for n in (10, 20, 50, 200)})
""")
md("## 7. Probes: does the state model carry the whole table?\nA linear probe per layer, for the whole table (6 classes, chance 17%) and for the state (3 classes, chance 33%).")
code("""
for k, row in enumerate(probe(models["state"])): print("embedding" if k == 0 else f"after layer {k}", {t: round(v, 2) for t, v in row.items()})
""")
nb["cells"] = C
out = pathlib.Path(__file__).parent.parent / "notebooks" / "A2_what_can_it_compute.ipynb"
out.parent.mkdir(exist_ok=True); nbf.write(nb, out); print("wrote", out)
