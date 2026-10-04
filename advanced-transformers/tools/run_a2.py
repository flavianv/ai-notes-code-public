# A2 experiment: one forward pass on the machine. Depth x length (state labels at every position), counting vs state, probes. Writes results/a2/*.json
import json, sys, torch
from atlib.expressivity import *

torch.set_num_threads(2)
what, seed = sys.argv[1], int(sys.argv[2])
if what == "depth":                                                      # train AND test at one fixed length n; how deep must the model be?
    out = {}
    for n in (8, 32, 128):
        for L in (1, 2, 4):
            m = train_labelled("state", n_layers=L, lengths=(n, n), steps=2000, seed=seed)
            out[f"n{n}_L{L}"] = labelled_accuracy(m, "state", n)
            print(n, L, out[f"n{n}_L{L}"], flush=True)
else:                                                                    # train on 1..20, test longer: counting vs state tracking, then probe
    out = {}
    for task in ("count", "state"):
        m = train_labelled(task, n_layers=4, lengths=(1, 20), steps=3000, seed=seed)
        out[task] = {n: labelled_accuracy(m, task, n) for n in (10, 20, 50, 100, 200)}
        if task == "state": out["probe"] = probe(m)
json.dump(out, open(f"results/a2/{what}_s{seed}.json", "w"), indent=1)
print(what, seed, "done")
