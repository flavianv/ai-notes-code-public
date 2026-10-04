# A4 experiment: the trained interleaved model inside five systems. Needs results/a4/interleaved_s<seed>.pt (tools/train_a4_models.py).
import json, sys, time, torch
from atlib.systems import load, sampling, search, harness, workers, reliability
torch.set_num_threads(3)
seed = int(sys.argv[1]); m = load(f"results/a4/interleaved_s{seed}.pt"); t0 = time.time()
if "--workers-only" in sys.argv:                                          # recompute just the workers experiment into an existing result file
    p = f"results/a4/systems_s{seed}.json"; d = json.load(open(p)); d["workers"] = workers(m, seed=seed); json.dump(d, open(p, "w"), indent=1); print("workers", d["workers"]); sys.exit()
out = {"seed": seed,
       "sampling": {n: sampling(m, n, seed=seed) for n in (50, 100)},
       "search": {200: search(m, 200, seed=seed)},
       "harness": {n: harness(m, n, seed=seed) for n in (200, 1000)},
       "workers": workers(m, seed=seed),
       "reliability": reliability(m, seed=seed)}
out["seconds"] = round(time.time() - t0)
json.dump(out, open(f"results/a4/systems_s{seed}.json", "w"), indent=1); print("done", seed, out["seconds"], "s")
