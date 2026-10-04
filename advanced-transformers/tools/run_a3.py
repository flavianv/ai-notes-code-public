# A3 experiment: train one setup on lengths 1..20, test at longer lengths and with more I actions. Writes results/a3/<fmt>_s<seed>.json
import json, sys, time, torch
from atlib.automata import train, evaluate

fmt, seed = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
dev = "mps" if fmt == "looped" else "cpu"
torch.set_num_threads(2)
t0 = time.time(); log = []
model = train(fmt, steps=steps, seed=seed, dev=dev, log=log)
out = {"fmt": fmt, "seed": seed, "steps": steps, "train_seconds": round(time.time() - t0), "loss": log,
       "params": sum(p.numel() for p in model.parameters()), "lengths": {}, "i_heavy": {}}
for n in (10, 20, 50, 100, 200):
    out["lengths"][n] = evaluate(model, fmt, n, dev=dev)
for n in (20, 100):                                                      # 80% identity actions: same rule, different statistics
    out["i_heavy"][n] = evaluate(model, fmt, n, dev=dev, p=(0.1, 0.1, 0.8))
json.dump(out, open(f"results/a3/{fmt}_s{seed}.json", "w"), indent=1)
print(fmt, seed, {n: r["final"] for n, r in out["lengths"].items()})
