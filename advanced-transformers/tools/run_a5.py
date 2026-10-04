# A5 experiment: exact table machine, learned controller (checked on every input), end-to-end transformer. Writes results/a5/tape_s<seed>.json
import json, random, sys, time, torch
from atlib.tape import run_tm, table_controller, increment, train_controller, exhaustive_check, controller_data, train_e2e, e2e_accuracy, INCREMENT
torch.set_num_threads(2)
seed = int(sys.argv[1]); rng = random.Random(seed); t0 = time.time(); out = {"seed": seed, "table_rows": len(INCREMENT)}
long = ["1" + "".join(rng.choice("01") for _ in range(999)) for _ in range(100)]
out["table_1000_digits"] = sum(run_tm(s, table_controller)[0] == increment(s) for s in long) / len(long)
ctrl, n_seen = train_controller(seed=seed); ok, total = exhaustive_check(ctrl)
out["controller"] = {"inputs_seen_in_training": n_seen, "exhaustive_ok": ok, "exhaustive_total": total,
                     "acc_1000_digits": sum(run_tm(s, ctrl)[0] == increment(s) for s in long) / len(long)}
m = train_e2e(seed=seed)
out["e2e"] = {n: e2e_accuracy(m, n) for n in (8, 16, 24, 32, 48, 64)}
out["seconds"] = round(time.time() - t0)
json.dump(out, open(f"results/a5/tape_s{seed}.json", "w"), indent=1); print("done", seed, out)
