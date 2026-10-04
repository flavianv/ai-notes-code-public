# A1 experiment: induction circuit (1 vs 2 layers, head scores, ablation), superposition toy model, sparse autoencoder. Writes results/a1/*.json
import json, sys, torch
from atlib.circuits import *
from atlib.automata import TRANSITIONS

torch.set_num_threads(2)
what = sys.argv[1]
if what == "ablate":                                                     # recompute the group ablations from saved models, no retraining
    for seed in (0, 1, 2):
        m = AttnOnly(50); m.load_state_dict(torch.load(f"results/a1/induction_s{seed}.pt")); m.eval()
        p = f"results/a1/induction_s{seed}.json"; d = json.load(open(p)); d["L2"]["group_ablate"] = group_ablation(m)
        json.dump(d, open(p, "w"), indent=1)
elif what == "induction":
    seed = int(sys.argv[2]); out = {"seed": seed}
    for L in (1, 2):
        log = []; m = train_induction(n_layers=L, steps=3000, seed=seed, log=log)
        prev, ind = head_scores(m)
        r = {"curve": log, "loss_halves": halves_loss(m), "acc": {h: repeat_accuracy(m, half=h) for h in (8, 20, 32, 50)},
             "prev": {f"{l}.{h}": v for (l, h), v in prev.items()}, "ind": {f"{l}.{h}": v for (l, h), v in ind.items()}}
        if L == 2:                                                       # switch off the best previous-token head, then the best induction head
            p0 = max((k for k in prev if k[0] == 0), key=prev.get); i1 = max((k for k in ind if k[0] == 1), key=ind.get)
            r["ablate"] = {"prev_head": f"{p0[0]}.{p0[1]}", "ind_head": f"{i1[0]}.{i1[1]}",
                           "acc_prev_off": repeat_accuracy(m, off={p0}), "acc_ind_off": repeat_accuracy(m, off={i1}),
                           "acc_other_off": repeat_accuracy(m, off={k for k in prev if k[0] == 0 and k != p0})}
            r["group_ablate"] = group_ablation(m)
            torch.save(m.state_dict(), f"results/a1/induction_s{seed}.pt")
        out[f"L{L}"] = r
    json.dump(out, open(f"results/a1/induction_s{seed}.json", "w"), indent=1)
else:                                                                    # superposition + SAE
    out = {"softmax_example": torch.tensor([0., 6., 0., 0.]).softmax(-1)[1].item(), "toy": {}}
    for S in (0.0, 0.5, 0.7, 0.9, 0.97):
        W, b = train_toy(sparsity=S)
        out["toy"][S] = {"represented": represented(W), "norms": W.norm(dim=0).tolist(), "W": W.tolist()}
    W, _ = train_toy(sparsity=0.9)
    D = train_sae(W, sparsity=0.9)
    out["sae"] = {"recovery": recovery(W, D).tolist(), "dict": D.tolist(), "live": int((D.norm(dim=0) > 0.1).sum())}
    json.dump(out, open("results/a1/superposition.json", "w"), indent=1)
print(what, "done")
