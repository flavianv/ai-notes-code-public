"""Transformer Training Dynamics (AI Notes), part VII: generalization. Modular addition a + b mod 97, small MLP, CPU.

Three conditions, seeds 0-2, full-batch AdamW:
  grok      real labels, weight decay 1.0   -> memorizes first, generalizes much later
  no_decay  real labels, weight decay 0.0   -> the control: same model and data, no decay
  random    shuffled labels, weight decay 1.0 -> the same network fits pure noise; test stays at chance
Run: python3 training-dynamics/grokking.py   (writes results/grokking.json; a few minutes on a laptop CPU)
"""
import json, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
P, FRAC, STEPS, EVERY = 97, 0.3, 20000, 100


class MLP(nn.Module):
    def __init__(self, p=P, d=128, h=512):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.w1 = nn.Linear(2 * d, h)
        self.w2 = nn.Linear(h, p)

    def forward(self, ab):
        x = self.emb(ab).flatten(1)          # concatenate the two operand embeddings
        return self.w2(F.relu(self.w1(x)))


def data(seed, shuffle):
    g = torch.Generator().manual_seed(seed)
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    ab, y = torch.stack([a.flatten(), b.flatten()], 1), (a + b).flatten() % P
    perm = torch.randperm(P * P, generator=g); n = int(FRAC * P * P)
    tr, te = perm[:n], perm[n:]
    ytr = y[tr][torch.randperm(n, generator=g)] if shuffle else y[tr]   # random labels: same inputs, labels permuted
    return ab[tr], ytr, ab[te], y[te]


def run(cond, seed, steps=STEPS):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data(seed, cond == "random")
    model = MLP(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0 if cond == "no_decay" else 1.0, betas=(0.9, 0.98))
    log = []
    for step in range(steps + 1):
        if step % EVERY == 0:
            with torch.no_grad():
                acc = lambda x, y: (model(x).argmax(-1) == y).float().mean().item()
                norm = sum(p.pow(2).sum() for p in model.parameters()).sqrt().item()
                log.append([step, acc(xtr, ytr), acc(xte, yte), norm])
        loss = F.cross_entropy(model(xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    first = lambda k, th: next((s for s, *v in log if v[k] >= th), None)
    return {"cond": cond, "seed": seed, "log": log, "train_99": first(0, .99), "test_99": first(1, .99), "final_train": log[-1][1], "final_test": log[-1][2]}


def main():
    torch.set_num_threads(4)
    out = []
    for cond in ["grok", "no_decay", "random"]:
        for seed in range(3):
            r = run(cond, seed); out.append(r)
            print(cond, seed, "train≥99% at", r["train_99"], "test≥99% at", r["test_99"], "final", round(r["final_train"], 3), round(r["final_test"], 3), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "grokking.json").write_text(json.dumps({"p": P, "train_fraction": FRAC, "steps": STEPS, "runs": out}) + "\n")


if __name__ == "__main__":
    main()
