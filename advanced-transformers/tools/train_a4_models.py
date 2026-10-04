# Train and save the interleaved A3 model (same settings as A3: lengths 1-20, 3000 steps) for the A4 system experiments.
import sys, torch
from atlib.automata import train
torch.set_num_threads(2)
seed = int(sys.argv[1])
m = train("interleaved", steps=3000, seed=seed)
torch.save(m.state_dict(), f"results/a4/interleaved_s{seed}.pt"); print("saved", seed)
