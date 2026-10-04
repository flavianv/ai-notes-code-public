# Understanding Transformers, A5: a tape as two stacks, a Turing machine that adds one, a learned controller, and an end-to-end contrast.
import random, torch, torch.nn as nn, torch.nn.functional as F
from tlib.model import GPT

BLANK = "_"
SYMS = ["0", "1", BLANK]


# 1. The tape as two stacks: L holds the cells left of the head (top = nearest), R holds the current cell and everything right (top = current)
class TwoStacks:
    def __init__(self, s):
        self.L, self.R = [], list(reversed(s))                            # R's top (end of the list) is the first cell
    def read(self): return self.R[-1] if self.R else BLANK
    def left_top(self): return self.L[-1] if self.L else BLANK
    def write(self, x):
        if self.R: self.R[-1] = x
        else: self.R.append(x)
    def move(self, d):
        if d == "R": self.L.append(self.R.pop() if self.R else BLANK)     # pass the current cell to the left stack
        else: self.R.append(self.L.pop() if self.L else BLANK)            # take the nearest left cell back
    def tape(self): return ("".join(self.L) + "".join(reversed(self.R))).strip(BLANK)


# 2. Binary increment: walk right to the end, then turn trailing 1s into 0s and the first 0 (or blank) into 1
#    The controller only sees (state, left top, right top) and answers (write, move, next state)
INCREMENT = {("R", "0"): ("0", "R", "R"), ("R", "1"): ("1", "R", "R"), ("R", BLANK): (BLANK, "L", "C"),
             ("C", "1"): ("0", "L", "C"), ("C", "0"): ("1", "L", "H"), ("C", BLANK): ("1", "L", "H")}
STATES = ["R", "C"]


def table_controller(state, left, right): return INCREMENT[(state, right)]


def run_tm(s, controller, max_steps=None):
    t, state, steps = TwoStacks(s), "R", 0
    max_steps = max_steps or 4 * len(s) + 10
    while state != "H" and steps < max_steps:
        w, d, state = controller(state, t.left_top(), t.read()); t.write(w); t.move(d); steps += 1
    return t.tape(), steps


def increment(s): return bin(int(s, 2) + 1)[2:]


# 3. A learned controller: a small MLP on the one-hot (state, left top, right top) -> (write, move, next state)
ACTS = [(w, d, n) for w in SYMS for d in "LR" for n in ["R", "C", "H"]]


def features(state, left, right):
    x = torch.zeros(2 + 3 + 3); x[STATES.index(state)] = 1; x[2 + SYMS.index(left)] = 1; x[5 + SYMS.index(right)] = 1; return x


def controller_data(max_len=4):                                           # every (input, action) the table uses on all inputs up to max_len digits
    seen = {}
    for n in range(1, max_len + 1):
        for v in range(2 ** n):
            t, state = TwoStacks(format(v, f"0{n}b")), "R"
            while state != "H":
                key = (state, t.left_top(), t.read()); a = table_controller(*key); seen[key] = a
                t.write(a[0]); t.move(a[1]); state = a[2]
    return seen


def train_controller(max_len=4, steps=400, seed=0):
    torch.manual_seed(seed); data = controller_data(max_len)
    X = torch.stack([features(*k) for k in data]); Y = torch.tensor([ACTS.index(a) for a in data.values()])
    net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, len(ACTS))); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = F.cross_entropy(net(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    def ctrl(state, left, right):
        with torch.no_grad(): return ACTS[net(features(state, left, right)[None]).argmax().item()]
    return ctrl, len(data)


def exhaustive_check(ctrl):                                               # every possible input the controller can see while running
    combos = [(s, l, r) for s in STATES for l in SYMS for r in SYMS]
    return sum(ctrl(*k) == table_controller(*k) for k in combos), len(combos)


# 4. End to end: a transformer that writes the incremented number directly
#    tokens: 0, 1, '=' (2), BOS (3); input bits, '=', then the answer's bits (always n + 1 digits, leading zero kept)
EQ, BOS_T = 2, 3


def e2e_batch(n, size, rng):
    xs = [format(rng.getrandbits(n), f"0{n}b") for _ in range(size)]
    ys = [format(int(x, 2) + 1, f"0{n + 1}b") for x in xs]
    ids = torch.tensor([[BOS_T] + [int(c) for c in x] + [EQ] + [int(c) for c in y] for x, y in zip(xs, ys)])
    return ids, n


def train_e2e(max_len=16, steps=3000, seed=0, d=128, layers=4):
    torch.manual_seed(seed); rng = random.Random(seed)
    m = GPT(4, d=d, n_heads=4, n_layers=layers); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        ids, n = e2e_batch(rng.randint(1, max_len), 128, rng)
        logits = m(ids[:, :-1])[:, n + 1:, :2]                            # predict only the answer digits
        loss = F.cross_entropy(logits.reshape(-1, 2), ids[:, n + 2:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def e2e_accuracy(m, n, size=300, seed=5):
    ids, _ = e2e_batch(n, size, random.Random(seed + n)); x = ids[:, :n + 2]
    for _ in range(n + 1):                                                # greedy, digit by digit
        nxt = m(x)[:, -1, :2].argmax(-1, keepdim=True); x = torch.cat([x, nxt], 1)
    return (x[:, n + 2:] == ids[:, n + 2:]).all(1).float().mean().item()
