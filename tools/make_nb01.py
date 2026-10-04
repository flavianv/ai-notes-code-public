# Builds notebooks/01_intro_to_transformers.ipynb (run: python tools/make_nb01.py)
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Intro to Transformers · part 1 of 3: the general picture
Companion notebook to the video. Every number you saw on screen is computed here.

**The series:** 1. the general picture (this notebook) · 2. attention from scratch, up to FlashAttention · 3. production: MoE, speculative decoding, quantization.
Fine-tuning after pretraining: see the RL playlist on the channel.

Runs on a laptop (CPU or Apple GPU) or Colab. Training takes about 2 minutes on an Apple GPU.
""")
code("""
import sys, subprocess, os
if "google.colab" in sys.modules:   # on Colab, install the course library from GitHub
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/flavianv/ai-notes-code-public"], check=True)
import torch
from tlib import *
torch.manual_seed(0)
DEV = device(); print("device:", DEV)
""")
md("## 1. Tokenisation: byte-pair encoding\nStart from characters, repeatedly merge the most common adjacent pair.")
code("""
merges = bpe_train({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 4)
print(merges)                       # [('e','s'), ('es','t'), ('l','o'), ('lo','w')]
print(bpe_encode("newest", merges))  # ['n', 'e', 'w', 'est']
print(bpe_encode("lowest", merges))  # unseen word, known pieces: ['low', 'est']
""")
md("*For real:* production tokenisers work on raw bytes and are trained on hundreds of GB; use `tiktoken`, SentencePiece or Hugging Face tokenizers. The algorithm is the one above.")
md("## 2. Embeddings\nA learned table: token id in, one row (a vector) out.")
code("""
emb = torch.nn.Embedding(1000, 128)
print(emb(torch.tensor([791, 10065 % 1000, 3287 % 1000])).shape)   # (3 tokens, 128 numbers each)
""")
md("## 3. Position: RoPE\nRotate each (even, odd) pair of features by *position × frequency*. The query·key score then depends only on the distance.")
code("""
q, k = torch.randn(64), torch.randn(64)
dot = lambda m, n: float(rope(q[None], torch.tensor([float(m)]))[0] @ rope(k[None], torch.tensor([float(n)]))[0])
print("positions (5, 3)    :", round(dot(5, 3), 4))
print("positions (105, 103):", round(dot(105, 103), 4), " <- same distance, same score")
print("positions (5, 4)    :", round(dot(5, 4), 4), " <- different distance")
""")
md("The clock-hand example from the video: one pair turning 30° per position. Query at position 3 (90°), key at position 1 (30°): angle between = 60°, score ∝ cos 60° = 0.5. Slide both by 5: still 60°.")
code("""
import math
for pq, pk in [(3, 1), (8, 6)]:
    print(pq, pk, "angle between:", (pq - pk) * 30, "deg  cos =", round(math.cos(math.radians((pq - pk) * 30)), 2))
""")
md("## 4. The block\nAttention + MLP, each with a residual connection and a pre-norm. Read the source:")
code("""
import inspect
print(inspect.getsource(Block))
""")
code("""
m = GPT(50).eval()
a = torch.randint(0, 50, (1, 10)); b = a.clone(); b[0, 7] = (b[0, 7] + 1) % 50
with torch.no_grad():
    print("causal mask, max change before token 7:", float((m(a)[:, :7] - m(b)[:, :7]).abs().max()))        # exactly 0
    full = m(a); caches = [{} for _ in m.blocks]
    steps = torch.cat([m(a[:, t:t + 1], t, caches) for t in range(10)], 1)
    print("KV cache vs full pass, max diff:", float((steps - full).abs().max()))                         # ~1e-7
""")
md("## 5. Training: next-token prediction on Tiny Shakespeare\nHold out the last 10% as a validation set. Watch validation loss: it bottoms out and then rises (overfitting) when the model is big enough.")
code("""
STEPS = int(os.environ.get("STEPS", 1500)); T, B = 128, 64
text, chars, data = shakespeare()
n = int(0.9 * len(data)); train, val = data[:n], data[n:]
model = GPT(len(chars), d=128, n_heads=4, n_layers=4).to(DEV)
print("params:", sum(p.numel() for p in model.parameters()), " ln(vocab) =", round(math.log(len(chars)), 2))
batch = lambda src: torch.stack([src[i:i + T + 1] for i in torch.randint(0, len(src) - T - 1, (B,))]).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(STEPS + 1):
    if step % max(1, STEPS // 6) == 0:
        model.eval()
        with torch.no_grad(): vl = sum(next_token_loss(model, batch(val)).item() for _ in range(10)) / 10
        model.train(); print(f"step {step:5d}  val loss {vl:.3f}")
    loss = next_token_loss(model, batch(train)); opt.zero_grad(); loss.backward(); opt.step()
""")
md("## 6. Generation\nLogits → temperature → top-k/top-p → softmax → sample, with a KV cache.")
code("""
z = torch.tensor([2.0, 1.0, 0.1])
print("T=1.0:", z.softmax(-1).tolist()); print("T=0.5:", (z / 0.5).softmax(-1).tolist())
stoi = {c: i for i, c in enumerate(chars)}
prompt = torch.tensor([[stoi[c] for c in "ROMEO:\\n"]], device=DEV)
for kw in [dict(temperature=0.7, top_k=5), dict(temperature=1.0)]:
    print(kw); print("".join(chars[i] for i in generate(model, prompt, 200, **kw)[0].tolist())); print("-" * 40)
""")
md("""
## Next
Part 2: attention from scratch, multi-head / grouped-query, and FlashAttention. Part 3: the KV cache's memory costs, mixture of experts, speculative decoding, quantization.
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/01_intro_to_transformers.ipynb")
