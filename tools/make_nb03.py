# Builds notebooks/03_serving_faster.ipynb (part 3a: KV-cache memory, mixture of experts, speculative decoding)
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Intro to Transformers · part 3a: serving faster
Companion notebook to the video: why generation is memory-bound, the KV cache's memory bill, mixture of experts, and speculative decoding, all in PyTorch.
Part 3b (making models small: quantization, GGUF/Ollama, 4-bit training) has its own notebook.
""")
code("""
import sys, subprocess, os, math, time
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/flavianv/ai-notes-code-public"], check=True)
import torch
from tlib import GPT, next_token_loss, shakespeare, device
from tlib.serving import kv_bytes, MoE, MoEGPT, speculative
torch.manual_seed(0); DEV = device(); print("device:", DEV)
""")
md("## 1. Generation is memory-bound (back of the envelope)\nPublished H100 peaks: 3.35 TB/s memory bandwidth, ~989 TFLOP/s (dense BF16). Real numbers are worse; the gap is the point.")
code("""
params, bytes_per = 8e9, 2
weights = params * bytes_per
read_ms = weights / 3.35e12 * 1e3                 # read every weight once per token
math_ms = 2 * params / 989e12 * 1e3               # 2 FLOPs per parameter
print(f"weights {weights / 1e9:.0f} GB: read {read_ms:.2f} ms, compute {math_ms * 1000:.1f} us per token, ratio {read_ms / math_ms:.0f}x")
""")
md("## 2. The KV cache's memory bill\n`2 × layers × kv_heads × head_dim × tokens × bytes`")
code("""
GiB = 2 ** 30
for name, heads in (("MHA (32 kv heads)", 32), ("GQA (8)", 8), ("MQA (1)", 1)):
    print(f"{name:18s} {kv_bytes(32, heads, 128, 1) / 1024:5.0f} KiB/token   8k tokens: {kv_bytes(32, heads, 128, 8192) / GiB:6.3f} GiB   32 conversations: {kv_bytes(32, heads, 128, 8192, 32) / GiB:6.1f} GiB")
assert kv_bytes(32, 8, 128, 8192) == 1 * GiB
""")
md("## 3. Mixture of experts\nA router picks the top-2 of 8 experts per token; only those run. The sparse version equals a dense reference that runs everything.")
code("""
m = MoE(32, n_experts=8, k=2, hidden=64); x = torch.randn(2, 10, 32)
print("sparse vs dense max diff:", float((m(x) - m.forward_dense(x)).abs().max()))
print("share of tokens per expert:", [round(v, 2) for v in m.load.tolist()], " balancing loss:", round(float(m.aux), 3), "(1.0 = perfectly even)")
""")
md("Dense vs MoE on Tiny Shakespeare (same compute per token: dense MLP hidden 512 vs 8 experts of 256, top-2). In the video: 1500 steps; here `STEPS` is small by default so it runs in a couple of minutes.")
code("""
STEPS = int(os.environ.get("STEPS", 300)); T, B = 128, 64
text, chars, data = shakespeare(); n = int(.9 * len(data)); train, val = data[:n], data[n:]
batch = lambda src: torch.stack([src[i:i + T + 1] for i in torch.randint(0, len(src) - T - 1, (B,))]).to(DEV)
def fit(model, aux=0.0):
    model.to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for s in range(STEPS + 1):
        loss = next_token_loss(model, batch(train)); tot = loss + aux * model.aux_loss() if aux else loss
        opt.zero_grad(); tot.backward(); opt.step()
    model.eval()
    with torch.no_grad(): return sum(next_token_loss(model, batch(val)).item() for _ in range(10)) / 10
for name, ne, k, hid, aux in (("dense", 1, 1, 512, 0.0), ("MoE + balancing", 8, 2, 256, 0.02)):
    torch.manual_seed(0); mdl = MoEGPT(len(chars), 128, 4, 4, ne, k, hid)
    print(f"{name:16s} total params {sum(p.numel() for p in mdl.parameters()):,}  val loss after {STEPS} steps: {fit(mdl, aux):.3f}")
""")
md("## 4. Speculative decoding\nA small draft model proposes k tokens; the big model checks them in one pass. With greedy decoding the output must equal the big model's own, token for token.")
code("""
target, draft = GPT(len(chars), 128, 4, 3).to(DEV), GPT(len(chars), 32, 2, 1).to(DEV)
for mdl, lr in ((target, 1e-3), (draft, 3e-3)):
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr)
    for s in range(STEPS + 1):
        loss = next_token_loss(mdl, batch(train)); opt.zero_grad(); loss.backward(); opt.step()
    mdl.eval()
prompt = train[:24][None].to(DEV)
out, stats = speculative(target, draft, prompt, 100, k=4, greedy=True)
ref = prompt
with torch.no_grad():
    for _ in range(100): ref = torch.cat([ref, target(ref)[:, -1].argmax(-1, keepdim=True)], 1)
print("identical to the target's own greedy output:", torch.equal(out, ref))
print(f"accepted {stats['accepted']} of {stats['proposed']} proposals; {100 / stats['target_calls']:.2f} tokens per target pass")
""")
md("## Next\nPart 3b: making models small: quantization, the GGUF format, running locally with Ollama, and training in 4 bits.")
nb["cells"] = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/03_serving_faster.ipynb")
