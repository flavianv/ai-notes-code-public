# Builds notebooks/04_making_models_small.ipynb (part 3b: quantization, GGUF/Ollama, QLoRA-style fine-tuning)
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Intro to Transformers · part 3b: making models small
Companion notebook to the video: quantization from scratch (absmax, per-channel, group-wise, NF4), reading a real GGUF file, and a QLoRA-style fine-tune in plain PyTorch.
""")
code("""
import sys, subprocess, os, math, shutil
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/flavianv/ai-notes-code-public", "gguf"], check=True)
import torch
from tlib import GPT, next_token_loss, shakespeare, device
from tlib.quant import quant_absmax, dequant_absmax, fake_quant, scheme_bytes, add_qlora, NF4
torch.manual_seed(0); DEV = device(); print("device:", DEV)
""")
md("## 1. Absmax quantization, by hand\nThe example from the video: eight weights, 4 bits, scale = max|w| / 7.")
code("""
w = torch.tensor([0.12, -0.80, 0.31, 0.05, -0.44, 0.67, -0.02, 0.25])
q, scale = quant_absmax(w, bits=4)
print("scale:", float(scale), " integers:", q.tolist())
print("back :", [round(v, 3) for v in dequant_absmax(q, scale).tolist()])
print("max error:", float((dequant_absmax(q, scale) - w).abs().max()), "<= half a step:", float(scale) / 2)
""")
md("## 2. Which scheme? Error on Gaussian weights, and with one outlier")
code("""
g = torch.randn(512, 512); go = g.clone(); go[0, 0] = 50.0
mse = lambda x, s: (fake_quant(x, s) - x).pow(2).mean().item()
for s in ("int8-channel", "int4-tensor", "int4-g64", "nf4-g64"): print(f"{s:13s} gaussian {mse(g, s):.5f}   with one outlier {mse(go, s):.5f}")
""")
md("## 3. Quantize a model and measure it\nA small GPT trained briefly on Tiny Shakespeare (`STEPS`, default 300; the video used a 3.2 M model trained longer). Every linear layer is quantized; embeddings and norms stay in 16 bits.")
code("""
STEPS = int(os.environ.get("STEPS", 300)); T, B = 128, 64
text, chars, data = shakespeare(); n = int(.9 * len(data)); train, val = data[:n], data[n:]
batch = lambda src: torch.stack([src[i:i + T + 1] for i in torch.randint(0, len(src) - T - 1, (B,))]).to(DEV)
base = GPT(len(chars), 128, 4, 4).to(DEV); opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
for s in range(STEPS + 1):
    loss = next_token_loss(base, batch(train)); opt.zero_grad(); loss.backward(); opt.step()
gv = torch.Generator().manual_seed(1); VB = [torch.stack([val[i:i + T + 1] for i in torch.randint(0, len(val) - T - 1, (B,), generator=gv)]).to(DEV) for _ in range(20)]
@torch.no_grad()
def val_loss(m): m.eval(); return sum(next_token_loss(m, b).item() for b in VB) / len(VB)
def quantized(scheme):
    m = GPT(len(chars), 128, 4, 4).to(DEV); m.load_state_dict(base.state_dict())
    with torch.no_grad():
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and not name.startswith("head"): mod.weight.copy_(fake_quant(mod.weight.data.cpu(), scheme).to(DEV))
    return m
lin = sum(mod.weight.numel() for name, mod in base.named_modules() if isinstance(mod, torch.nn.Linear) and not name.startswith("head"))
print(f"{'fp32':13s} val loss {val_loss(base):.4f}   linear weights {lin * 4 / 2**20:6.2f} MB")
for s in ("int8-channel", "int4-tensor", "int4-g64", "nf4-g64"):
    print(f"{s:13s} val loss {val_loss(quantized(s)):.4f}   linear weights {scheme_bytes(lin, s) / 2**20:6.2f} MB")
""")
md("## 4. QLoRA-style fine-tuning\nFreeze the NF4 base, train only small low-rank adapters (rank 8). Compare with the same adapters on the full-precision base.")
code("""
def tune(m, steps=STEPS):
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for s in range(steps + 1):
        m.train(); loss = next_token_loss(m, batch(train)); opt.zero_grad(); loss.backward(); opt.step()
    return val_loss(m)
mq = GPT(len(chars), 128, 4, 4).to(DEV); mq.load_state_dict(base.state_dict()); mq = add_qlora(mq, r=8).to(DEV)
trainable = sum(p.numel() for p in mq.parameters() if p.requires_grad)
print("NF4 base before tuning:", round(val_loss(mq), 4), f"  trainable adapter parameters: {trainable:,}")
print("NF4 + LoRA after tuning:", round(tune(mq), 4))
""")
md("""
Memory to fine-tune a 7 B model (estimates): full fine-tuning with Adam is about 16 bytes per parameter (weights, gradients, two moments) = **112 GB**; an NF4 base plus ~40 M adapter parameters is about **4.4 GB** before activations.
For the real thing on large models use `bitsandbytes` on an NVIDIA GPU (a free Colab T4 works) or MLX on Apple silicon.
""")
md("## 5. Look inside a real GGUF file (optional)\nNeeds Ollama and the `gguf` package: `brew install ollama && ollama serve`, `ollama pull qwen2.5:0.5b`, `pip install gguf`.")
code("""
import re, collections
try:
    from gguf import GGUFReader
    out = subprocess.run(["ollama", "show", "qwen2.5:0.5b", "--modelfile"], capture_output=True, text=True).stdout
    blob = re.search(r"^FROM (\\S+)", out, re.M).group(1); rd = GGUFReader(blob)
    print(os.path.getsize(blob) // 2**20, "MB,", len(rd.tensors), "tensors,", len(rd.fields), "metadata fields")
    by = collections.Counter(t.tensor_type.name for t in rd.tensors); print(dict(by))
except Exception as e:
    print("skipped (needs gguf + a pulled Ollama model):", type(e).__name__)
""")
md("## Next\nThat completes the introduction: the general picture, attention, production. For fine-tuning beyond adapters (reinforcement learning), see the RL playlist.")
nb["cells"] = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/04_making_models_small.ipynb")
