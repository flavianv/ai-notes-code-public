# Builds notebooks/02_attention_from_scratch.ipynb (run: python tools/make_nb02.py)
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
md("""
# Intro to Transformers · part 2 of 3: attention from scratch
Companion notebook to the video: queries, keys, values, the causal mask, multi-head / multi-query / grouped-query attention, and FlashAttention's tiled online softmax, all in plain PyTorch.

**The series:** 1. the general picture · 2. attention from scratch (this notebook) · 3. production (KV cache, MoE, speculative decoding, quantization).
""")
code("""
import sys, subprocess, math, time
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/flavianv/ai-notes-code-public"], check=True)
import torch, torch.nn.functional as F
from tlib.attention import attention, MultiHeadAttention, Block, flash_attention
from tlib import device
torch.manual_seed(0); DEV = device(); print("device:", DEV)
""")
md("## 1. Attention in ten lines\nscores = Q·Kᵀ/√d, mask the future, softmax (a vote), then a weighted average of the values.")
code("""
import inspect
print(inspect.getsource(attention))
""")
code("""
q, k, v = (torch.randn(2, 8, 256, 64) for _ in range(3))
for causal in (False, True):
    ours, ref = attention(q, k, v, causal), F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    print("causal =", causal, " max diff vs PyTorch SDPA:", float((ours - ref).abs().max()))
""")
md("## 2. Multi-head, multi-query, grouped-query\n`n_kv_heads` = number of heads = MHA, 1 = MQA, in between = GQA. Fewer key-value heads means a smaller KV cache.")
code("""
for kv in (8, 2, 1):
    m = MultiHeadAttention(512, 8, kv)
    print(f"kv heads = {kv}: parameters {sum(p.numel() for p in m.parameters()):,}   KV cache per token per layer (fp16): {2 * kv * 64 * 2} bytes")
x = torch.randn(2, 128, 512); print(Block(512, 8, 2)(x).shape)
""")
md("The KV cache for a Llama-3-8B-like model (32 layers, head size 128, fp16):")
code("""
for name, kv in (("MHA, 32 kv heads", 32), ("GQA, 8 kv heads", 8), ("MQA, 1 kv head", 1)):
    per = 2 * 32 * kv * 128 * 2
    print(f"{name:18s} {per / 1024:6.0f} KiB per token   {per * 8192 / 2**30:6.3f} GiB at 8k tokens")
""")
md("## 3. FlashAttention: tiled, with an online softmax\nNever builds the n×n matrix: stream blocks of keys and values past a block of queries, keeping a running max, sum and output.")
code("""
print(inspect.getsource(flash_attention))
q, k, v = (torch.randn(2, 8, 256, 64) for _ in range(3))
print("tiled vs plain attention, max diff:", float((flash_attention(q, k, v, block=64) - attention(q, k, v)).abs().max()))
""")
md("## 4. Timing: the n×n bill\nOur naive attention materialises the score matrix; PyTorch's `scaled_dot_product_attention` picks a faster kernel (the real FlashAttention kernel needs a supported NVIDIA GPU).")
code("""
sync = torch.mps.synchronize if DEV == "mps" else (torch.cuda.synchronize if DEV == "cuda" else (lambda: None))
for n in (512, 1024, 2048, 4096):
    q, k, v = (torch.randn(1, 8, n, 64, device=DEV) for _ in range(3)); row = {}
    for name, fn in (("naive", lambda: attention(q, k, v, True)), ("sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))):
        fn(); sync(); t = time.perf_counter()
        for _ in range(3): fn()
        sync(); row[name] = (time.perf_counter() - t) / 3 * 1000
    print(n, {k_: round(v_, 2) for k_, v_ in row.items()}, "ms;  score matrix", round(8 * n * n * 4 / 2**20), "MB")
""")
md("## Next\nPart 3: production, the KV cache's memory costs, mixture of experts, speculative decoding, quantization.")
nb["cells"] = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/02_attention_from_scratch.ipynb")
