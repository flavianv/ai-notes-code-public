# ai-notes-code

Code for the AI Notes YouTube channel: every number shown in a video is computed here.

## Transformers series

| Part | Video | Notebook | Library |
|---|---|---|---|
| 1 | Intro to Transformers: the general picture | `notebooks/01_intro_to_transformers.ipynb` | `tlib.bpe`, `tlib.model`, `tlib.sampling` |
| 2 | Attention from scratch, up to FlashAttention | `notebooks/02_attention_from_scratch.ipynb` | `tlib.attention` |
| 3a | Production: serving faster (KV-cache memory, mixture of experts, speculative decoding) | `notebooks/03_serving_faster.ipynb` | `tlib.serving` |
| 3b | Production: making models small (quantization, GGUF/Ollama, 4-bit training) | `notebooks/04_making_models_small.ipynb` | `tlib.quant` |

Fine-tuning after pretraining: see the RL playlist on the channel.

## Understanding Transformers (advanced series)

Five episodes, A1–A5, in [`advanced-transformers/`](advanced-transformers/README.md): circuits, what one forward pass can compute, what training learns, computation as a system, and universality and scaling. Each has a notebook and a test file that pins every number in the video.

- Videos: [the playlist](https://www.youtube.com/playlist?list=PLSQN85XNghpY)
- Written guide: [Understanding Transformers: what they compute, and what they need](https://flavianv.github.io/articles/understanding-transformers.html)
- Bonus episode on real LLMs: AbstractGym ([part 1](https://youtu.be/QIBMdjqpPAs), [part 2](https://youtu.be/q38WEwK3fEI)), code in [flavianv/abstractgym-public](https://github.com/flavianv/abstractgym-public)

## Transformer Training Dynamics

One film on how training shapes a Transformer: signals at initialization, gradients through attention, optimizer updates (SGD, AdamW, Muon, Shampoo) and the representations they build. Its NumPy lab, with a test pinning every on-screen number, is in [`training-dynamics/`](training-dynamics/README.md).

## Run it

```bash
pip install -e ".[dev]"      # needs Python 3.9+, installs PyTorch
pytest                       # checks every number from the videos (CPU is enough)
jupyter notebook notebooks/
```

Runs on CPU, Apple-silicon GPU (MPS) or CUDA; `tlib.device()` picks the best one. On Colab, the first notebook cell installs this repo from GitHub.
The Shakespeare training in notebook 1 takes about 2 minutes on an Apple GPU (`STEPS=120` for a quick run).

## Layout

- `tlib/`: the library: `bpe`, `model` (RoPE, attention with KV cache, GPT), `sampling`, `attention` (part 2, as in the film), `serving` (KV-cache maths, MoE, speculative decoding), `quant` (int8/int4/NF4 quantization and a QLoRA-style layer).
- `tests/`: one test per claim in the videos (part 1: BPE merges, softmax numbers, RoPE offset invariance, causal mask, KV cache equals full pass; part 2: attention equals PyTorch SDPA, tiled FlashAttention equals plain attention, MHA/GQA/MQA; part 3a: KV-cache GiB figures, MoE sparse equals dense, speculative greedy equals target greedy; part 3b: NF4 levels, int8 round-trip error, error ordering of schemes, outliers vs group scales, 4-bit memory, QLoRA layer starts as the 4-bit base).
- `notebooks/`: one per episode, built by `tools/make_nb*.py`.

Note: `tlib.model.Attention` also handles several new tokens on top of a cache (needed for speculative decoding), which the part 1 video's simplified version does not.
