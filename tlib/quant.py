# Part 3b: making models small. Quantization from scratch (int8, int4, NF4) and a QLoRA-style layer: frozen 4-bit base + trainable low-rank adapters.
import math
import torch, torch.nn as nn

# the 16 NormalFloat4 levels from the QLoRA paper: quantiles of a standard normal, rescaled to [-1, 1], with an exact zero
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141033173, 0.24611230194950104, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


def quant_absmax(w, bits=8, dim=None):
    """Symmetric integers: scale = max|w| / (2^(bits-1) - 1). dim=None: per tensor, dim=1: per row."""
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().max() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    scale = amax.clamp(min=1e-8) / qmax
    return (w / scale).round().clamp(-qmax, qmax).to(torch.int8), scale


def dequant_absmax(q, scale): return q.float() * scale


def quant_group(w, bits=4, group=64):
    """Group-wise: every `group` weights share one scale (an outlier only hurts its group)."""
    flat = w.reshape(-1, group); q, scale = quant_absmax(flat, bits, dim=1)
    return q, scale


def dequant_group(q, scale, shape): return (q.float() * scale).reshape(shape)


def quant_nf4(w, group=64):
    """NF4: scale each group to [-1, 1], then snap to the nearest of 16 normal-quantile levels."""
    flat = w.reshape(-1, group); amax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    idx = (flat / amax)[..., None].sub(NF4.to(w.device)).abs().argmin(-1)           # (n_groups, group): index 0..15
    return idx.to(torch.uint8), amax


def dequant_nf4(idx, amax, shape): return (NF4.to(idx.device)[idx.long()] * amax).reshape(shape)


def fake_quant(w, scheme):
    """Quantize then dequantize: the weights the model computes with. E.g. 'int8-channel', 'nf4-g64'."""
    kind, _, opt = scheme.partition("-")
    bits = 8 if kind == "int8" else 4
    if kind in ("int8", "int4") and opt == "tensor": return dequant_absmax(*quant_absmax(w, bits))
    if kind in ("int8", "int4") and opt == "channel": return dequant_absmax(*quant_absmax(w, bits, dim=1))
    if kind == "int4" and opt.startswith("g"): return dequant_group(*quant_group(w, 4, int(opt[1:])), w.shape)
    if kind == "nf4": return dequant_nf4(*quant_nf4(w, int(opt[1:])), w.shape)
    raise ValueError(scheme)


def scheme_bytes(numel, scheme, scale_bytes=2):
    """Storage for a weight matrix: packed codes plus fp16 scales."""
    kind, _, opt = scheme.partition("-")
    if kind == "fp32": return numel * 4
    if kind in ("fp16", "bf16"): return numel * 2
    bits = 8 if kind == "int8" else 4
    if opt in ("tensor", "channel"): return numel * bits // 8 + (scale_bytes if opt == "tensor" else 0)          # per-channel scales: add rows x scale_bytes at the call site
    return numel * bits // 8 + numel // int(opt[1:]) * scale_bytes


class QLoRALinear(nn.Module):
    """y = x W^T + (alpha / r) x A^T B^T. W is frozen, stored in NF4; only A and B are trained."""
    def __init__(self, linear, r=8, alpha=16, group=64):
        super().__init__()
        self.shape, self.group, self.scaling = linear.weight.shape, group, alpha / r
        idx, amax = quant_nf4(linear.weight.data, group)
        self.register_buffer("idx", idx); self.register_buffer("amax", amax)                                    # frozen 4-bit base weights
        self.register_buffer("bias", None if linear.bias is None else linear.bias.data.clone())
        out_f, in_f = self.shape
        self.A = nn.Parameter(torch.randn(r, in_f) / math.sqrt(in_f))                                           # trainable
        self.B = nn.Parameter(torch.zeros(out_f, r))                                                            # zero: the model starts exactly as the 4-bit base

    def forward(self, x):
        w = dequant_nf4(self.idx, self.amax, self.shape).to(x.dtype)                                            # dequantize on the fly, compute in higher precision
        return nn.functional.linear(x, w, self.bias) + self.scaling * ((x @ self.A.t()) @ self.B.t())


def add_qlora(model, r=8, alpha=16, group=64, skip=("head",)):
    """Replace every nn.Linear in the blocks by a QLoRALinear; freeze everything else."""
    for p in model.parameters(): p.requires_grad_(False)
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname not in skip:
                setattr(mod, cname, QLoRALinear(child, r, alpha, group))
    return model
