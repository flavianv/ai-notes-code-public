# Part 3b: quantization from scratch and the QLoRA-style layer.
import torch
from tlib.quant import NF4, quant_absmax, dequant_absmax, fake_quant, scheme_bytes, QLoRALinear, add_qlora
from tlib.model import GPT

torch.manual_seed(0)


def mse(w, scheme): return (fake_quant(w, scheme) - w).pow(2).mean().item()


def test_nf4_levels_are_sorted_with_exact_zero_and_unit_range():
    assert torch.equal(NF4, NF4.sort().values) and NF4[7] == 0 and NF4[0] == -1 and NF4[-1] == 1 and len(NF4) == 16


def test_int8_roundtrip_error_is_half_a_step():
    w = torch.randn(64, 128); q, s = quant_absmax(w, 8, dim=1)
    assert (dequant_absmax(q, s) - w).abs().max() <= s.max() / 2 + 1e-6


def test_error_ordering_on_gaussian_weights():
    w = torch.randn(512, 512)
    assert mse(w, "int8-channel") < mse(w, "nf4-g64") < mse(w, "int4-g64") < mse(w, "int4-tensor")


def test_an_outlier_ruins_per_tensor_scaling_but_not_group_scaling():
    w = torch.randn(256, 256); w[0, 0] = 50.0
    assert mse(w, "int4-tensor") > 20 * mse(w, "int4-g64")


def test_memory_of_4bit_weights():
    n = 1_000_000
    assert scheme_bytes(n, "fp16") == 2 * n and scheme_bytes(n, "int8-channel") == n and scheme_bytes(n, "nf4-g64") == n // 2 + n // 64 * 2


def test_qlora_layer_starts_as_the_4bit_base_and_trains_only_adapters():
    lin = torch.nn.Linear(64, 32); q = QLoRALinear(lin, r=4); x = torch.randn(5, 64)
    base = torch.nn.functional.linear(x, fake_quant(lin.weight.data, "nf4-g64"), lin.bias)
    assert (q(x) - base).abs().max() < 1e-5
    assert {n for n, p in q.named_parameters() if p.requires_grad} == {"A", "B"}


def test_add_qlora_freezes_the_rest_of_the_model():
    m = add_qlora(GPT(30, 64, 4, 2), r=4)
    names = {n.split(".")[-1] for n, p in m.named_parameters() if p.requires_grad}
    assert names == {"A", "B"} and m(torch.randint(0, 30, (1, 8))).shape == (1, 8, 30)
