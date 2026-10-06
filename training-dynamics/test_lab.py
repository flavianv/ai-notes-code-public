# Pins every lab number shown in the Transformer Training Dynamics film, and checks the saved results match a fresh run.
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lab import run

R = run()
SAVED = json.loads((pathlib.Path(__file__).parent / "results" / "lab-results.json").read_text())


def test_saved_results_match_a_fresh_run():
    assert json.dumps(R, sort_keys=True) == json.dumps(SAVED, sort_keys=True)


def test_score_variance_table():          # chapter 08: raw ≈ d, ÷√d ≈ 1, ÷d shrinks
    V = {r["d"]: r for r in R["variance"]}
    assert round(V[64]["raw"], 1) == 63.6 and round(V[64]["scaled"], 2) == 0.99
    assert round(V[256]["raw"]) == 259 and round(V[256]["scaled"], 2) == 1.01
    assert round(V[1024]["divided_by_d"], 3) == 0.001


def test_attention_gradients_match_finite_differences():   # chapter 12: worst error 2e-10
    errs = R["gradient_max_abs_error"]
    assert 1.5e-10 < max(errs.values()) < 2.5e-10


def test_gauge_and_permutation():          # chapters 07 and 27
    assert R["gauge_max_abs_error"] < 5e-16 and R["permutation_max_abs_error"] < 1e-16


def test_routing_step():                   # chapter 13: 0.5/0.5 -> 0.475/0.525, loss 0.5 -> 0.451
    r = R["routing"]
    assert r["gradient"] == [0.5, -0.5] and [round(a, 3) for a in r["attention_after"]] == [0.475, 0.525]
    assert r["loss_before"] == 0.5 and round(r["loss_after"], 3) == 0.451


def test_spectral_budget():                # chapter 20: same spectral budget, SGD -1.010 vs ideal Muon -1.101
    s = R["spectral_budget"]
    assert [round(v, 6) for v in s["sgd_singular_values"]] == [0.01, 0.001, 0.00001]
    assert [round(v, 6) for v in s["ideal_muon_singular_values"]] == [0.01, 0.01, 0.01]
    assert round(s["linear_change_sgd"], 3) == -1.010 and round(s["linear_change_ideal_muon"], 3) == -1.101


E = R["extras"]


def test_extras_are_finite():
    flat = json.dumps(E); assert "NaN" not in flat and "Infinity" not in flat


def test_film1_signals():                  # film 1: moments, sums, init, residual variance
    assert round(E["moments"]["var_3z"], 2) == 9.0 and round(E["sums"]["std_independent"]) == 10
    assert round(E["init"]["forward_var"], 1) == 1.0 and round(E["init"]["backward_var"], 1) == 4.0
    assert [round(E["residual"][f"{L}:alpha_1"]) for L in (12, 24, 48)] == [13, 25, 49]
    assert all(round(E["residual"][f"{L}:alpha_inv_sqrt_L"]) == 2 for L in (12, 24, 48))


def test_film1_blocks():                   # film 1: MLP, activations, QK-norm
    m = E["mlp"]; assert m["mlp_share"] == 2 / 3 and m["kv_error"] < 1e-14 and round(m["active_fraction"], 2) == 0.50 and m["dead_value_grad_max"] == 0
    a = E["activations"]; assert round(a["gelu_prime_m1"], 2) == -0.08 and round(a["silu_prime_m1"], 2) == 0.07 and a["swiglu_weights"] == a["classic_weights"]
    q = E["qk_norm"]; assert round(q["raw_max_grown"] / q["raw_max"]) == 100 and abs(q["qknorm_max_grown"] - q["qknorm_max"]) < 1e-9 and q["qknorm_max"] <= q["g"]


def test_film2_step():                     # film 2: objective example, RMSProp, schedules, batch, QK-Clip
    o = E["objective"]; assert round(o["loss"], 2) == 0.44 and round(o["grad"][0], 2) == -0.36
    r = E["rmsprop_first_step"]; assert round(r["rmsprop"], 1) == 31.6 and round(r["adam_corrected"], 6) == 1.0
    s = E["schedules"]; assert round(s["final_loss_constant"], 3) == 0.060 and round(s["final_loss_wsd"], 3) == 0.009
    assert s["steps_to_target_by_batch"]["1"] == 3000 and s["steps_to_target_by_batch"]["1024"] == 10
    assert all(abs(v - 100) < 1e-9 for v in E["qk_clip"]["s_max_after"])


def test_film3():                          # film 3: implicit bias, delta rule
    b = E["implicit_bias"]; assert b["residual"] < 1e-12 and b["dist_to_pinv"] < 1e-12 and b["norm_gd"] < b["norm_other_fit"]
    assert all(v["delta"] < v["additive"] for v in E["delta_rule"].values())
