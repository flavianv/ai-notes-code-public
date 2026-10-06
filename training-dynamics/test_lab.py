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
