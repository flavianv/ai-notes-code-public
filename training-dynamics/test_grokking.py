# Pins the grokking numbers shown in film 3 against results/grokking.json (re-run grokking.py to regenerate: a few minutes on a CPU).
import json, pathlib
G = json.loads((pathlib.Path(__file__).parent / "results" / "grokking.json").read_text())
runs = lambda c: sorted((r for r in G["runs"] if r["cond"] == c), key=lambda r: r["seed"])


def test_grokking_memorizes_then_generalizes():
    g = runs("grok")
    assert [r["train_99"] for r in g] == [100, 100, 100] and [r["test_99"] for r in g] == [11600, 10600, 12000]
    assert all(round(r["log"][1][3]) in (113, 114) and round(r["log"][-1][3]) in (62, 63, 64) for r in g)   # weight norm falls


def test_no_weight_decay_never_generalizes():
    for r in runs("no_decay"):
        assert r["train_99"] == 100 and r["test_99"] is None and r["final_test"] == 0.0
        assert max(e[2] for e in r["log"] if e[0] >= 1000) < 0.001 and r["log"][-1][3] > 280


def test_random_labels_are_memorized_test_stays_at_chance():
    for r in runs("random"):
        assert r["final_train"] == 1.0 and 0.009 < r["final_test"] < 0.012   # chance is 1/97 ≈ 0.0103
