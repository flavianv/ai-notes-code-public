# Every number in the A1 script, checked against the saved results (results/a1/*.json, from tools/run_a1.py).
import json, pathlib, pytest

R = pathlib.Path(__file__).parent.parent / "results" / "a1"


def ind():
    files = [R / f"induction_s{s}.json" for s in (0, 1, 2)]
    if not all(f.exists() for f in files): pytest.skip("run tools/run_a1.py induction <seed> first")
    return [json.load(open(f)) for f in files]


def sup():
    if not (R / "superposition.json").exists(): pytest.skip("run tools/run_a1.py superposition first")
    return json.load(open(R / "superposition.json"))


def test_one_layer_fails_two_layers_succeed():
    rs = ind()
    assert all(0.05 <= r["L1"]["acc"]["20"] <= 0.105 for r in rs)                     # "6 to 10 percent"
    assert all(0.9965 <= r["L2"]["acc"]["20"] <= 0.9985 for r in rs)                  # "99.7 to 99.8 percent"
    assert all(3.9 <= r["L2"]["loss_halves"][0] <= 4.1 for r in rs)                   # random half sits at about 4.0


def test_sudden_drop_between_300_and_600():
    for r in ind():
        c = {step: rep for step, _, rep in r["L2"]["curve"]}
        assert c[300] > 3.0 and c[600] < 0.7


def test_heads_found():
    rs = ind()
    n_prev = sorted(len(r["L2"]["group_ablate"]["prev_heads"]) for r in rs)
    assert n_prev == [1, 2, 2]                                                          # one run: one head; two runs: two heads
    assert max(rs[0]["L2"]["prev"].values()) >= 0.955                                   # "96 percent of its attention"
    for r in rs:
        scores = [v for k, v in r["L2"]["ind"].items() if k.startswith("1.")]
        assert len(scores) == 4 and all(0.865 <= v <= 0.925 for v in scores)            # all four, 87 to 92 percent


def test_ablations():
    for r in ind():
        a, g = r["L2"]["ablate"], r["L2"]["group_ablate"]
        assert 0.975 <= a["acc_ind_off"] <= 0.99                                       # one induction head off: ~98%
        assert 0.045 <= g["all_prev_off"] <= 0.085                                     # "between 5 and 8 percent"
        assert 0.01 <= g["all_ind_off"] <= 0.03                                        # "2 percent"
        assert 0.55 <= g["one_ind_kept"] <= 0.665                                      # "between 56 and 66"
        assert 0.79 <= g["non_prev_L0_off"] <= 0.84                                    # "about 80"


def test_longer_repeats_break():
    accs = [r["L2"]["acc"]["50"] for r in ind()]
    assert 0.135 <= min(accs) and max(accs) <= 0.285                                   # "between 14 and 28 percent"


def test_superposition_and_sae():
    d = sup()
    assert [d["toy"][k]["represented"] for k in ("0.0", "0.5", "0.7", "0.9", "0.97")] == [2, 4, 4, 5, 5]
    rec = sorted(d["sae"]["recovery"])
    assert 0.775 <= rec[0] <= 0.785 and all(v >= 0.93 for v in rec[1:]) and rec[-1] >= 0.9995
