# Every number in the A5 script, checked against the saved results (results/a5/tape_s*.json, from tools/run_a5.py).
import json, pathlib, pytest

R = pathlib.Path(__file__).parent.parent / "results" / "a5"


def res():
    files = [R / f"tape_s{s}.json" for s in (0, 1, 2)]
    if not all(f.exists() for f in files): pytest.skip("run tools/run_a5.py <seed> first")
    return [json.load(open(f)) for f in files]


def test_table_and_learned_controller():
    for r in res():
        assert r["table_rows"] == 6 and r["table_1000_digits"] == 1.0                          # six rules, 1,000 digits: 100%
        c = r["controller"]
        assert (c["inputs_seen_in_training"], c["exhaustive_ok"], c["exhaustive_total"]) == (15, 18, 18)   # saw 15 of 18, all 18 right
        assert c["acc_1000_digits"] == 1.0


def test_end_to_end_collapses():
    rs = res()
    assert all(r["e2e"]["8"] == 1.0 for r in rs)
    assert 0.99 <= min(r["e2e"]["16"] for r in rs)                                               # "99 to 100 percent" at 16
    for n in ("24", "32", "48", "64"):
        assert all(r["e2e"][n] == 0.0 for r in rs)                                               # "zero ... on any run"
