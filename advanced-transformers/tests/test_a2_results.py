# Every number in the A2 script, checked against the saved results (results/a2/*.json, from tools/run_a2.py).
import json, pathlib, pytest
from atlib.expressivity import balanced_depth, PERMS
from atlib.automata import summarize, TRANSITIONS

R = pathlib.Path(__file__).parent.parent / "results" / "a2"


def res(what):
    files = [R / f"{what}_s{s}.json" for s in (0, 1, 2)]
    if not all(f.exists() for f in files): pytest.skip(f"run tools/run_a2.py {what} <seed> first")
    return [json.load(open(f)) for f in files]


def test_exact_facts():
    assert balanced_depth(8) == 3 and balanced_depth(32) == 5 and balanced_depth(200) == 8
    assert sorted(PERMS) == sorted({summarize(s) for s in ("", "A", "AA", "B", "AB", "BA")})     # A and B make all 6 tables


def test_depth_helps_at_8():
    d = res("depth"); last = lambda k: [r[k]["last"] for r in d]
    assert 0.48 <= min(last("n8_L1")) and max(last("n8_L1")) <= 0.615                 # "49 to 61"
    assert all(0.955 <= v <= 0.975 for v in last("n8_L2"))                            # "96 to 97"
    assert all(0.985 <= v <= 1.0 for v in last("n8_L4"))                              # "99 to 100"


def test_no_depth_works_at_32_or_128():
    d = res("depth")
    for n in (32, 128):
        for L in (1, 2, 4):
            assert all(0.295 <= r[f"n{n}_L{L}"]["last"] <= 0.355 for r in d)          # "30 to 35": chance
    assert all(0.635 <= r["n32_L4"]["all"] <= 0.685 for r in d)                       # "about two thirds" over all positions


def test_count_generalizes_state_does_not():
    g = res("general")
    assert all(r["count"]["20"]["last"] == 1.0 for r in g)
    assert 0.865 <= min(r["count"]["200"]["last"] for r in g) and max(r["count"]["200"]["last"] for r in g) <= 0.935
    assert all(0.965 <= r["state"]["10"]["last"] <= 0.985 for r in g)                 # "97 to 98"
    assert all(0.375 <= r["state"]["20"]["last"] <= 0.405 for r in g)                 # "38 to 40"


def test_probe():
    for r in res("general"):
        p = r["probe"]
        assert round(p[0]["function"], 2) == 0.22 and 0.475 <= p[-1]["function"] <= 0.495
        assert round(p[0]["state"], 2) == 0.37 and 0.805 <= p[-1]["state"] <= 0.825
        assert all(a["function"] <= b["function"] for a, b in zip(p, p[1:]))          # climbs layer by layer
