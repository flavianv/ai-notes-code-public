# Every number in the A4 script, checked against the saved results (results/a4/systems_s*.json, from tools/run_a4.py).
import json, pathlib, pytest

R = pathlib.Path(__file__).parent.parent / "results"


def res():
    files = [R / "a4" / f"systems_s{s}.json" for s in (0, 1, 2)]
    if not all(f.exists() for f in files): pytest.skip("run tools/run_a4.py <seed> first")
    return [json.load(open(f)) for f in files]


rng = lambda vals: (round(min(vals) * 100), round(max(vals) * 100))
k64 = lambda r, n: r["sampling"][str(n)]["N"][-1]


def test_sampling_at_50():
    rs = res()
    assert rng([r["sampling"]["50"]["single"] for r in rs]) == (75, 88)            # one attempt: 75 to 88
    assert rng([k64(r, 50)["checker"] for r in rs]) == (99, 100)                   # checker: 99 to 100
    assert rng([k64(r, 50)["vote"] for r in rs]) == (78, 94)                       # vote: 78 to 94
    assert rng([k64(r, 50)["likelihood"] for r in rs]) == (82, 94)                 # likelihood: 82 to 94


def test_sampling_at_100_nothing_to_rescue():
    for r in res():
        s = r["sampling"]["100"]
        assert abs(s["N"][-1]["checker"] - s["single"]) < 1e-6                              # checker never improves on the first sample
    rs = res()
    assert rng([r["sampling"]["100"]["single"] for r in rs]) == (29, 34)
    assert rng([k64(r, 100)["coverage"] for r in rs]) == (60, 77)                   # "at least one right final state": mostly luck


def test_search_harness_workers():
    rs = res()
    for r in rs:
        assert r["search"]["200"]["success"] == 1.0
        assert r["harness"]["200"]["final"] == 1.0 and r["harness"]["1000"]["final"] == 1.0
        w = r["workers"]; assert w["tables"] == 1.0 and w["sequential_chunks"] == 1.0
        assert w["model_calls_sequential"] * 3 == w["model_calls_tables"] and (w["serial_rounds_tables"], w["serial_rounds_sequential"]) == (1, 10)
    assert [round(min(v), 1) for v in [[r["search"]["200"]["checker_calls_per_step"] for r in rs]]] == [1.3]
    assert max(r["search"]["200"]["checker_calls_per_step"] for r in rs) < 1.42
    assert rng([r["search"]["200"]["model_top1_per_step"] for r in rs]) == (68, 77)
    assert rng([r["workers"]["final_only"] for r in rs]) == (28, 38)


def test_reliability():
    for r in res():
        assert r["reliability"] == {"steps": 20000, "errors": 0, "trace_ok": 1.0}
    a3 = [json.load(open(R / "a3" / f"interleaved_s{s}.json")) for s in (0, 1, 2)]
    fails = [1 - r["lengths"]["50"]["trace"] for r in a3]
    assert rng(fails) == (5, 16)                                                     # "between 5 and 16 percent"
