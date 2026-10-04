# Every number in the A3 script, checked against the saved results (results/a3/*.json, produced by tools/run_a3.py).
import json, pathlib, pytest

R = pathlib.Path(__file__).parent.parent / "results" / "a3"
SEEDS = (0, 1, 2)


def res(fmt):
    files = [R / f"{fmt}_s{s}.json" for s in SEEDS]
    if not all(f.exists() for f in files): pytest.skip(f"run tools/run_a3.py {fmt} <seed> first")
    return [json.load(open(f)) for f in files]


def final(fmt, n): return [r["lengths"][str(n)]["final"] for r in res(fmt)]
def first_err(fmt, n): return [r["lengths"][str(n)]["first_error_median"] for r in res(fmt)]
def heavy(fmt): return [r["i_heavy"]["20"]["final"] for r in res(fmt)]


def test_three_setups_perfect_in_training_range():
    for fmt in ("after", "interleaved", "looped"):
        for n in (10, 20):
            assert min(final(fmt, n)) >= 0.995, (fmt, n)


def test_direct_does_not_learn_even_in_range():
    assert 0.44 <= min(final("direct", 10)) and max(final("direct", 10)) <= 0.68     # "between 45 and 67 percent"
    assert all(0.35 <= a <= 0.37 for a in final("direct", 20))                       # "36 percent on every seed"


def test_longer_sequences():
    assert all(0.31 <= a <= 0.35 for a in final("after", 50))                         # trace after: chance at 50
    assert 0.84 <= min(final("interleaved", 50)) and max(final("interleaved", 50)) <= 0.96   # "85 to 95 percent"
    assert sorted(round(a, 2) for a in final("looped", 50)) == [0.33, 0.35, 0.61]     # one seed 61%, others chance
    for fmt in ("direct", "after", "interleaved", "looped"):
        for n in (100, 200):
            assert all(0.28 <= a <= 0.40 for a in final(fmt, n)), (fmt, n)           # everyone at chance


def test_where_traces_break():
    assert all(2 <= e <= 4 for e in first_err("after", 50))                           # trace after: wrong from step 2-4
    assert all(42 <= e <= 46 for e in first_err("interleaved", 50))
    assert sorted(first_err("interleaved", 100)) == [53, 54, 57]
    assert first_err("interleaved", 100) == first_err("interleaved", 200)             # same break point at 100 and 200 actions


def test_identity_heavy_shift():
    assert min(heavy("interleaved")) == 1.0
    assert 0.84 <= min(heavy("after")) and max(heavy("after")) <= 0.98
    assert 0.81 <= min(heavy("looped")) and max(heavy("looped")) <= 0.93


def test_follow_up_position_is_not_the_cause():
    assert 0.62 <= min(final("interleaved-offset", 50)) and max(final("interleaved-offset", 50)) <= 0.87   # "63 to 86 percent"
    assert all(45 <= e <= 56 for e in first_err("interleaved-offset", 100))
    assert all(a <= 0.36 for a in final("interleaved-nope", 50))                                    # no positions: chance at 50
    assert all(34 <= e <= 37 for e in first_err("interleaved-nope", 50))


def test_follow_up_window_runs_at_ten_times_training_length():
    for r in res("interleaved-window"):
        for n in ("50", "100", "200"):
            assert r["lengths"][n]["final"] == 1.0 and r["lengths"][n]["trace"] == 1.0                 # every trace right, every step
        assert r["i_heavy"]["100"]["final"] == 1.0


def test_offset_breaks_where_the_original_did():
    assert sorted(first_err("interleaved-offset", 100)) == [53, 53, 56]                         # "step 53 to 56, the same place as before"
