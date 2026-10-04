# A4: exact facts and the harness logic (no trained model needed).
import random
from atlib.automata import run, compose, TRANSITIONS, random_actions
from atlib.systems import PREFIX, trace_ok


def test_prefixes_reach_each_start_state():
    assert {s: run(p)[0] for s, p in PREFIX.items()} == {0: 0, 1: 1, 2: 2}


def test_exact_checker():
    s = "ABAIB"; good = run(s)[1][1:]
    assert trace_ok(s, good) and not trace_ok(s, good[:-1] + [(good[-1] + 1) % 3])


def test_chunk_tables_compose_to_the_answer():
    rng = random.Random(0)
    for _ in range(200):
        s = random_actions(200, rng); tab = (0, 1, 2)
        for k in range(10):
            ch = s[k * 20:(k + 1) * 20]
            tab = compose(tuple(run(PREFIX[q] + ch)[0] for q in range(3)), tab)   # a worker's table: run from each start state
        assert tab[0] == run(s)[0]


def test_final_state_only_loses_information():
    assert run("I")[0] == run("AB")[0] == 0                                           # same answer from 0 ...
    assert run("I", start=1)[0] == 1 and run("AB", start=1)[0] == 2                   # ... different from 1


def test_reliability_arithmetic():
    assert round(1 - 0.9 ** 50, 3) == 0.995                                          # 1 - (1 - p)^N, p = 0.1, N = 50
    assert round(0.999 ** 1000, 3) == 0.368 and 4.4e-5 < 0.999 ** 10000 < 4.6e-5


def test_sequential_chunks_with_a_perfect_model_is_exact():
    import torch
    from atlib import systems
    real = systems.roll
    systems.roll = lambda model, seqs, *a, **k: (torch.tensor([run(s)[1][1:] for s in seqs]), None)   # a perfect stand-in model
    try:
        r = systems.workers(None, n=200, chunk=20, size=30)
    finally:
        systems.roll = real
    assert r["tables"] == 1.0 and r["sequential_chunks"] == 1.0 and r["final_only"] < 0.6
