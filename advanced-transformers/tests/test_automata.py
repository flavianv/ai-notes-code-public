# Understanding Transformers, A3: the 3-state machine and its encodings.
import random
from atlib.automata import (run, summarize, compose, encode, TRANSITIONS, IDENTITY, random_actions,
                            turnstile, in_string, tree_rounds, table_rounds)


def test_worked_example_and_order_matters():
    assert run("ABAIB") == (0, [0, 1, 0, 1, 1, 0])
    assert run("AB")[0] == 0 and run("BA")[0] == 2                      # same histogram, different answer


def test_balanced_composition_matches_sequential():
    rng = random.Random(0)
    for _ in range(2000):
        s = random_actions(rng.randint(0, 300), rng)
        assert summarize(s)[0] == run(s)[0]
        assert summarize(s) == tuple(run(s, start=q)[0] for q in range(3))   # the summary is right from every start state


def test_identity_insertion_preserves_answer():
    rng = random.Random(1)
    for _ in range(500):
        s = random_actions(30, rng); i = rng.randint(0, 30)
        assert run(s[:i] + "I" + s[i:])[0] == run(s)[0]


def test_composition_is_associative():
    T = list(TRANSITIONS.values())
    for a in T:
        for b in T:
            for c in T:
                assert compose(a, compose(b, c)) == compose(compose(a, b), c)
    assert compose(TRANSITIONS["I"], TRANSITIONS["A"]) == TRANSITIONS["A"] and TRANSITIONS["I"] == IDENTITY


def test_encodings_mark_the_state_tokens():
    assert encode("AB", "direct") == ([7, 0, 1, 3, 4], [False, False, False, True])
    assert encode("AB", "after") == ([7, 0, 1, 3, 5, 4], [False, False, False, True, True])
    assert encode("AB", "interleaved") == ([7, 0, 5, 1, 4], [False, True, False, True])


def test_everyday_machines():
    assert turnstile(["coin", "push", "push"]) == "locked"
    assert turnstile(["push", "coin", "coin"]) == "unlocked"
    line = 'say("hi") # "x"'
    assert "".join("i" if x else "o" for x in in_string(line)) == "ooooiiioooooiio"


def test_tree_sum_rounds():
    r = tree_rounds([3, 1, 4, 1, 5, 9, 2, 6], lambda a, b: a + b)
    assert r == [[3, 1, 4, 1, 5, 9, 2, 6], [4, 5, 14, 8], [9, 22], [31]]
    assert len(tree_rounds(list(range(200)), lambda a, b: a + b)) - 1 == 8   # 200 numbers: 8 rounds


def test_tree_of_tables_film_example():
    assert compose(TRANSITIONS["B"], TRANSITIONS["A"]) == (0, 2, 1)       # "A then B"
    r = table_rounds("ABAIBAAB")
    assert r[1] == [(0, 2, 1), (1, 2, 0), (2, 1, 0), (0, 2, 1)]
    assert r[2] == [(1, 0, 2), (1, 2, 0)]
    assert r[3] == [(2, 1, 0)] and r[3][0][0] == run("ABAIBAAB")[0] == 2
    assert len(table_rounds("A" * 200)) - 1 == 8


def test_position_variants_cache_matches_full_pass():
    import torch
    from atlib.automata import make_model
    torch.manual_seed(0)
    ids = torch.randint(0, 8, (2, 12))
    for fmt in ("interleaved", "interleaved-nope", "interleaved-window"):
        m = make_model(fmt).eval()
        with torch.no_grad():
            full = m(ids); caches = [{} for _ in m.blocks]
            steps = torch.cat([m(ids[:, t:t + 1], t, caches) for t in range(12)], 1)
        assert (full - steps).abs().max() < 1e-4
    m = make_model("interleaved-nope").eval()                            # no position signal: shifting the start changes nothing
    with torch.no_grad():
        assert (m(ids, 0) - m(ids, 300)).abs().max() < 1e-5


def test_window_attention_ignores_far_tokens():
    import torch
    from atlib.automata import make_model
    torch.manual_seed(0)
    m = make_model("interleaved-window").eval()
    a = torch.randint(0, 8, (1, 40)); b = a.clone(); b[0, :10] = (b[0, :10] + 1) % 8   # change only tokens far from the end
    with torch.no_grad():
        assert (m(a)[:, -1] - m(b)[:, -1]).abs().max() < 1e-5                 # 4 layers x 7 tokens back = 28: positions 0-9 are out of reach of position 39
