# A5: the two-stack tape, the increment machine, and the exhaustive check of a learned controller.
import random
from atlib.tape import TwoStacks, run_tm, table_controller, increment, train_controller, exhaustive_check, INCREMENT


def test_two_stacks_move_and_write():
    t = TwoStacks("abcde"); t.move("R"); t.move("R")
    assert t.read() == "c" and t.left_top() == "b"
    t.write("X"); t.move("L"); assert t.read() == "b"; t.move("R"); assert t.read() == "X"
    assert t.tape() == "abXde"


def test_table_machine_every_input_up_to_12_digits():
    assert len(INCREMENT) == 6                                            # "a hand-written table with 6 rows"
    for n in range(1, 13):
        for v in range(2 ** n):
            s = format(v, f"0{n}b"); out, _ = run_tm(s, table_controller)
            assert int(out, 2) == int(s, 2) + 1


def test_table_machine_long_random_inputs():
    rng = random.Random(0)
    for _ in range(50):
        s = "1" + "".join(rng.choice("01") for _ in range(rng.randint(100, 1000)))
        assert run_tm(s, table_controller)[0] == increment(s)


def test_learned_controller_is_exact_everywhere():
    ctrl, n_train = train_controller()
    ok, total = exhaustive_check(ctrl)
    assert (ok, total) == (18, 18)
    rng = random.Random(1)
    s = "1" + "".join(rng.choice("01") for _ in range(999))
    assert run_tm(s, ctrl)[0] == increment(s)                             # 1,000 digits, never seen in training
