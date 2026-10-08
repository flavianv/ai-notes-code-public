# Pins the numbers shown in the eight reward hacking films, and checks that a fresh run of every lab reproduces the saved results.
import json, math, pathlib, shutil, subprocess, sys
import pytest

HERE = pathlib.Path(__file__).parent
FILMS = ["1_intro", "2_actual_success", "3_criteria_first", "4_hidden_tests", "5_uncertain_scores", "6_attempted_cheats", "7_kl_budget", "8_secure_env"]
load = lambda p: json.loads((HERE / p).read_text())


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """Run every lab in a scratch copy (shared labs first, the film labs read their results)."""
    root = tmp_path_factory.mktemp("rh")
    for d in ["shared"] + FILMS:
        shutil.copytree(HERE / d, root / d)
        for f in (root / d).glob("results*.json"):
            f.unlink()
    for d, script in [("shared", "lab_fixes.py"), ("shared", "lab_pressure.py")] + [(f, "lab.py") for f in FILMS]:
        subprocess.run([sys.executable, script], cwd=root / d, check=True, capture_output=True)
    return root


@pytest.mark.parametrize("path", ["shared/results_fixes.json", "shared/results_pressure.json"] + [f"{f}/results.json" for f in FILMS])
def test_saved_results_match_a_fresh_run(fresh, path):
    assert json.loads((fresh / path).read_text()) == load(path)


def test_1_intro_policy_gradient():        # hacks take over without a check; scoring them 0 restores honest work
    K = load("1_intro/results.json")
    assert K["no_check"]["final"][0] < .01 and round(K["no_check"]["quality"], 2) == 0
    assert K["fixed_reward"]["final"][0] > .999 and round(K["fixed_reward"]["quality"], 2) == 1


def test_2_actual_success():               # appearance grader picks fakes; a sound checker never accepts a wrong program
    K = load("2_actual_success/results.json"); A, C, F = K["appearance"], K["checker"], K["checker_formula"]
    assert round(A[4]["fake"], 2) == .63 and round(A[4]["correct"], 2) == .30 and round(max(a["correct"] for a in A), 2) == .35
    assert all(c["false_accept"] == 0 for c in C) and [round(f, 2) for f in F[1:4]] == [.59, .97, 1.0]
    assert K["union"]["bound"] == .01 and round(K["union"]["exact_independent"] * 100, 3) == .995


def test_3_criteria_first():               # 1 - (1 - r)^m = 0.65; share of accepted that is wrong: 19% fixed vs 60% after
    K = load("3_criteria_first/results.json")
    assert round(K["pass_one"]["adaptive_formula"], 2) == .65
    assert round(K["share_formula"]["fixed"], 2) == .19 and round(K["share_formula"]["adaptive"], 2) == .60
    assert round(K["adaptive"][2]["wrong_accepted"], 2) == .60 and round(K["fixed"][2]["wrong_accepted"], 2) == .19
    assert [round(K["by_m"][m]["wrong_accepted"], 2) for m in ("1", "3", "10")] == [.19, .39, .60]


def test_4_hidden_tests():                 # (1 - p)^n, 90 tests for 1%, commit first, pool wear-out
    K = load("4_hidden_tests/results.json"); n = K["survive"]["n"]
    assert K["survive"]["n_for_1pct_exact"] == 90 and K["survive"]["n_for_1pct"] == 93
    assert all(abs(n[k]["simulated"] - n[k]["formula"]) < .01 for k in n)
    assert K["payoff"]["n_needed_p5"] == 4 and round(K["payoff"]["q_at_4"], 2) == .19
    assert K["commit"]["visible_reward"] == 1 and round(K["commit"]["visible_true_acc"], 2) == .11 and round(K["commit"]["commit_reward"], 2) == .10
    W = K["wear"]; assert W["fixed_pool_100"]["reward_1000"] == 1 and round(W["rotated_200"]["reward_1000"], 2) == .39 and round(W["rotated_200"]["holdout_end"], 1) == .1


def test_5_uncertain_scores():             # optimizer's curse, mean - lambda * spread, shared errors
    K = load("5_uncertain_scores/results.json")
    assert round(K["curse"]["256"], 2) == 2.83 and round(K["single"]["exploit"], 2) == .52 and round(K["mean5"]["exploit"], 2) == .31
    assert K["lam"]["1.5"]["exploit"] < .03 and round(K["lam"]["1.5"]["quality"], 2) == .77
    assert round(K["rho"]["0.6"]["exploit"], 2) == .12 and round(K["rho"]["1.0"]["exploit"], 2) == .52
    assert round(1.5 * math.sqrt(.7), 2) == 1.25


def test_6_attempted_cheats():             # the audit loop: 94% -> below 0.1% in seven rounds; more pressure exposes hole E
    K = load("6_attempted_cheats/results.json"); L, H = K["loop_N256"], K["after_N4096"]
    assert round(L[0]["hack_rate"], 2) == .94 and L[7]["hack_rate"] < .001 and round(L[7]["quality"], 2) == .92
    assert round(K["surface"]["pA_N256"], 2) == .92 and round(K["rounds"]["D_find"], 2) == .21 and round(K["rounds"]["E256_expected_rounds"], -1) == 390
    assert H[4]["patched"] == ["A", "B", "C", "D", "E"] and H[4]["hack_rate"] == 0


def test_7_kl_budget():                    # coin KL 0.60, the split identity, budgets, tilt, tails, canaries
    K = load("7_kl_budget/results.json")
    assert round(K["coins"]["exploit_term"], 2) == .69 and round(K["coins"]["honest_term"], 2) == -.09 and round(K["coins"]["total"], 1) == .6
    assert abs(K["split"]["shift"] - K["split"]["scale"] - .1 * math.log(2)) < 1e-9
    assert round(K["budget"]["0.01"], 2) == .14 and round(K["budget"]["1e-06"], 2) == 1.06
    T = K["tilt"]; mh = dict(zip(T["beta"], T["measured_hack"]))
    assert round(T["beta_star"], 2) == .12 and [round(mh[b], 2) for b in (.5, .2, .12)] == [.06, .22, .53]
    assert round(K["tails"]["light"][6], 2) == 2.58 and round(K["tails"]["heavy"][6], 2) == .02
    assert K["canary_training"]["canary_10pct_step"] == 38


def test_8_secure_env():                   # feedback leakage, spot checks, telescoping shaping, learned world model
    K = load("8_secure_env/results.json"); L = K["leakage"]
    assert L["fixed_per_test"]["reward_last200"] == 1 and round(L["fixed_per_test"]["true_accuracy_end"], 2) == .11
    assert round(L["fresh_per_test"]["reward_last200"], 2) == .53 and round(L["fresh_per_test"]["true_accuracy_end"], 2) == .54
    assert [round(K["spot"][k]["analytic"], 2) for k in ("5", "20")] == [.59, .12]
    assert K["shaping"]["sum"] == K["shaping"]["end_minus_start"] == 4
    assert [round(x, 1) for x in K["replan"]] == [1.5, 6.3, 7.9]
