# Fighting reward hacking

Code for the AI Notes series *Fighting reward hacking*: every toy number shown in the eight films is computed here. The toys are small synthetic simulations (NumPy, fixed seeds). No language model is trained; numbers from published studies are cited in the films, not reproduced.

```bash
cd reward-hacking/shared && python3 lab_fixes.py && python3 lab_pressure.py   # about 45 s
cd ../4_hidden_tests && python3 lab.py                                        # any film: run from its folder
cd ../.. && pytest reward-hacking                                             # re-runs every lab and pins the on-screen numbers (about a minute)
```

Each lab writes its results next to itself. Films 1, 7 and 8 also read the shared results, so run the two shared labs first.

| # | Film | Folder | What the lab computes |
|---|---|---|---|
| 1 | What reward hacking is | `1_intro/` | a three-action policy-gradient toy: hacks scoring 1.2 take over (quality 0.00); scoring them 0 restores honest work (quality 1.00) |
| 2 | Checking actual success | `2_actual_success/` | best of N with an appearance grader (confident fakes reach 63% at N = 256) vs a sound outcome checker (0 wrong programs accepted; correct = 1 − 0.8^N); the union bound Tδ |
| 3 | Setting criteria in advance | `3_criteria_first/` | criteria chosen after the answer pass a wrong answer with 1 − (1 − r)^m = 0.65; share of accepted answers that are wrong: 19% fixed vs 60% after |
| 4 | Hidden challenges | `4_hidden_tests/` | survival (1 − p)^n, simulated; 90 tests for 1% at p = 5%; cheating stops paying at q > 1/6; commit-first and pool wear-out toys |
| 5 | Discounting uncertain scores | `5_uncertain_scores/` | optimizer's curse; an ensemble scored by mean − λ·spread (cheats 52% → 2.8%); λ sweep; shared-error sweep |
| 6 | Training on attempted cheats | `6_attempted_cheats/` | the audit loop over five holes (cheating 94% → below 0.1% in seven rounds); a rare hole exposed at higher pressure |
| 7 | A KL budget | `7_kl_budget/` | coin KL 0.69 − 0.09 ≈ 0.60, the exact split of the KL, budgets for 10%; tilt crossing β* ≈ 0.12, light vs heavy tails, canaries |
| 8 | Secure RL environments | `8_secure_env/` | feedback leakage, spot checks (1 − ε)^k, telescoping shaping, decoupled-approval arithmetic, a learned world model with replanning |

`shared/lab_fixes.py`: best of N with exploits, pessimistic ensembles, the audit loop, monitored training, and a learned world model (replan, stay near the data).
`shared/lab_pressure.py`: the tilt model and measured exploit rates, light vs heavy tails, canaries, feedback leakage, spot checks and test wear-out.
