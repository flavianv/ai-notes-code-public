"""Every toy number in the secure-environments film. Results in results.json. Synthetic toys, fixed seeds, no language model trained.
Reuses the shared toys in ../shared:
  leakage and spot checks: ../shared/lab_pressure.py -> results_pressure.json
  learned world model (replan, stay near the data): ../shared/lab_fixes.py -> results_fixes.json
and adds the shaping telescoping check and the decoupled-approval arithmetic."""
import json
R = {}
P3 = json.load(open('../shared/results_pressure.json')); RF = json.load(open('../shared/results_fixes.json'))
R['leakage'] = P3['leakage']; R['spot'] = P3['spot_checks']
R['replan'] = [r['real_return'] for r in RF['replan'][:3]]; R['pessimism'] = {str(r['lambda']): r['real_return'] for r in RF['pessimism']}
# potential-based shaping: F = gamma*Phi(s') - Phi(s) with gamma = 1 telescopes to Phi(end) - Phi(start)
phi = [0, 1, 2, 3, 4]; R['shaping'] = {'phi': phi, 'sum': sum(phi[k + 1] - phi[k] for k in range(4)), 'end_minus_start': phi[-1] - phi[0]}
# decoupled approval: collect apple (true 1), push the reward counter (true 0, adds +5 to every later feedback)
R['decoupled'] = {'ordinary_push': 0 + 5, 'ordinary_collect': 1, 'asked_collect_after_push': 1 + 5, 'asked_push_after_push': 0 + 5}
json.dump(R, open('results.json', 'w'), indent=1)
print(json.dumps(R, indent=1)[:1800])
