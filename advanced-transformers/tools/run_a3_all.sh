#!/bin/sh
# Run all A3 jobs (4 setups x 3 seeds), 4 at a time, detached: python3 -c "import os; os.setsid(); os.execvp('sh', ['sh', 'advanced-transformers/tools/run_a3_all.sh'])" &
cd "$(dirname "$0")/.." && mkdir -p results/a3
for s in 0 1 2; do for f in direct after interleaved looped; do echo "$f $s"; done; done |
  nice -n 19 xargs -P 4 -n 2 sh -c 'PYTHONPATH=. ../.venv/bin/python tools/run_a3.py $0 $1 > results/a3/$0_s$1.log 2>&1'
echo done > results/a3/ALL_DONE
