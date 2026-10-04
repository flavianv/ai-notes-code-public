#!/bin/sh
# A2: depth x length and counting vs state + probes, 3 seeds each, 2 at a time, lowest priority.
cd "$(dirname "$0")/.." && mkdir -p results/a2
for s in 0 1 2; do echo "depth $s"; echo "general $s"; done |
  nice -n 19 xargs -P 2 -n 2 sh -c 'PYTHONPATH=. ../.venv/bin/python tools/run_a2.py $0 $1 > results/a2/$0_s$1.log 2>&1'
echo done > results/a2/A2_DONE
