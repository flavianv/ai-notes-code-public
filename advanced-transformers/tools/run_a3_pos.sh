#!/bin/sh
# A3 follow-up: is the interleaved break point about position? NoPE and random-offset variants, 3 seeds each, 3 at a time, lowest priority.
cd "$(dirname "$0")/.." && mkdir -p results/a3
for s in 0 1 2; do for f in interleaved-nope interleaved-offset; do echo "$f $s"; done; done |
  nice -n 19 xargs -P 3 -n 2 sh -c 'PYTHONPATH=. ../.venv/bin/python tools/run_a3.py $0 $1 > results/a3/$0_s$1.log 2>&1'
echo done > results/a3/POS_DONE
