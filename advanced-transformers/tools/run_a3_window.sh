#!/bin/sh
# A3 follow-up 2: attention limited to the last 8 tokens. Waits for the position runs, then 3 seeds, 3 at a time, lowest priority.
cd "$(dirname "$0")/.." && mkdir -p results/a3
while pgrep -f 'run_a3.py' >/dev/null; do sleep 20; done
for s in 0 1 2; do echo "interleaved-window $s"; done |
  nice -n 19 xargs -P 3 -n 2 sh -c 'PYTHONPATH=. ../.venv/bin/python tools/run_a3.py $0 $1 > results/a3/$0_s$1.log 2>&1'
echo done > results/a3/WINDOW_DONE
