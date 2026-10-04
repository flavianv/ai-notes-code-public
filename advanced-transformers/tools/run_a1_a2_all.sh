#!/bin/sh
# Run A1 and A2 (after any A3 jobs), 4 jobs at a time, lowest priority. Launch detached:
#   python3 -c "import os; os.setsid(); os.execvp('sh', ['sh', 'advanced-transformers/tools/run_a1_a2_all.sh'])" &
cd "$(dirname "$0")/.." && mkdir -p results/a1 results/a2
while pgrep -f 'run_a3.py' >/dev/null; do sleep 30; done            # wait for A3 to free the CPU
{ for s in 0 1 2; do echo "tools/run_a1.py induction $s"; done
  echo "tools/run_a1.py superposition"
  for s in 0 1 2; do echo "tools/run_a2.py depth $s"; echo "tools/run_a2.py general $s"; done; } |
  nice -n 19 xargs -P 4 -L 1 sh -c 'PYTHONPATH=. ../.venv/bin/python "$@" > "results/$(echo $1 | sed "s/.*run_//;s/.py//")/$2${3:+_s$3}.log" 2>&1' _
echo done > results/A1_A2_DONE
