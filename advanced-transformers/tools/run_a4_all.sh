#!/bin/sh
# A4: wait for the trained models, then run the system experiments for 3 seeds (lowest priority).
cd "$(dirname "$0")/.." && mkdir -p results/a4
for s in 0 1 2; do (while [ ! -e results/a4/interleaved_s$s.pt ]; do sleep 15; done; PYTHONPATH=. nice -n 19 ../.venv/bin/python tools/run_a4.py $s > results/a4/systems_s$s.log 2>&1) & done; wait
echo done > results/a4/A4_DONE
