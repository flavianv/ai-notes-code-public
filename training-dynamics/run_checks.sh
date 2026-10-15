#!/bin/sh
# Runs every narration check: each asserts the numbers of a film or part against its saved results.
cd "$(dirname "$0")" || exit 1
ok=0; bad=0
for d in films/* parts/*; do
  [ -f "$d/make_narration.py" ] || continue
  if (cd "$d" && python3 make_narration.py > /dev/null 2>&1); then ok=$((ok+1)); else echo "FAILED: $d"; bad=$((bad+1)); fi
done
echo "$ok passed, $bad failed"
