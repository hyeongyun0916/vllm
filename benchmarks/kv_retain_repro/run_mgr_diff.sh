#!/bin/bash
# Manager-level differential test: run mgr_trace.py against the PR tree and the base tree for many seeds and compare
# traces line by line. usage: run_mgr_diff.sh <pr_tree> <base_tree> [seeds] [steps] [num_blocks]
PR=$1; BASE=$2; SEEDS=${3:-100}; STEPS=${4:-1500}; NB=${5:-40}
PY=${PY:-python}; HERE=$(cd "$(dirname "$0")" && pwd); OUT=$(mktemp -d)
diverged=0; evictions=0
for s in $(seq 0 $((SEEDS - 1))); do
  (cd "$OUT" && PYTHONPATH=$PR "$PY" "$HERE/mgr_trace.py" "$s" "$STEPS" "$NB" > pr.$s 2>/dev/null) &
  (cd "$OUT" && PYTHONPATH=$BASE "$PY" "$HERE/mgr_trace.py" "$s" "$STEPS" "$NB" > base.$s 2>/dev/null) &
  wait
  [ -s "$OUT/pr.$s" ] || { echo "seed $s: PR run produced no output"; exit 2; }
  if ! cmp -s "$OUT/pr.$s" "$OUT/base.$s"; then
    diverged=$((diverged + 1)); echo "seed $s diverges at: $(diff "$OUT/pr.$s" "$OUT/base.$s" | head -2 | tail -1 | cut -c1-120)"
  fi
done
lines=$(cat "$OUT"/pr.* | wc -l); hits=$(grep -ho "hits=[1-9][0-9]*" "$OUT"/pr.* | wc -l); evictions=$(grep -h "^end evictions=" "$OUT"/pr.* | cut -d= -f2 | awk '{s+=$1} END{print s+0}')
echo "$SEEDS seeds x $STEPS steps ($lines ops, $hits allocations with cache hits, $evictions evictions): $diverged divergent"
rm -rf "$OUT"
