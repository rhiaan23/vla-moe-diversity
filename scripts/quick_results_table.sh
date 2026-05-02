#!/usr/bin/env bash
# Quick-look results across all manual-routing sweeps in one table.
# Usage: ./scripts/quick_results_table.sh
set -uo pipefail
OUT=/scratch/gpfs/FHEIDE/rj2807/outputs/evals/manual_routing

printf "%-50s %s\n" "condition" "successes/N"
printf "%-50s %s\n" "----------" "-----------"
for d in $OUT/*/*/eval_info.json; do
  [ -f "$d" ] || continue
  cond=$(basename $(dirname "$d"))
  family=$(basename $(dirname $(dirname "$d")))
  python -c "
import json
d = json.load(open('$d'))
results = []
for ent in d.get('per_task', []):
    m = ent['metrics']
    s = m['successes']
    results.append(f'T{ent[\"task_id\"]}={sum(s)}/{len(s)}')
print('  ' + ' '.join(results))
" 2>/dev/null | xargs -I{} printf "%-50s %s\n" "$family/$cond" "{}"
done
