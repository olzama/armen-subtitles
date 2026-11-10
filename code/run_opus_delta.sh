#!/usr/bin/env bash
# Extra Opus 5.5 translations for one film, to reach a sensitivity of 0.1.
# See docs/handoff-opus-delta.md.
#
# Usage (from the repository root):
#   code/run_opus_delta.sh <film> translate   # add the missing translations (claude -p, slow)
#   code/run_opus_delta.sh <film> map         # map meme spans by time code, then verify them
#   code/run_opus_delta.sh <film> eval        # Sonnet 5.5 Meme-MQM v2, 2 runs, new translations only
#   code/run_opus_delta.sh <film> stats       # aggregate + variance (target 0.1)
#   code/run_opus_delta.sh <film> all         # translate, map, eval, stats in order
# <film>: ivan-vas | diamond-arm | gentelmen | carnival-night | pokrov-gate
set -euo pipefail

FILM=${1:?film}; PHASE=${2:?phase}
CFG=yaml-pipelines/films/${FILM}-russian-english-opus-delta.yaml
MODEL=claude-opus-5-5; JUDGE=claude-sonnet-5-5
METHODS=zero,list,list-analysis
PROMPT=experiments/films/prompts/eval/mqm-memes-v2.txt
TDIR=experiments/films/output/translations/${FILM}/Russian-English/${MODEL}
EDIR=experiments/films/output/eval/llm-eval/${FILM}/Russian-English/${MODEL}-by-${JUDGE}/mqm-memes-v2
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }

# Translation runs above 3 that exist on disk but have no Sonnet v2 score yet ("4,5,...").
new_runs() {
  python3 - "$TDIR" "$EDIR" <<'EOF'
import sys
from pathlib import Path
tdir, edir = Path(sys.argv[1]), Path(sys.argv[2])
runs = set()
for m in ("zero", "list", "list-analysis"):
    for f in (tdir / m / "translations").glob("translation-*.txt"):
        r = int(f.stem.split("-")[1])
        if r > 3 and not (edir / m / f"run_{r}_eval_1.json").exists():
            runs.add(r)
print(",".join(str(r) for r in sorted(runs)))
EOF
}

translate() { python3 run_pipeline.py "$CFG" --step translate; }

map() {
  # Uniform mapping (docs/decisions-2026-10.md): time-code spans, GPT-6 Sol trims; never touches scored spans.
  python3 code/extract_meme_spans.py "$FILM" "$MODEL" Russian English --methods "$METHODS" --only-missing
  python3 code/verify_spans.py "$FILM" "$MODEL" Russian English --methods "$METHODS" | tee "/tmp/verify-${FILM}.log"
  echo
  echo "If the last line lists FLAGGED method:run pairs for runs above 3, re-locate them by content:"
  echo "  python3 code/extract_meme_spans.py $FILM $MODEL Russian English --runs <method:run,...> --content"
  echo "and re-run this phase. Flags on runs 1-3 were reviewed before; leave them."
}

evaluate() {
  RUNS=$(new_runs)
  if [ -z "$RUNS" ]; then echo "no unscored new runs"; return; fi
  echo "scoring runs $RUNS"
  python3 code/evaluate_mqm_parallel.py "$FILM" Russian English "$MODEL" "$JUDGE" 2 "$PROMPT" \
      --methods "$METHODS" --runs "$RUNS"
}

stats() {
  python3 code/aggregate_mqm.py "$FILM" "$MODEL" "$JUDGE" Russian English mqm-memes-v2
  python3 code/variance.py "$FILM" "$MODEL" "$JUDGE" 0.1 Russian English mqm-memes-v2
}

case "$PHASE" in
  translate) translate ;;
  map) map ;;
  eval) evaluate ;;
  stats) stats ;;
  all) translate; map; evaluate; stats ;;
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac
