#!/bin/bash
# Moves old flat-structure eval results into the prompt-named subdirectory.
#
# The evaluate_mqm_parallel.py script was updated to write results under
# a subdirectory named after the prompt file (e.g. mqm-memes/).
# Results produced with the old script sit directly inside the model dir.
# This script detects that situation and moves the method directories into
# mqm-memes/ so that aggregate_mqm.py and variance.py find them correctly.

BASE="films/output/eval/llm-eval/ivan-vas/Russian-Galician/gpt-5.2-by-gpt-5.4-mini"
METHODS="zero characters narratives list-analysis noise"
TARGET="$BASE/mqm-memes"

found=0
for m in $METHODS; do
    if [ -d "$BASE/$m" ]; then
        found=1
        break
    fi
done

if [ "$found" -eq 0 ]; then
    echo "Nothing to do — no flat-structure method directories found under $BASE."
    exit 0
fi

echo "Found old-style method directories. Moving them into $TARGET/ ..."
mkdir -p "$TARGET"
for m in $METHODS; do
    if [ -d "$BASE/$m" ]; then
        mv "$BASE/$m" "$TARGET/"
        echo "  Moved: $m"
    fi
done
echo "Done."
