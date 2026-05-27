#!/bin/bash
# Moves old flat-structure eval results into the prompt-named subdirectory.
#
# The evaluate_mqm_parallel.py script was updated to write results under
# a subdirectory named after the prompt file (e.g. mqm-memes/).
# Results produced with the old script sit directly inside the model dir.
# This script scans the entire llm-eval tree, finds every model directory
# (identified by the *-by-* naming convention), and moves all of its
# contents — subdirectories and files alike — into mqm-memes/, leaving
# any already-correct mqm-* subdirectories in place.
#
# Directories under any path component named "old" are skipped.

ROOT="experiments/films/output/eval/llm-eval"

find "$ROOT" -type d | sort | while read -r dir; do
    # Skip anything inside an "old" archive directory
    case "$dir" in *"/old"*) continue ;; esac

    # Only operate on model-level directories (name contains "-by-")
    case "$(basename "$dir")" in *-by-*) ;; *) continue ;; esac

    # Collect everything in this dir that is not already mqm-named
    items=()
    while IFS= read -r -d '' item; do
        name=$(basename "$item")
        case "$name" in mqm-*) continue ;; esac
        items+=("$item")
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0)

    [ ${#items[@]} -eq 0 ] && continue

    target="$dir/mqm-memes"
    echo "Migrating: $dir"
    mkdir -p "$target"
    for item in "${items[@]}"; do
        echo "  Moving: $(basename "$item")"
        mv "$item" "$target/"
    done
done

echo "Done."
