#!/usr/bin/env python3
"""Non-interactive first pass for map_translation_segments.py.

propose: run the normal mapping logic, but instead of prompting, write every
         batch proposal to a review file. Literal matches with stored approved
         hypotheses are still auto-approved and saved, exactly as in the
         interactive script.
apply:   merge the proposals marked "accept" into <model>.json (and add their
         text to the stored hypotheses). The interactive script then skips them
         and shows only the rest.

Usage:
    python code/map_first_pass.py propose ivan-vas gpt-6-sol Russian English [--methods zero,list]
    python code/map_first_pass.py apply   ivan-vas gpt-6-sol Russian English
Review file: experiments/films/output/translations/<film>/<src>-<tgt>/<model>/mapping/first_pass.json
Each proposal gets "decision": null; set it to "accept" or "flag" before apply.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import map_translation_segments as m
from lang_utils import normalize_lang


def mapping_dir(film, model, src, tgt):
    return (Path("experiments/films/output/translations") / film
            / f"{normalize_lang(src)}-{normalize_lang(tgt)}" / model / "mapping")


def propose(film, model, src, tgt, rest):
    proposals = []

    def record_batch(batch, srt_map, method_name, run_number, batch_idx, n_batches,
                     method_outputs, progress_data, progress_file,
                     item_text_hypotheses, overrides_file, target_lang):
        for item, expected_segs, proposed_segs, proposed_text, source_note in batch:
            item_id = str(item["id"])
            hyps = item_text_hypotheses.get(item_id) or []
            srt_text = proposed_text or m.join_segments(proposed_segs, srt_map)
            trimmed = next((t for h in reversed(hyps) for t in [m._apply_hypothesis_trim(srt_text, h)] if t), None) if hyps else None
            lo, hi = min(proposed_segs + expected_segs) - 2, max(proposed_segs + expected_segs) + 2
            proposals.append({
                "item_id": item_id, "label": m.get_item_label(item), "method": method_name, "run": str(run_number),
                "original": (item.get("original") or {}).get("rus", ""),
                "reference": m.get_reference_translation(item, target_lang),
                "prior_hypotheses": hyps,
                "expected_segs": expected_segs, "proposed_segs": proposed_segs, "note": source_note,
                "text": trimmed or srt_text,
                "context": {str(s): srt_map[s] for s in range(lo, hi + 1) if s in srt_map},
                "decision": None,
            })
        return []  # nothing flagged, nothing saved

    m._process_batch = record_batch
    sys.argv = ["map_translation_segments.py", film, model, src, tgt] + rest
    m.main()  # saves auto-approved items into <model>.json as usual
    out = mapping_dir(film, model, src, tgt) / "first_pass.json"
    m.save_json(proposals, out)
    print(f"\n{len(proposals)} proposal(s) written to {out}")


def apply(film, model, src, tgt):
    # Accepted proposals are merged straight into <model>.json. They are not routed through
    # progress.json: map_translation_segments.py skips items found there but never merges
    # them into its output, and then deletes the progress file.
    d = mapping_dir(film, model, src, tgt)
    proposals = m.load_json(d / "first_pass.json")
    output_path = Path("experiments/films/output/translations") / film / f"{model}.json"
    base = m.load_json(output_path)
    new = {}
    for p in proposals:
        if p.get("decision") == "accept":
            new.setdefault(p["item_id"], {}).setdefault(p["method"], {})[p["run"]] = p["text"]
    merged = m.merge_into_existing_json(base, base, new, model, m.lang_code(tgt))
    m.save_json(merged, output_path)
    overrides_file = d.parent.parent / "mapping" / "overrides.json"
    ov = m.load_json(overrides_file) if overrides_file.exists() else {"hypotheses": {}}
    hyps = ov.setdefault("hypotheses", {})
    n = 0
    for p in proposals:
        if p.get("decision") != "accept":
            continue
        known = hyps.setdefault(p["item_id"], [])
        if p["text"] not in known:
            known.append(p["text"])
        n += 1
    m.save_json(ov, overrides_file)
    flagged = sum(1 for p in proposals if p.get("decision") == "flag")
    print(f"Accepted {n}, flagged {flagged}, undecided {len(proposals) - n - flagged}. "
          f"Merged into {output_path}.")


if __name__ == "__main__":
    mode, film, model, src, tgt, *rest = sys.argv[1:]
    {"propose": lambda: propose(film, model, src, tgt, rest), "apply": lambda: apply(film, model, src, tgt)}[mode]()
