#!/usr/bin/env python3
# Requires in LaTeX preamble: \usepackage{colortbl} \usepackage{booktabs} \usepackage{xcolor} \usepackage{adjustbox}

import argparse
import json
import math
from pathlib import Path


DEFAULT_METHOD_ORDER = [
    "zero", "zero-lang",
    "summary",
    "characters", "characters-lang",
    "narratives",
    "intertext",
    "examples",
    "list", "list-lang",
    "list-analysis", "list-analysis-lang",
    "meme-search",
    "given", "given-lang",
    "noise",
]

# Methods placed after the | separator
SEPARATOR_METHODS = {"given", "given-lang", "noise"}

METHOD_DISPLAY = {
    "zero": "zero",
    "zero-lang": "+lang",
    "summary": "summary",
    "characters": "characters",
    "characters-lang": "+lang",
    "narratives": "narratives",
    "intertext": "intertext",
    "examples": "examples",
    "list": "meme list",
    "list-lang": "+lang",
    "list-analysis": "list+analysis",
    "list-analysis-lang": "+lang",
    "meme-search": "meme-search",
    "given": "given trans.",
    "given-lang": "+lang",
    "noise": "added noise",
}

METHOD_ALIASES = {
    "meme-list": "list",
    "meme_list": "list",
    "list+analysis": "list-analysis",
    "given-trans": "given",
    "given_trans": "given",
}


def canonical_method_name(name):
    return METHOD_ALIASES.get(name.strip(), name.strip())


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_float(x, digits=2):
    if x is None:
        return "--"
    if not isinstance(x, (int, float)):
        return "--"
    if math.isnan(x) or math.isinf(x):
        return "--"
    return f"{x:.{digits}f}"


def format_intlike(x):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "--"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "--"
        return str(int(round(x)))
    try:
        return str(int(round(float(x))))
    except Exception:
        return "--"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    methods_raw = data.get("methods")
    if not isinstance(methods_raw, list):
        raise ValueError(f"{path}: expected a top-level 'methods' list")

    methods = {}
    for item in methods_raw:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("method")
        if not isinstance(raw_name, str):
            continue
        methods[canonical_method_name(raw_name)] = item

    data["_methods_by_name"] = methods
    return data


def load_human_json(path):
    """Load a summary_human.json and return a dict keyed by canonical method name.

    Each value contains at least: mean_major_equiv_per_unit, ci_95_half_width, se_method.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    per_method_raw = data.get("per_method")
    if not isinstance(per_method_raw, dict):
        raise ValueError(f"{path}: expected a top-level 'per_method' dict")

    return {
        canonical_method_name(k): v
        for k, v in per_method_raw.items()
        if isinstance(v, dict)
    }


def get_metric(method_entry, keys):
    for key in keys:
        val = method_entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def get_nested_metric(method_entry, parent_keys, child_keys):
    for parent in parent_keys:
        obj = method_entry.get(parent)
        if isinstance(obj, dict):
            for child in child_keys:
                if child in obj:
                    return obj[child]
    return None


def infer_film_name(data, fallback):
    for key in ["film", "dataset_name", "title", "name"]:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return fallback


def extract_row(data, method_order, metric_keys, digits=2):
    out = []
    methods = data["_methods_by_name"]

    for method in method_order:
        entry = methods.get(method)
        if entry is None:
            out.append("")
            continue
        out.append(format_float(get_metric(entry, metric_keys), digits=digits))

    return out


def extract_sensitivity_row(data, method_order, child_keys):
    out = []
    methods = data["_methods_by_name"]

    for method in method_order:
        entry = methods.get(method)
        if entry is None:
            out.append("")
            continue

        val = get_nested_metric(
            entry,
            parent_keys=["sensitivity", "delta_sensitivity_requirements"],
            child_keys=child_keys,
        )
        out.append(format_intlike(val))

    return out


def validate_sensitivity_present(data, method_order):
    methods = data["_methods_by_name"]
    missing = []

    for method in method_order:
        entry = methods.get(method)
        if entry is None:
            continue

        min_t = get_nested_metric(
            entry,
            parent_keys=["sensitivity", "delta_sensitivity_requirements"],
            child_keys=["min_T_required_at_current_E"],
        )
        min_e = get_nested_metric(
            entry,
            parent_keys=["sensitivity", "delta_sensitivity_requirements"],
            child_keys=["min_E_required_at_current_T"],
        )

        if min_t is None or min_e is None:
            missing.append(method)

    if missing:
        print(
            "Sensitivity info is missing for method(s): "
            + ", ".join(missing)
        )


LANG_METHODS = {m for m in DEFAULT_METHOD_ORDER if m.endswith("-lang")}

# These 10 methods are always shown as columns, even when absent from the data (empty cell).
CORE_METHODS = {
    "zero", "summary", "characters", "narratives", "intertext",
    "examples", "list", "list-analysis", "meme-search", "given", "noise",
}


def build_column_spec(method_order):
    cols = ["l"]
    separator_inserted = False
    for method in method_order:
        if method in SEPARATOR_METHODS and not separator_inserted:
            cols.append("|")
            separator_inserted = True
        if method in LANG_METHODS:
            cols.append(r">{\columncolor{gray!15}}r")
        else:
            cols.append("r")
    return "".join(cols)


def build_header_rows(method_order):
    """
    Build two-row headers for paired (method, method-lang) columns.

    Returns:
        row1      list of cell strings (multicolumn for pairs)
        row2      list of cell strings (empty except '+lang' for lang columns)
        cmidrules list of (start_col, end_col) 1-indexed for cmidrule lines
    """
    row1 = ["Film"]
    row2 = [""]
    cmidrules = []
    col = 2  # column 1 is Film

    i = 0
    while i < len(method_order):
        method = method_order[i]
        is_pair = (
            i + 1 < len(method_order)
            and method_order[i + 1] == method + "-lang"
        )
        display = latex_escape(METHOD_DISPLAY.get(method, method))
        border = "|" if (method in SEPARATOR_METHODS and not
                         any(m in SEPARATOR_METHODS for m in method_order[:i])) else ""

        if is_pair:
            row1.append(rf"\multicolumn{{2}}{{{border}c}}{{{display}}}")
            row2.append("")
            row2.append("+lang")
            cmidrules.append((col, col + 1))
            col += 2
            i += 2
        else:
            if border:
                row1.append(rf"\multicolumn{{1}}{{{border}r}}{{{display}}}")
            else:
                row1.append(display)
            row2.append("")
            col += 1
            i += 1

    return row1, row2, cmidrules


def infer_sensitivity_target(data):
    """Return the sensitivity target from any method's sensitivity.target field, or None."""
    for entry in data.get("methods", []):
        sens = entry.get("sensitivity")
        if isinstance(sens, dict):
            val = sens.get("target")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
    return None


def extract_human_row(human_methods, method_order, metric_key, digits=2):
    """One cell per method from human per_method dict; '--' where data is absent."""
    out = []
    for method in method_order:
        entry = human_methods.get(method) if human_methods else None
        out.append("--" if entry is None else format_float(entry.get(metric_key), digits=digits))
    return out


def build_table(data_list, film_names, method_order, caption, label,
                human_data_list=None, digits=2):
    colspec = build_column_spec(method_order)
    row1, row2, cmidrules = build_header_rows(method_order)

    if human_data_list is None:
        human_data_list = [None] * len(data_list)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(" & ".join(row1) + r" \\")
    if cmidrules:
        lines.append(" ".join(rf"\cmidrule(lr){{{s}-{e}}}" for s, e in cmidrules))
    lines.append(" & ".join(row2) + r" \\")
    lines.append(r"\hline")

    for idx, (data, film_name, human_methods) in enumerate(
        zip(data_list, film_names, human_data_list)
    ):
        # ── Auto-eval rows ──────────────────────────────────────────────
        score_row  = extract_row(data, method_order,
                                 metric_keys=["mean_major_equiv_per_unit"], digits=digits)
        ci_row     = extract_row(data, method_order,
                                 metric_keys=["ci_95_half_width"], digits=digits)
        se_row     = extract_row(data, method_order,
                                 metric_keys=["avg_eval_noise"], digits=digits)
        sens_t_row = extract_sensitivity_row(data, method_order,
                                             child_keys=["min_T_required_at_current_E"])
        sens_e_row = extract_sensitivity_row(data, method_order,
                                             child_keys=["min_E_required_at_current_T"])

        lines.append(r"\textbf{" + latex_escape(film_name) + "} & "
                     + " & ".join(score_row) + r" \\")
        lines.append("")
        lines.append(r"\emph{Method 95\% CI} & " + " & ".join(ci_row) + r" \\")
        lines.append("")
        lines.append(r"\emph{SE due to Evaluator SD} & " + " & ".join(se_row) + r" \\")
        lines.append("")
        delta = infer_sensitivity_target(data)
        delta_str = format_float(delta, digits=digits) if delta is not None else "?"
        lines.append(rf"\emph{{N/translations to {delta_str} sensitivity}} & "
                     + " & ".join(sens_t_row) + r" \\")
        lines.append("")
        lines.append(rf"\emph{{N/eval runs to {delta_str} sensitivity}} & "
                     + " & ".join(sens_e_row) + r" \\")
        lines.append("")

        lines.append(r"\hline")
        if idx != len(data_list) - 1:
            lines.append("")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{" + latex_escape(caption) + "}")
    lines.append(r"\label{" + latex_escape(label) + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_human_table(film_names, method_order, human_data_list,
                      caption, label, digits=2):
    """Build a separate LaTeX table showing human evaluation scores per method.

    Films with no human data are skipped entirely.
    Methods not covered by the human evaluator show '--'.
    """
    colspec = build_column_spec(method_order)
    row1, row2, cmidrules = build_header_rows(method_order)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(" & ".join(row1) + r" \\")
    if cmidrules:
        lines.append(" ".join(rf"\cmidrule(lr){{{s}-{e}}}" for s, e in cmidrules))
    lines.append(" & ".join(row2) + r" \\")
    lines.append(r"\hline")

    film_blocks = [
        (film_name, human_methods)
        for film_name, human_methods in zip(film_names, human_data_list)
        if human_methods is not None
    ]

    for idx, (film_name, human_methods) in enumerate(film_blocks):
        score_row = extract_human_row(human_methods, method_order,
                                      "mean_major_equiv_per_unit", digits=digits)
        ci_row    = extract_human_row(human_methods, method_order,
                                      "ci_95_half_width", digits=digits)
        se_row    = extract_human_row(human_methods, method_order,
                                      "se_method", digits=digits)

        lines.append(r"\textbf{" + latex_escape(film_name) + "} & "
                     + " & ".join(score_row) + r" \\")
        lines.append("")
        lines.append(r"\emph{95\% CI} & " + " & ".join(ci_row) + r" \\")
        lines.append("")
        lines.append(r"\emph{SE} & " + " & ".join(se_row) + r" \\")
        lines.append("")

        lines.append(r"\hline")
        if idx != len(film_blocks) - 1:
            lines.append("")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{" + latex_escape(caption) + "}")
    lines.append(r"\label{" + latex_escape(label) + "}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert one or more method_comparison JSON files into a LaTeX table."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[],
        help="Input JSON file(s). Each file becomes one film block. Optional if --human is used alone.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .tex file path.",
    )
    parser.add_argument(
        "--film-names",
        nargs="*",
        default=None,
        help="Optional film names, one per input file, in the same order.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHOD_ORDER,
        help="Column method order.",
    )
    parser.add_argument(
        "--caption",
        default="MQM adapted score (major-error equivalents per meaning unit). "
                "Shaded columns (+lang) repeat the same method with target-language-specific instructions added; "
                "given trans. and added noise serve as reference conditions.",
        help="LaTeX caption text.",
    )
    parser.add_argument(
        "--lang-pair",
        default=None,
        help=(
            "Language pair prefix for the caption, e.g. 'Rus->Eng'. "
            "If omitted, inferred from the input file path (looks for a 'Src-Tgt' directory component)."
        ),
    )
    parser.add_argument(
        "--label",
        default="tab:mqm-method-comparison",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal places for floating-point values.",
    )
    parser.add_argument(
        "--human",
        nargs="*",
        default=None,
        metavar="HUMAN_JSON",
        help=(
            "Optional summary_human.json file(s), one per input film in the same order. "
            "Use 'none' as a placeholder for films without human data."
        ),
    )
    return parser.parse_args()


def infer_lang_pair(paths):
    """Return 'Src->Tgt' inferred from the first path that has a 'Word-Word' directory component."""
    import re
    for p in paths:
        for part in Path(p).parts:
            m = re.match(r'^([A-Z][a-z]+)-([A-Z][a-z]+)$', part)
            if m:
                return f"{m.group(1)}->{m.group(2)}"
    return None


def prepend_lang_pair(caption, lang_pair):
    if lang_pair:
        return f"{lang_pair}: {caption}"
    return caption


def main():
    args = parse_args()

    if not args.inputs and not args.human:
        raise ValueError("Provide at least one input JSON or --human file(s).")

    input_paths = [Path(p) for p in args.inputs]
    method_order = [canonical_method_name(m) for m in args.methods]

    # If using the default order, always keep core methods; drop non-core (e.g. -lang)
    # variants only when absent from all input files.
    if args.methods == DEFAULT_METHOD_ORDER and args.inputs:
        all_present = set()
        for p in [Path(p) for p in args.inputs]:
            d = json.load(open(p))
            for item in d.get("methods", []):
                all_present.add(canonical_method_name(item.get("method", "")))
        method_order = [m for m in method_order if m in CORE_METHODS or m in all_present]

    all_paths = list(input_paths) + ([Path(h) for h in args.human] if args.human else [])
    lang_pair = args.lang_pair or infer_lang_pair(all_paths)
    caption = prepend_lang_pair(args.caption, lang_pair)

    # Human-only mode: no LLM input files
    if not input_paths:
        human_paths = [Path(h) for h in args.human]
        if args.film_names and len(args.film_names) > 0:
            if len(args.film_names) != len(human_paths):
                raise ValueError(
                    f"--film-names must contain exactly {len(human_paths)} entries"
                )
            film_names = args.film_names
        else:
            film_names = [p.stem for p in human_paths]

        human_data_list = [load_human_json(p) for p in human_paths]

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        human_tex = build_human_table(
            film_names=film_names,
            method_order=method_order,
            human_data_list=human_data_list,
            caption=caption.rstrip(".") + " (human evaluation).",
            label=args.label + "-human",
            digits=args.digits,
        )
        out_path.write_text(human_tex + "\n", encoding="utf-8")
        return

    data_list = [load_json(p) for p in input_paths]

    for data in data_list:
        validate_sensitivity_present(data, method_order)

    if args.film_names is not None and len(args.film_names) > 0:
        if len(args.film_names) != len(input_paths):
            raise ValueError(
                f"--film-names must contain exactly {len(input_paths)} entries"
            )
        film_names = args.film_names
    else:
        film_names = [
            infer_film_name(data, fallback=path.stem)
            for data, path in zip(data_list, input_paths)
        ]

    # Load human data: None for films without a human summary
    if args.human:
        if len(args.human) != len(input_paths):
            raise ValueError(
                f"--human must contain exactly {len(input_paths)} entries "
                f"(use 'none' as a placeholder)"
            )
        human_data_list = [
            None if h.lower() == "none" else load_human_json(Path(h))
            for h in args.human
        ]
    else:
        human_data_list = None

    tex = build_table(
        data_list=data_list,
        film_names=film_names,
        method_order=method_order,
        caption=caption,
        label=args.label,
        digits=args.digits,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex + "\n", encoding="utf-8")

    if human_data_list is not None:
        human_tex = build_human_table(
            film_names=film_names,
            method_order=method_order,
            human_data_list=human_data_list,
            caption=caption.rstrip(".") + " (human evaluation).",
            label=args.label + "-human",
            digits=args.digits,
        )
        human_out_path = out_path.with_stem(out_path.stem + "_human")
        human_out_path.write_text(human_tex + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()