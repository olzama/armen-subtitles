#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


DEFAULT_METHOD_ORDER = [
    "zero",
    "summary",
    "characters",
    "narratives",
    "intertext",
    "examples",
    "list",
    "list-analysis",
    "given",
    "noise",
]

METHOD_DISPLAY = {
    "zero": "zero",
    "noise": "added noise",
    "summary": "summary",
    "characters": "characters",
    "narratives": "narratives",
    "intertext": "intertext",
    "examples": "examples",
    "list": "meme list",
    "list-analysis": "list+analysis",
    "given": "given trans.",
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
            out.append("--")
            continue
        out.append(format_float(get_metric(entry, metric_keys), digits=digits))

    return out


def extract_sensitivity_row(data, method_order, child_keys):
    out = []
    methods = data["_methods_by_name"]

    for method in method_order:
        entry = methods.get(method)
        if entry is None:
            out.append("--")
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


def build_column_spec(num_methods):
    if num_methods < 1:
        raise ValueError("Need at least one method column")
    if num_methods == 1:
        return "l|l"
    return "l" + ("r" * (num_methods - 1)) + "|r"


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
    colspec = build_column_spec(len(method_order))
    header_methods = [METHOD_DISPLAY.get(m, m) for m in method_order]

    if human_data_list is None:
        human_data_list = [None] * len(data_list)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append("Film & " + " & ".join(header_methods) + r" \\")
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
    colspec = build_column_spec(len(method_order))
    header_methods = [METHOD_DISPLAY.get(m, m) for m in method_order]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{adjustbox}{width=\textwidth}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append("Film & " + " & ".join(header_methods) + r" \\")
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
        nargs="+",
        help="Input JSON file(s). Each file becomes one film block.",
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
        default="MQM adapted score (major-error equivalents per meaning unit).",
        help="LaTeX caption text.",
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


def main():
    args = parse_args()

    input_paths = [Path(p) for p in args.inputs]
    method_order = [canonical_method_name(m) for m in args.methods]

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
        caption=args.caption,
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
            caption=args.caption.rstrip(".") + " (human evaluation).",
            label=args.label + "-human",
            digits=args.digits,
        )
        human_out_path = out_path.with_stem(out_path.stem + "_human")
        human_out_path.write_text(human_tex + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()