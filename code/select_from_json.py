import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

VERSIONS = ["general-text", "general-all", "general+lang-text", "all"]


def select_analysis(analysis: Dict, version: str) -> Dict:
    general = analysis.get("general", {})
    lang_specific = analysis.get("language_specific", {})

    if isinstance(general, dict):
        general_text = general.get("text")
        general_nb = general.get("nb")
    else:
        general_text = general
        general_nb = None

    if version == "general-text":
        return {"general": {"text": general_text}}

    elif version == "general-all":
        return {"general": {"text": general_text, "nb": general_nb}}

    elif version == "general+lang-text":
        lang_texts = {
            lang: {"text": (val.get("text") if isinstance(val, dict) else val)}
            for lang, val in lang_specific.items()
        }
        return {"general": {"text": general_text}, "language_specific": lang_texts}

    elif version == "all":
        return {"general": {"text": general_text, "nb": general_nb}, "language_specific": lang_specific}

    raise ValueError(f"Unknown version '{version}'. Must be one of: {VERSIONS}")


def select_item(item: Dict, version: str) -> Dict:
    result = {}
    for key, val in item.items():
        if key == "reference":
            continue
        if key == "analysis":
            result[key] = select_analysis(val, version)
        else:
            result[key] = val
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Select analysis fields from reference JSON for use as unit_list."
    )
    parser.add_argument("input", type=Path, help="Input JSON file (reference.json)")
    parser.add_argument("output", type=Path, nargs="?", default=None,
                        help="Output JSON file (default: list-analysis-{version}.json next to input)")
    parser.add_argument("--version", choices=VERSIONS, default="general-all",
                        help="Analysis version to include (default: general-all)")
    args = parser.parse_args()
    output = args.output or args.input.parent / f"list-analysis-{args.version}.json"

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        result = {k: v for k, v in data.items() if k != "items"}
        result["items"] = [select_item(item, args.version) for item in data["items"]]
    elif isinstance(data, list):
        result = [select_item(item, args.version) for item in data]
    else:
        raise ValueError("Unexpected JSON structure: expected a list or a dict with 'items'")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
