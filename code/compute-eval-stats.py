import json
import sys
from pathlib import Path
from evaluate_mqm_parallel import compute_mqm_score

def batch_update_summaries(source_dir, output_dir=None):
    source_path = Path(source_dir)

    # If output_dir is provided, create it; otherwise, we overwrite in source_path
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"Output mode: Saving new files to {out_path}")
    else:
        out_path = source_path
        print(f"Output mode: Overwriting files in {source_path}")

    if not source_path.is_dir():
        print(f"Error: {source_dir} is not a valid directory.")
        return

    json_files = list(source_path.glob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return

    for file_path in json_files:
        # Avoid processing the aggregated summary file if it exists
        if "aggregated_summary" in file_path.name:
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "items" not in data:
                continue

            # Compute score using the file itself as the reference for units
            summary = compute_mqm_score(data, data)
            data["summary"] = summary

            # Determine final save location
            save_path = out_path / file_path.name

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"  [PROCESSED] {file_path.name}")

        except Exception as e:
            print(f"  [ERROR] Failed {file_path.name}: {e}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    # Usage:
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <source_directory> <output_directory>")
        sys.exit(1)

    src = sys.argv[1]
    out = sys.argv[2]

    batch_update_summaries(src, out)