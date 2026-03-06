import json
import sys
from typing import Any, Dict, List, Union


JSONType = Union[Dict[str, Any], List[Any]]


def remove_fields(data: JSONType, fields_to_remove: List[str]) -> JSONType:
    """
    Recursively remove specified fields from nested JSON structure.
    """

    if isinstance(data, dict):
        new_dict = {}

        for key, value in data.items():
            # Skip keys that should be removed
            if key in fields_to_remove:
                continue

            # Recurse into nested structures
            filtered_value = remove_fields(value, fields_to_remove)

            new_dict[key] = filtered_value

        return new_dict

    elif isinstance(data, list):
        return [remove_fields(item, fields_to_remove) for item in data]

    else:
        return data


def main():
    if len(sys.argv) < 4:
        print("Usage: python filter_json.py input.json output.json field1 field2 ...")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    fields_to_remove = sys.argv[3:]

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = remove_fields(data, fields_to_remove)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
