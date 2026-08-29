"""
generate_classes.py
====================
Reads data.yaml from the dataset and writes classes.txt (one class name
per line, in index order) at the project root. Run this once after
placing the dataset -- or just copy classes.txt from the earlier CNN
project, since it's the same dataset.

Handles both common YOLO data.yaml "names" formats:
    names: [almond, apple, ...]          <- list style
    names:
      0: almond
      1: apple
      ...                                 <- dict style (index: name)

Usage:
    python scripts/generate_classes.py
"""

import os
import yaml

DATA_YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "LVIS_Fruits_And_Vegetables", "data.yaml",
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classes.txt"
)


def main():
    if not os.path.exists(DATA_YAML_PATH):
        raise SystemExit(f"Could not find {DATA_YAML_PATH}. Place the dataset first (see README.md).")

    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        raise SystemExit("Could not find a 'names' entry in data.yaml.")

    if isinstance(names, dict):
        # dict style: {0: "almond", 1: "apple", ...} -- sort by the
        # integer index so the output order matches the class ids used
        # in the label .txt files.
        ordered_names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        ordered_names = names
    else:
        raise SystemExit(f"Unrecognized 'names' format in data.yaml: {type(names)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for name in ordered_names:
            f.write(str(name) + "\n")

    print(f"Wrote {len(ordered_names)} class names to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
