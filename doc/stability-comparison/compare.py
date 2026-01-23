#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn"]
# ///
#
import itertools
import json
from pathlib import Path

import sklearn.metrics.pairwise

data: dict[str, list[float]] = {}

for file in sorted(Path(".").glob("*.json")):
    f_data = json.loads(file.read_text(encoding="utf-8"))
    data[file.stem] = f_data["data"][0]["embedding"]

names = list(data.keys())

distances: dict[tuple[str, str], float] = {}
for a, b in itertools.combinations(names, 2):
    dist = sklearn.metrics.pairwise.euclidean_distances([data[a]], [data[b]])[
        0
    ][0]
    distances[a, b] = dist
    distances[b, a] = dist

# Render markdown table with aligned columns
col_width = max(len(n) for n in names)
col_width = max(col_width, 6)  # at least wide enough for "0.0000"


def pad(s: str) -> str:
    return s.rjust(col_width)


header = (
    f"| {''.ljust(col_width)} | " + " | ".join(pad(n) for n in names) + " |"
)
separator = (
    f"| {'-' * col_width} | "
    + " | ".join("-" * (col_width - 1) + ":" for _ in names)
    + " |"
)
print(header)
print(separator)
for i, row in enumerate(names):
    cells = []
    for j, col in enumerate(names):
        if j <= i:
            cells.append(pad(""))
        else:
            cells.append(pad(f"{distances[row, col]:.4f}"))
    print(f"| {row.ljust(col_width)} | " + " | ".join(cells) + " |")
