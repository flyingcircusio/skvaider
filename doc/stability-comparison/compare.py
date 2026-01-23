#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn"]
# ///
#
import itertools
import json
import math
import re
from pathlib import Path

import sklearn.metrics.pairwise

data: dict[str, list[float]] = {}

for file in sorted(Path(".").glob("*.json")):
    f_data = json.loads(file.read_text(encoding="utf-8"))
    vec = f_data["data"][0]["embedding"]
    norm = math.sqrt(sum(x * x for x in vec))
    print(f"{file.stem}: L2 norm = {norm:.6f}")
    data[file.stem] = [x / norm for x in vec]

names = list(data.keys())

PRECISIONS = re.compile(r"(Q[0-9]+_[0-9A-Za-z_]+|F[0-9]+)")


def precision(name: str) -> str | None:
    m = PRECISIONS.search(name)
    return m.group(1) if m else None


euclidean: dict[tuple[str, str], float] = {}
cosine: dict[tuple[str, str], float] = {}
for a, b in itertools.combinations(names, 2):
    ed = sklearn.metrics.pairwise.euclidean_distances([data[a]], [data[b]])[0][
        0
    ]
    euclidean[a, b] = ed
    euclidean[b, a] = ed
    cs = sklearn.metrics.pairwise.cosine_similarity([data[a]], [data[b]])[0][0]
    cosine[a, b] = cs
    cosine[b, a] = cs


output: list[str] = []


def print_table(
    title: str,
    values: dict[tuple[str, str], float],
    fmt: str = ".4f",
    suffix: str = "",
) -> None:
    # Bold adds 4 chars of non-visible width
    col_width = max(len(n) for n in names)
    col_width = max(col_width, 10)  # wide enough for "**0.0000**"

    def pad(s: str, extra: int = 0) -> str:
        return s.rjust(col_width + extra)

    output.append(f"## {title}")
    output.append("")
    header = (
        f"| {''.ljust(col_width)} | " + " | ".join(pad(n) for n in names) + " |"
    )
    separator = (
        f"| {'-' * col_width} | "
        + " | ".join("-" * (col_width - 1) + ":" for _ in names)
        + " |"
    )
    output.append(header)
    output.append(separator)
    for i, row in enumerate(names[:-1]):
        cells = []
        for j, col in enumerate(names):
            if j <= i:
                cells.append(pad(""))
            else:
                val = f"{values[row, col]:{fmt}}{suffix}"
                rp = precision(row)
                cp = precision(col)
                if rp and cp and rp == cp:
                    val = f"**{val}**"
                    cells.append(pad(val, 4))
                else:
                    cells.append(pad(val))
        output.append(f"| {row.ljust(col_width)} | " + " | ".join(cells) + " |")
    output.append("")


angles: dict[tuple[str, str], float] = {
    k: math.degrees(math.acos(min(1.0, max(-1.0, v))))
    for k, v in cosine.items()
}

print_table("Euclidean Distance", euclidean)
print_table("Cosine Similarity", cosine)
print_table("Angle", angles, fmt=".1f", suffix="°")

BEGIN = "<!-- BEGIN comparison -->"
END = "<!-- END comparison -->"

readme = Path("README.md")
content = readme.read_text(encoding="utf-8")
before = content[: content.index(BEGIN) + len(BEGIN)]
after = content[content.index(END) :]
readme.write_text(
    before + "\n" + "\n".join(output) + "\n" + after, encoding="utf-8"
)
