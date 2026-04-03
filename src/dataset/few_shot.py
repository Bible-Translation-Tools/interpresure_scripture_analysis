"""Few-shot example loading and rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .schema import json_safe


PROSE_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".adoc", ".org", ".text", ".prose"}


def detect_example_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in PROSE_SUFFIXES or not suffix:
        return "prose"
    return "prose"


def _load_json_example(path: Path) -> tuple[Any, str | None]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    row_count: str | None = None

    if isinstance(loaded, dict):
        rows = loaded.get("rows")
        if isinstance(rows, list):
            row_count = str(len(rows))
    elif isinstance(loaded, list):
        row_count = str(len(loaded))

    return loaded, row_count


def load_few_shot_example(path: Path) -> dict[str, Any]:
    example_format = detect_example_format(path)

    if example_format == "csv":
        df = pd.read_csv(path)
        return {
            "path": path,
            "format": example_format,
            "row_count": len(df),
            "columns": [str(column) for column in df.columns.tolist()],
            "content": df.to_csv(index=False),
        }

    if example_format == "json":
        loaded, row_count = _load_json_example(path)
        content = json.dumps(json_safe(loaded), indent=2, ensure_ascii=False)
        columns: list[str] = []
        if isinstance(loaded, dict):
            rows = loaded.get("rows")
            if isinstance(rows, list) and rows and all(isinstance(item, dict) for item in rows):
                first_row = rows[0]
                columns = [str(column) for column in first_row.keys()]
        elif isinstance(loaded, list) and loaded and all(isinstance(item, dict) for item in loaded):
            columns = [str(column) for column in loaded[0].keys()]
        return {
            "path": path,
            "format": example_format,
            "row_count": row_count,
            "columns": columns,
            "content": content,
        }

    content = path.read_text(encoding="utf-8")
    line_count = len(content.splitlines())
    return {
        "path": path,
        "format": example_format,
        "line_count": line_count,
        "content": content,
    }


def render_few_shot_examples(example_paths: Iterable[Path]) -> str:
    paths = [Path(path) for path in example_paths]
    if not paths:
        return ""

    blocks = [
        "FEW-SHOT EXAMPLES:",
        "Each file below is one expert chapter example.",
        "The example files may use different columns from the target schema.",
        "Use them to learn the annotation style, span granularity, and label conventions.",
        "Do not assume columns from one example exist in the current target schema.",
        "",
    ]

    for index, path in enumerate(paths, start=1):
        example = load_few_shot_example(path)
        blocks.append(f"Example {index}: {path}")
        blocks.append(f"Format: {example['format']}")
        if example.get("row_count") is not None:
            blocks.append(f"Rows: {example['row_count']}")
        if example.get("line_count") is not None:
            blocks.append(f"Lines: {example['line_count']}")
        columns = example.get("columns") or []
        if columns:
            blocks.append("Columns: " + ", ".join(columns))
        blocks.append("")
        fence_language = "json" if example["format"] == "json" else ("csv" if example["format"] == "csv" else "text")
        blocks.append(f"```{fence_language}")
        blocks.append(str(example["content"]).rstrip())
        blocks.append("```")
        blocks.append("")

    return "\n".join(blocks).strip()
