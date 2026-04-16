"""Comparison helpers for pragmatic analysis JSON outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple, set)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        try:
            if value != value:  # NaN
                return None
        except Exception:
            pass

    if isinstance(value, str):
        return " ".join(value.split())

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)

    return value


def _json_safe(value: Any) -> Any:
    value = _normalize_scalar(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _clean_value_for_comparison(value: Any) -> Any:
    value = _normalize_scalar(value)
    if isinstance(value, (dict, list)):
        return _json_safe(value)
    return value


def _signature(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=str)


def _load_document(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Pragmatic analysis output must be a JSON object: {path}")
    return loaded


def _normalize_evaluations(document: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = document.get("evaluation")
    if isinstance(evaluation, list):
        return [item for item in evaluation if isinstance(item, dict)]

    analysis = document.get("analysis")
    if isinstance(analysis, list):
        return [
            {
                "book": document.get("book"),
                "chapter": document.get("chapter"),
                "pragmatic_goal": document.get("pragmatic_goal"),
                "analysis": analysis,
            }
        ]

    return []


def _verse_key(evaluation: dict[str, Any], verse_block: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        evaluation.get("book"),
        evaluation.get("chapter"),
        verse_block.get("verse"),
    )


def _group_verses(document: dict[str, Any]) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    verses: dict[tuple[Any, Any, Any], dict[str, Any]] = {}

    for evaluation in _normalize_evaluations(document):
        for verse_block in evaluation.get("analysis") or []:
            if not isinstance(verse_block, dict):
                continue
            key = _verse_key(evaluation, verse_block)
            verses[key] = {
                "book": evaluation.get("book"),
                "chapter": evaluation.get("chapter"),
                "pragmatic_goal": evaluation.get("pragmatic_goal"),
                "verse": verse_block.get("verse"),
                "biblical_text": verse_block.get("biblical_text"),
                "translation": verse_block.get("translation"),
                "analysis": verse_block.get("analysis", []),
            }

    return verses


def _compare_analysis_items(left_items: Iterable[Any], right_items: Iterable[Any]) -> dict[str, Any]:
    left_items = [item for item in left_items if isinstance(item, dict)]
    right_items = [item for item in right_items if isinstance(item, dict)]

    left_signatures = Counter(_signature(item) for item in left_items)
    right_signatures = Counter(_signature(item) for item in right_items)

    left_only: list[dict[str, Any]] = []
    right_only: list[dict[str, Any]] = []

    left_lookup = {_signature(item): _json_safe(item) for item in left_items}
    right_lookup = {_signature(item): _json_safe(item) for item in right_items}

    for signature, count in (left_signatures - right_signatures).items():
        left_only.extend([left_lookup[signature]] * count)

    for signature, count in (right_signatures - left_signatures).items():
        right_only.extend([right_lookup[signature]] * count)

    return {
        "comparison_mode": "multiset",
        "matched_items": int(sum((left_signatures & right_signatures).values())),
        "left_only_items": left_only,
        "right_only_items": right_only,
    }


def compare_pragmatic_analysis_documents(left_document: dict[str, Any], right_document: dict[str, Any]) -> dict[str, Any]:
    left_verses = _group_verses(left_document)
    right_verses = _group_verses(right_document)

    verse_keys = sorted(set(left_verses) | set(right_verses), key=lambda value: str(value))
    verse_reports: list[dict[str, Any]] = []

    for key in verse_keys:
        left_verse = left_verses.get(key)
        right_verse = right_verses.get(key)

        if left_verse is None or right_verse is None:
            verse_reports.append(
                {
                    "book": key[0],
                    "chapter": key[1],
                    "verse": key[2],
                    "status": "left_only" if right_verse is None else "right_only",
                    "left_present": left_verse is not None,
                    "right_present": right_verse is not None,
                    "field_diffs": [],
                    "analysis_comparison": {
                        "comparison_mode": "multiset",
                        "matched_items": 0,
                        "left_only_items": _json_safe(left_verse.get("analysis", [])) if left_verse else [],
                        "right_only_items": _json_safe(right_verse.get("analysis", [])) if right_verse else [],
                    },
                }
            )
            continue

        field_diffs = []
        for field in ("biblical_text", "translation"):
            left_value = _clean_value_for_comparison(left_verse.get(field))
            right_value = _clean_value_for_comparison(right_verse.get(field))
            if left_value != right_value:
                field_diffs.append(
                    {
                        "field": field,
                        "left": left_value,
                        "right": right_value,
                    }
                )

        analysis_comparison = _compare_analysis_items(left_verse.get("analysis", []), right_verse.get("analysis", []))

        verse_reports.append(
            {
                "book": key[0],
                "chapter": key[1],
                "verse": key[2],
                "status": "matched",
                "left_present": True,
                "right_present": True,
                "field_diffs": field_diffs,
                "analysis_comparison": analysis_comparison,
            }
        )

    translation_diffs = []
    left_translation = left_document.get("translation") or {}
    right_translation = right_document.get("translation") or {}
    for field in ("title", "language"):
        left_value = _clean_value_for_comparison(left_translation.get(field))
        right_value = _clean_value_for_comparison(right_translation.get(field))
        if left_value != right_value:
            translation_diffs.append(
                {
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                }
            )

    return {
        "left_path": left_document.get("source_path"),
        "right_path": right_document.get("source_path"),
        "left_translation": _json_safe(left_document.get("translation", {})),
        "right_translation": _json_safe(right_document.get("translation", {})),
        "translation_diffs": translation_diffs,
        "left_evaluation_count": len(_normalize_evaluations(left_document)),
        "right_evaluation_count": len(_normalize_evaluations(right_document)),
        "verse_report_count": len(verse_reports),
        "verse_reports": verse_reports,
    }


def compare_pragmatic_analysis_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_document = _load_document(left_path)
    right_document = _load_document(right_path)
    left_document["source_path"] = str(left_path)
    right_document["source_path"] = str(right_path)
    return compare_pragmatic_analysis_documents(left_document, right_document)


def comparison_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for verse in report.get("verse_reports", []):
        analysis = verse.get("analysis_comparison", {})
        rows.append(
            {
                "book": verse.get("book"),
                "chapter": verse.get("chapter"),
                "verse": verse.get("verse"),
                "status": verse.get("status"),
                "field_diffs": len(verse.get("field_diffs", [])),
                "matched_items": analysis.get("matched_items", 0),
                "left_only_items": len(analysis.get("left_only_items", [])),
                "right_only_items": len(analysis.get("right_only_items", [])),
            }
        )
    return rows


def print_comparison_summary(report: dict[str, Any]) -> None:
    verse_reports = report.get("verse_reports", [])
    mismatched = [
        verse
        for verse in verse_reports
        if verse.get("status") != "matched"
        or verse.get("field_diffs")
        or verse.get("analysis_comparison", {}).get("left_only_items")
        or verse.get("analysis_comparison", {}).get("right_only_items")
    ]

    print(f"Compared {len(verse_reports)} verse blocks.")
    print(f"Left evaluation blocks: {report.get('left_evaluation_count', 0)}")
    print(f"Right evaluation blocks: {report.get('right_evaluation_count', 0)}")
    print(f"Verse blocks with differences: {len(mismatched)}")

    translation_diffs = report.get("translation_diffs", [])
    if translation_diffs:
        print("Translation metadata differences:")
        for diff in translation_diffs:
            print(f"  - {diff.get('field')}: left={diff.get('left')} | right={diff.get('right')}")
