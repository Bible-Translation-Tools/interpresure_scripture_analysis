"""Comparison helpers for pragmatic analysis JSON outputs."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Literal
from pathlib import Path
from typing import Any, Iterable

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, ConfigDict, Field

from model.config import get_config_for_model


class ComparisonJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    better_side: Literal["left", "right", "tie"] = Field(
        ...,
        description="Which side better analyzed the verse. Use tie when neither side clearly wins.",
    )
    better_is_left: bool = Field(
        ...,
        description="True if the left analysis is better, false if the right analysis is better or the result is a tie.",
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in the comparison judgment from 0 to 100.",
    )
    reasoning: str = Field(
        ...,
        description="Markdown explanation of why the better side was chosen, or why the comparison is a tie.",
    )


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


def _safe_first_analysis_item(verse_block: dict[str, Any]) -> dict[str, Any]:
    analysis_items = [item for item in verse_block.get("analysis", []) if isinstance(item, dict)]
    if analysis_items:
        item = analysis_items[0]
        return {
            "model": item.get("model", ""),
            "score": _clean_value_for_comparison(item.get("score")),
            "confidence": _clean_value_for_comparison(item.get("confidence")),
            "reasoning": item.get("reasoning", "") or item.get("model_analysis", ""),
            "strengths": item.get("strengths", ""),
            "weaknesses": item.get("weaknesses", ""),
            "suggestions": item.get("suggestions", ""),
            "model_analysis": item.get("model_analysis", ""),
            "raw": _json_safe(item),
        }

    return {
        "model": "",
        "score": None,
        "confidence": None,
        "reasoning": "",
        "strengths": "",
        "weaknesses": "",
        "suggestions": "",
        "model_analysis": "",
        "raw": {},
    }


def _compare_system_message() -> str:
    return (
        "You are a careful evaluator comparing two pragmatic analyses of the same verse.\n"
        "Your job is to decide which analysis better explains how the translation handles the pragmatic dynamics of the original text.\n"
        "Favor analyses that are more faithful to the verse, better grounded in evidence, more complete, and clearer.\n"
        "If better_side is 'left', better_is_left must be true. If better_side is 'right' or 'tie', better_is_left must be false.\n"
        "Return only the requested JSON object."
    )


def _build_compare_prompt(verse_payload: dict[str, Any]) -> str:
    return (
        "Compare the two analyses below and decide which one is better.\n\n"
        "Criteria:\n"
        "- Which analysis more accurately identifies the pragmatic meaning.\n"
        "- Which analysis better explains how the translation preserves or misses those dynamics.\n"
        "- Which analysis is better supported by the verse text and the provided analysis evidence.\n"
        "- If the two analyses are effectively equivalent, choose tie.\n\n"
        f"Verse payload:\n{json.dumps(verse_payload, ensure_ascii=False, indent=2, default=str)}\n\n"
        "Output the comparison judgment as JSON."
    )


async def _compare_with_llm(
    *,
    verse_payload: dict[str, Any],
    comparison_model: str = "gpt-5-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> ComparisonJudgment:
    config = get_config_for_model(comparison_model)
    resolved_api_key = api_key if api_key is not None else config.get("key")
    resolved_base_url = base_url if base_url is not None else config.get("base_url")

    if resolved_api_key is None:
        raise ValueError(
            f"No API key available for comparison model '{comparison_model}'. "
            "Set the corresponding environment variable or pass api_key explicitly."
        )

    client_kwargs: dict[str, Any] = {
        "api_type": "openai",
        "model": config["model"],
        "api_key": resolved_api_key,
        "model_info": ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
        "timeout": 90,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "comparison_judgment",
                "strict": True,
                "schema": ComparisonJudgment.model_json_schema(),
            },
        },
    }
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url

    client = OpenAIChatCompletionClient(**client_kwargs)
    judge = AssistantAgent(
        name="PRAGMATIC_COMPARISON_JUDGE",
        model_client=client,
        system_message=_compare_system_message(),
    )
    result = await judge.run(task=_build_compare_prompt(verse_payload))
    raw = result.messages[-1].content
    if isinstance(raw, dict):
        return ComparisonJudgment(**raw)
    cleaned = str(raw).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return ComparisonJudgment(**json.loads(cleaned))


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


def _build_verse_report(
    key: tuple[Any, Any, Any],
    left_verse: dict[str, Any] | None,
    right_verse: dict[str, Any] | None,
    comparison_judgment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    left_primary = _safe_first_analysis_item(left_verse or {})
    right_primary = _safe_first_analysis_item(right_verse or {})

    if left_verse is None or right_verse is None:
        status = "left_only" if right_verse is None else "right_only"
        comparison_judgment = comparison_judgment or {
            "better_side": "tie",
            "better_is_left": False,
            "confidence": 0,
            "reasoning": "One side is missing this verse, so the comparison cannot be judged.",
        }
    else:
        status = "matched"
        comparison_judgment = comparison_judgment or {}

    verse_report = {
        "book": key[0],
        "chapter": key[1],
        "verse": key[2],
        "status": status,
        "translation_text": (left_verse or right_verse or {}).get("translation", ""),
        "biblical_text": (left_verse or right_verse or {}).get("biblical_text", ""),
        "left_present": left_verse is not None,
        "right_present": right_verse is not None,
        "left_model": left_primary.get("model", ""),
        "right_model": right_primary.get("model", ""),
        "left_analysis": left_primary,
        "right_analysis": right_primary,
        "left_analysis_items": _json_safe(left_verse.get("analysis", [])) if left_verse else [],
        "right_analysis_items": _json_safe(right_verse.get("analysis", [])) if right_verse else [],
        "field_diffs": [],
        "analysis_comparison": _compare_analysis_items(
            left_verse.get("analysis", []) if left_verse else [],
            right_verse.get("analysis", []) if right_verse else [],
        ),
        "comparison_judgment": _json_safe(comparison_judgment),
        "highlight_side": comparison_judgment.get("better_side") if comparison_judgment else None,
    }

    if left_verse is not None and right_verse is not None:
        for field in ("biblical_text", "translation"):
            left_value = _clean_value_for_comparison(left_verse.get(field))
            right_value = _clean_value_for_comparison(right_verse.get(field))
            if left_value != right_value:
                verse_report["field_diffs"].append(
                    {
                        "field": field,
                        "left": left_value,
                        "right": right_value,
                    }
                )

    return verse_report


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


async def compare_pragmatic_analysis_documents_async(
    left_document: dict[str, Any],
    right_document: dict[str, Any],
    *,
    comparison_model: str = "gpt-5-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    left_verses = _group_verses(left_document)
    right_verses = _group_verses(right_document)

    verse_keys = sorted(set(left_verses) | set(right_verses), key=lambda value: str(value))
    verse_reports: list[dict[str, Any]] = []

    for key in verse_keys:
        left_verse = left_verses.get(key)
        right_verse = right_verses.get(key)
        verse_payload = {
            "book": key[0],
            "chapter": key[1],
            "verse": key[2],
            "translation_text": (left_verse or right_verse or {}).get("translation", ""),
            "biblical_text": (left_verse or right_verse or {}).get("biblical_text", ""),
            "left": _safe_first_analysis_item(left_verse or {}),
            "right": _safe_first_analysis_item(right_verse or {}),
        }

        if left_verse is not None and right_verse is not None:
            verse_payload["left"]["analysis_items"] = _json_safe(left_verse.get("analysis", []))
            verse_payload["right"]["analysis_items"] = _json_safe(right_verse.get("analysis", []))
            judgment = await _compare_with_llm(
                verse_payload=verse_payload,
                comparison_model=comparison_model,
                api_key=api_key,
                base_url=base_url,
            )
            judgment_dict = judgment.model_dump()
        else:
            judgment_dict = {
                "better_side": "tie",
                "better_is_left": False,
                "confidence": 0,
                "reasoning": "One side is missing this verse, so the comparison cannot be judged.",
            }

        verse_reports.append(_build_verse_report(key, left_verse, right_verse, judgment_dict))

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
        "comparison_model": comparison_model,
        "left_translation": _json_safe(left_document.get("translation", {})),
        "right_translation": _json_safe(right_document.get("translation", {})),
        "translation_diffs": translation_diffs,
        "left_evaluation_count": len(_normalize_evaluations(left_document)),
        "right_evaluation_count": len(_normalize_evaluations(right_document)),
        "verse_report_count": len(verse_reports),
        "verse_reports": verse_reports,
    }


def compare_pragmatic_analysis_documents(
    left_document: dict[str, Any],
    right_document: dict[str, Any],
    *,
    comparison_model: str = "gpt-5-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        compare_pragmatic_analysis_documents_async(
            left_document,
            right_document,
            comparison_model=comparison_model,
            api_key=api_key,
            base_url=base_url,
        )
    )


def compare_pragmatic_analysis_files(
    left_path: Path,
    right_path: Path,
    *,
    comparison_model: str = "gpt-5-mini",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    left_document = _load_document(left_path)
    right_document = _load_document(right_path)
    left_document["source_path"] = str(left_path)
    right_document["source_path"] = str(right_path)
    return compare_pragmatic_analysis_documents(
        left_document,
        right_document,
        comparison_model=comparison_model,
        api_key=api_key,
        base_url=base_url,
    )


def comparison_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for verse in report.get("verse_reports", []):
        analysis = verse.get("analysis_comparison", {})
        judgment = verse.get("comparison_judgment", {}) or {}
        rows.append(
            {
                "book": verse.get("book"),
                "chapter": verse.get("chapter"),
                "verse": verse.get("verse"),
                "status": verse.get("status"),
                "better_side": judgment.get("better_side"),
                "judge_confidence": judgment.get("confidence"),
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
    print(f"Comparison model: {report.get('comparison_model', 'gpt-5-mini')}")
    print(f"Left evaluation blocks: {report.get('left_evaluation_count', 0)}")
    print(f"Right evaluation blocks: {report.get('right_evaluation_count', 0)}")
    print(f"Verse blocks with differences: {len(mismatched)}")

    judge_counts = Counter(
        verse.get("comparison_judgment", {}).get("better_side", "tie") for verse in verse_reports
    )
    if judge_counts:
        print(
            "Comparison judgments: "
            + ", ".join(f"{side}={count}" for side, count in sorted(judge_counts.items()))
        )

    translation_diffs = report.get("translation_diffs", [])
    if translation_diffs:
        print("Translation metadata differences:")
        for diff in translation_diffs:
            print(f"  - {diff.get('field')}: left={diff.get('left')} | right={diff.get('right')}")
