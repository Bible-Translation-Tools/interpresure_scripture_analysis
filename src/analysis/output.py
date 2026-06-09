"""Run directory writer.

Produces an output directory in the scripture-analysis-api upload format::

    {output_dir}/{timestamp}_{BOOK}_{chapter}/
    ├── repo.json
    ├── run.json
    └── {BOOK}_{chapter}.json   ← array of AnalysisItems

The scope file contains all items for the chapter in the order they were
produced: run metadata → discourse map → verse items → chapter summary item.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .repo_info import RepoInfo
from .schemas import AnalysisItem


# ---------------------------------------------------------------------------
# Git helpers (kept for test compatibility)
# ---------------------------------------------------------------------------


def _git_commit_sha(repo_root: Path | None = None) -> str:
    """Return the current git HEAD SHA, or ``'unknown'`` if unavailable."""
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.decode().strip()
    except Exception:
        return "unknown"


def _git_remote_url(repo_root: Path | None = None) -> str:
    """Return the origin remote URL, or ``''`` if unavailable."""
    try:
        result = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.decode().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# RunWriter
# ---------------------------------------------------------------------------


class RunWriter:
    """Writes a completed analysis run to an output directory.

    The directory layout matches what the scripture-analysis-api CLI expects:
    ``repo.json``, ``run.json``, and one scope JSON file per (book, chapter).

    ``repo.json`` is sourced from the translation's ``repo_info.yaml``.
    ``run.json`` commit_sha is the last git commit that touched the USFM file.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        repo_info: RepoInfo,
        commit_sha: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.repo_info = repo_info
        self.commit_sha = commit_sha

    def write(
        self,
        *,
        book: str,
        chapter: int,
        items: list[AnalysisItem],
        timestamp: datetime | None = None,
    ) -> Path:
        """Write the run directory and return its path.

        Args:
            book: USFM book code (e.g. ``"PHM"``).
            chapter: Chapter number.
            items: All ``AnalysisItem`` objects for this scope, in order.
            timestamp: Optional run timestamp; defaults to now (UTC).

        Returns:
            The path to the created run directory.
        """
        ts = timestamp or datetime.now(tz=timezone.utc)
        ts_slug = ts.strftime("%Y%m%dT%H%M%S")
        book_upper = book.strip().upper()

        run_dir = self.output_dir / f"{ts_slug}_{book_upper}_{chapter}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_repo_json(run_dir)
        self._write_run_json(run_dir)
        self._write_scope_json(run_dir, book_upper, chapter, items)

        return run_dir

    def _write_repo_json(self, run_dir: Path) -> None:
        repo = {
            "repo_id": self.repo_info.repo_id,
            "name": self.repo_info.name,
            "git_url": self.repo_info.git_url,
        }
        _write_json(run_dir / "repo.json", repo)

    def _write_run_json(self, run_dir: Path) -> None:
        run = {"commit_sha": self.commit_sha}
        _write_json(run_dir / "run.json", run)

    def _write_scope_json(
        self,
        run_dir: Path,
        book: str,
        chapter: int,
        items: list[AnalysisItem],
    ) -> None:
        scope_file = run_dir / f"{book}_{chapter}.json"
        payload = [item.model_dump(exclude_none=False) for item in items]
        _write_json(scope_file, payload)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Item assembly helpers
# ---------------------------------------------------------------------------


def build_scope_items(
    *,
    book: str,
    chapter: int,
    run_metadata_obs,
    discourse_map_obs,
    verse_observations: list[tuple[int, Any]],
    chapter_summary_obs,
) -> list[AnalysisItem]:
    """Assemble all items for a scope file in canonical order.

    Order: run metadata → discourse map → verse items (ascending) → chapter summary.

    Args:
        book: USFM book code.
        chapter: Chapter number.
        run_metadata_obs: ``AnalysisRunMetadataObservation`` instance.
        discourse_map_obs: ``DiscourseMapObservation`` instance.
        verse_observations: List of ``(verse_number, observation)`` tuples.
        chapter_summary_obs: Observation for the chapter-level summary
            (same type as verse observations but anchored at chapter level).

    Returns:
        Ordered list of ``AnalysisItem`` objects ready for serialization.
    """
    book_upper = book.strip().upper()
    chapter_anchor = f"{book_upper} {chapter}"
    items: list[AnalysisItem] = []

    # 1. Run metadata
    items.append(
        AnalysisItem.from_observation(
            run_metadata_obs,
            book=book_upper,
            chapter=chapter,
            anchor=chapter_anchor,
            anchor_level="chapter",
        )
    )

    # 2. Discourse map
    items.append(
        AnalysisItem.from_observation(
            discourse_map_obs,
            book=book_upper,
            chapter=chapter,
            anchor=chapter_anchor,
            anchor_level="chapter",
        )
    )

    # 3. Verse items (sorted by verse number)
    for verse_num, obs in sorted(verse_observations, key=lambda t: t[0]):
        items.append(
            AnalysisItem.from_observation(
                obs,
                book=book_upper,
                chapter=chapter,
                anchor=f"{book_upper} {chapter}:{verse_num}",
                anchor_level="verse",
            )
        )

    # 4. Chapter summary
    items.append(
        AnalysisItem.from_observation(
            chapter_summary_obs,
            book=book_upper,
            chapter=chapter,
            anchor=chapter_anchor,
            anchor_level="chapter",
        )
    )

    return items
