"""Scripture translation repository metadata.

Each language directory in ``lang/`` may contain a ``repo_info.yaml`` that
identifies the translation repository being analyzed.  This metadata drives
the ``repo.json`` and ``run.json`` files in the scripture-analysis-api upload
format, anchoring analysis output to the exact scripture source.

Expected ``repo_info.yaml`` format::

    repo_id: unfoldingword-en-ult
    name: Unlocked Literal Bible
    language: en
    git_url: https://git.door43.org/unfoldingWord/en_ult

Fields:
    repo_id  — stable unique identifier used as the primary key in the API
    name     — human-readable name of the translation
    language — BCP 47 language tag (should match the directory name)
    git_url  — canonical remote URL of the translation repository
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RepoInfo:
    repo_id: str
    name: str
    language: str
    git_url: str


def load_repo_info(usfm_root: Path, language: str) -> RepoInfo:
    """Load ``RepoInfo`` from ``{usfm_root}/{language}/repo_info.yaml``.

    Raises:
        FileNotFoundError: if ``repo_info.yaml`` is absent.
        ValueError: if required fields are missing or YAML is malformed.
    """
    yaml_path = Path(usfm_root) / language / "repo_info.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"repo_info.yaml not found: {yaml_path}\n"
            f"Create it with fields: repo_id, name, language, git_url"
        )

    if yaml is None:
        raise ImportError("PyYAML is required to load repo_info.yaml. Run: pip install pyyaml")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"repo_info.yaml must be a YAML mapping: {yaml_path}")

    missing = [f for f in ("repo_id", "name", "language", "git_url") if not raw.get(f)]
    if missing:
        raise ValueError(
            f"repo_info.yaml is missing required field(s): {', '.join(missing)}\n"
            f"File: {yaml_path}"
        )

    return RepoInfo(
        repo_id=str(raw["repo_id"]),
        name=str(raw["name"]),
        language=str(raw["language"]),
        git_url=str(raw["git_url"]),
    )


def get_usfm_commit_sha(usfm_file: Path) -> str:
    """Return the last git commit SHA that touched ``usfm_file``.

    Uses ``git log -1 --format=%H -- <file>`` so the SHA reflects the exact
    version of the scripture text being analyzed, not the analysis pipeline.
    Falls back to ``"unknown"`` if the file is not in a git repo or git is
    unavailable.
    """
    try:
        result = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(usfm_file.name)],
            cwd=str(usfm_file.parent),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        sha = result.decode().strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"
