"""Tests for analysis/output.py — RunWriter and scope item assembly."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from analysis.output import RunWriter, _git_commit_sha, _git_remote_url, build_scope_items
from analysis.repo_info import RepoInfo
from analysis.schemas import AnalysisItem


# ---------------------------------------------------------------------------
# _git_commit_sha
# ---------------------------------------------------------------------------


class TestGitCommitSha:
    def test_returns_sha_on_success(self):
        with patch("subprocess.check_output", return_value=b"abc1234def5678\n"):
            sha = _git_commit_sha()
        assert sha == "abc1234def5678"

    def test_returns_unknown_on_failure(self):
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git")):
            sha = _git_commit_sha()
        assert sha == "unknown"

    def test_returns_unknown_on_timeout(self):
        with patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired("git", 5)):
            sha = _git_commit_sha()
        assert sha == "unknown"


# ---------------------------------------------------------------------------
# _git_remote_url
# ---------------------------------------------------------------------------


class TestGitRemoteUrl:
    def test_returns_url_on_success(self):
        with patch(
            "subprocess.check_output",
            return_value=b"https://github.com/example/repo.git\n",
        ):
            url = _git_remote_url()
        assert url == "https://github.com/example/repo.git"

    def test_returns_empty_on_failure(self):
        with patch("subprocess.check_output", side_effect=Exception("no remote")):
            url = _git_remote_url()
        assert url == ""


# ---------------------------------------------------------------------------
# build_scope_items — canonical ordering and anchor assignment
# ---------------------------------------------------------------------------


class TestBuildScopeItems:
    def test_item_order(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[(10, basic_suggestion), (5, basic_suggestion)],
            chapter_summary_obs=basic_suggestion,
        )
        types = [i.type for i in items]
        assert types[0] == "analysis_run_metadata"
        assert types[1] == "discourse_map"
        assert types[-1] == "interpresure_suggestions"  # chapter summary last
        assert len(items) == 5  # meta + map + 2 verses + summary

    def test_verses_sorted_ascending(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[(25, basic_suggestion), (3, basic_suggestion), (10, basic_suggestion)],
            chapter_summary_obs=basic_suggestion,
        )
        verse_items = [i for i in items if i.anchor_level == "verse"]
        anchors = [i.anchor for i in verse_items]
        assert anchors == ["PHM 1:3", "PHM 1:10", "PHM 1:25"]

    def test_verse_anchor_format(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[(7, basic_suggestion)],
            chapter_summary_obs=basic_suggestion,
        )
        verse_item = next(i for i in items if i.anchor_level == "verse")
        assert verse_item.anchor == "PHM 1:7"

    def test_chapter_anchor_format(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        chapter_items = [i for i in items if i.anchor_level == "chapter"]
        anchors = {i.anchor for i in chapter_items}
        assert "PHM 1" in anchors

    def test_book_uppercase(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="phm",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        for item in items:
            if item.book:
                assert item.book == "PHM"

    def test_empty_verse_observations(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        # metadata + discourse + chapter summary = 3
        assert len(items) == 3

    def test_all_items_are_analysis_items(self, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM",
            chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[(1, basic_suggestion)],
            chapter_summary_obs=basic_suggestion,
        )
        for item in items:
            assert isinstance(item, AnalysisItem)


# ---------------------------------------------------------------------------
# RunWriter.write — file creation and content
# ---------------------------------------------------------------------------


class TestRunWriter:
    @pytest.fixture
    def writer(self, tmp_path):
        return RunWriter(
            output_dir=tmp_path,
            repo_info=RepoInfo(
                repo_id="test-repo",
                name="Test Repo",
                language="en",
                git_url="https://github.com/test/repo.git",
            ),
            commit_sha="abc1234",
        )

    def test_creates_run_directory(self, writer, tmp_path, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM", chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        run_dir = writer.write(book="PHM", chapter=1, items=items, timestamp=ts)

        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.name == "20260601T120000_PHM_1"

    def test_writes_repo_json(self, writer, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM", chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        run_dir = writer.write(book="PHM", chapter=1, items=items)

        repo_file = run_dir / "repo.json"
        assert repo_file.exists()
        data = json.loads(repo_file.read_text())
        assert data["repo_id"] == "test-repo"
        assert data["name"] == "Test Repo"
        assert data["git_url"] == "https://github.com/test/repo.git"

    def test_writes_run_json(self, writer, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM", chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        run_dir = writer.write(book="PHM", chapter=1, items=items)

        run_file = run_dir / "run.json"
        assert run_file.exists()
        data = json.loads(run_file.read_text())
        assert data["commit_sha"] == "abc1234"

    def test_writes_scope_json(self, writer, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM", chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[(5, basic_suggestion)],
            chapter_summary_obs=basic_suggestion,
        )
        run_dir = writer.write(book="PHM", chapter=1, items=items)

        scope_file = run_dir / "PHM_1.json"
        assert scope_file.exists()
        data = json.loads(scope_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 4  # meta + map + 1 verse + summary

    def test_scope_json_items_have_required_fields(self, writer, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="PHM", chapter=1,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        run_dir = writer.write(book="PHM", chapter=1, items=items)
        data = json.loads((run_dir / "PHM_1.json").read_text())

        for item in data:
            assert "type" in item
            assert "version" in item
            assert "anchor_level" in item
            assert "observation" in item

    def test_creates_parent_dirs(self, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        writer = RunWriter(
            output_dir=deep_dir,
            repo_info=RepoInfo(repo_id="r", name="R", language="en", git_url=""),
            commit_sha="abc",
        )
        items: list[AnalysisItem] = []
        run_dir = writer.write(book="PHM", chapter=1, items=items)
        assert run_dir.exists()

    def test_book_uppercase_in_dir_name(self, writer, run_metadata, phm_discourse_map, basic_suggestion):
        items = build_scope_items(
            book="phm", chapter=2,
            run_metadata_obs=run_metadata,
            discourse_map_obs=phm_discourse_map,
            verse_observations=[],
            chapter_summary_obs=basic_suggestion,
        )
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        run_dir = writer.write(book="phm", chapter=2, items=items, timestamp=ts)
        assert "PHM" in run_dir.name
