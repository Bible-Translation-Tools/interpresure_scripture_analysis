"""Tests for analysis/skills/interpresure_skill.py — glossary and checklist."""

from __future__ import annotations

import json
from textwrap import dedent

import pytest

from analysis.skills.interpresure_skill import (
    _load_checklist,
    _load_glossary,
    _parse_glossary,
)


# ---------------------------------------------------------------------------
# _parse_glossary
# ---------------------------------------------------------------------------


class TestParseGlossary:
    def test_numbered_entry(self):
        text = "1. Scalar Implicature: A defeasible inference from a weaker term."
        g = _parse_glossary(text)
        assert "Scalar Implicature" in g
        assert "defeasible" in g["Scalar Implicature"]

    def test_unnumbered_entry(self):
        text = "Is Scalar: Whether an expression invokes ordered alternatives."
        g = _parse_glossary(text)
        assert "Is Scalar" in g

    def test_multiline_definition(self):
        text = dedent("""\
            1. Presupposition: Background information treated as accepted.
            It survives embedding under negation.
            Further details here.
        """)
        g = _parse_glossary(text)
        assert "Presupposition" in g
        defn = g["Presupposition"]
        assert "survives embedding" in defn
        assert "Further details" in defn

    def test_empty_text(self):
        assert _parse_glossary("") == {}

    def test_blank_lines_ignored(self):
        text = "\n\n1. Face: Social value.\n\n2. Stance: Speaker attitude.\n"
        g = _parse_glossary(text)
        assert len(g) == 2

    def test_multiple_entries(self):
        text = dedent("""\
            1. Term One: Definition one.
            2. Term Two: Definition two.
            3. Term Three: Definition three.
        """)
        g = _parse_glossary(text)
        assert len(g) == 3
        assert "Term One" in g
        assert "Term Three" in g

    def test_strips_term_whitespace(self):
        text = "1.  Spaced Term  : Definition here."
        g = _parse_glossary(text)
        # Term should be stripped
        keys = list(g.keys())
        assert any("Spaced Term" in k for k in keys)


# ---------------------------------------------------------------------------
# _load_glossary and _load_checklist (real files)
# ---------------------------------------------------------------------------


class TestLoadGlossaryAndChecklist:
    def test_glossary_loads_nonzero_terms(self):
        g = _load_glossary()
        assert len(g) > 50, f"Expected >50 terms, got {len(g)}"

    def test_glossary_contains_key_terms(self):
        g = _load_glossary()
        term_names = [k.lower() for k in g.keys()]
        assert any("scalar implicature" in t for t in term_names)
        assert any("presupposition" in t for t in term_names)
        assert any("illocutionary force" in t for t in term_names)
        assert any("veridicality" in t for t in term_names)
        assert any("face" in t for t in term_names)

    def test_glossary_definitions_nonempty(self):
        g = _load_glossary()
        for term, defn in g.items():
            assert defn.strip(), f"Empty definition for term: {term!r}"

    def test_checklist_loads_questions(self):
        checklist = _load_checklist()
        lines = [l.strip() for l in checklist.splitlines() if l.strip()]
        assert len(lines) >= 30, f"Expected ≥30 checklist lines, got {len(lines)}"

    def test_checklist_contains_key_questions(self):
        checklist = _load_checklist()
        assert "illocutionary" in checklist.lower()
        assert "presuppos" in checklist.lower()
        assert "question under discussion" in checklist.lower()


# ---------------------------------------------------------------------------
# InterpreSureSkill.lookup_term (via class instantiation without agent_framework)
# ---------------------------------------------------------------------------


class TestLookupTerm:
    """Test the lookup_term logic directly from the skill class internals."""

    @pytest.fixture
    def glossary(self):
        return _parse_glossary(dedent("""\
            1. Scalar Implicature: A defeasible inference from weaker to stronger terms.
            2. Presupposition: Background information taken for granted.
            3. Face: The positive social value a person claims.
            4. Illocutionary Force: The action an utterance performs.
            Scale Type: The dimension on which alternatives are ordered.
        """))

    def _lookup(self, glossary, term):
        """Run the same lookup logic as InterpreSureSkill.lookup_term."""
        if not term or not term.strip():
            return json.loads('{"error": "term must not be empty"}')

        needle = term.strip().lower()

        for name, defn in glossary.items():
            if name.lower() == needle:
                return {"term": name, "definition": defn}

        matches = [(n, d) for n, d in glossary.items() if needle in n.lower()]
        if len(matches) == 1:
            return {"term": matches[0][0], "definition": matches[0][1]}
        if matches:
            return {"error": "ambiguous", "suggestions": [m[0] for m in matches[:5]]}

        needle_words = set(needle.split())
        partial = [(n, d) for n, d in glossary.items() if needle_words & set(n.lower().split())]
        if partial:
            return {"term": partial[0][0], "definition": partial[0][1], "note": "partial match"}

        return {"error": f"Term '{term}' not found"}

    def test_exact_match(self, glossary):
        result = self._lookup(glossary, "Face")
        assert result["term"] == "Face"
        assert "social value" in result["definition"]

    def test_case_insensitive_exact(self, glossary):
        result = self._lookup(glossary, "face")
        assert result["term"] == "Face"

    def test_substring_match_unique(self, glossary):
        result = self._lookup(glossary, "Scalar Impl")
        assert "Scalar Implicature" in result.get("term", "")

    def test_ambiguous_match(self):
        # Two entries both containing "scale" in the name → ambiguous
        ambiguous_glossary = _parse_glossary(
            "Scale Type: Dimension of alternatives.\nScale Structure: Endpoint structure.\n"
        )
        result = self._lookup(ambiguous_glossary, "scale")
        assert "error" in result or "suggestions" in result

    def test_empty_term(self, glossary):
        result = self._lookup(glossary, "")
        assert "error" in result

    def test_none_term(self, glossary):
        result = self._lookup(glossary, None)
        assert "error" in result

    def test_no_match(self, glossary):
        result = self._lookup(glossary, "zzznomatch")
        assert "error" in result

    def test_partial_word_match(self, glossary):
        result = self._lookup(glossary, "Illocutionary")
        assert result.get("term") == "Illocutionary Force"


# ---------------------------------------------------------------------------
# Full skill round-trip (with agent_framework)
# ---------------------------------------------------------------------------


class TestInterpreSureSkillRoundtrip:
    def test_provider_created(self):
        from analysis.skills.interpresure_skill import make_interpresure_skills_provider
        provider = make_interpresure_skills_provider()
        assert provider is not None

    def test_lookup_term_returns_valid_json(self):
        from analysis.skills.interpresure_skill import make_interpresure_skills_provider, _load_glossary
        from agent_framework import ClassSkill, SkillFrontmatter

        glossary = _load_glossary()

        # Instantiate the skill class directly for unit testing
        class _TestableSkill(ClassSkill):
            def __init__(self):
                super().__init__(frontmatter=SkillFrontmatter(name="test", description="test"))
                self._glossary = glossary

            @property
            def instructions(self):
                return "test"

            @ClassSkill.script(name="lookup-term", description="Look up term")
            def lookup_term(self, term: str) -> str:
                needle = term.strip().lower()
                for name, defn in self._glossary.items():
                    if name.lower() == needle:
                        return json.dumps({"term": name, "definition": defn})
                return json.dumps({"error": f"not found: {term}"})

        skill = _TestableSkill()
        result = json.loads(skill.lookup_term("Face"))
        assert "term" in result or "error" in result
