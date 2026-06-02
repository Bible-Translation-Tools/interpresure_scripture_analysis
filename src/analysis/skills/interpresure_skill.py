"""InterpreSure Agent Skill.

Packages the InterpreSure controlled vocabulary (glossary) and the analysis
checklist as an Agent Framework ClassSkill.  The skill follows the progressive
disclosure pattern:

  1. The description (~100 tokens) is injected at agent startup so the agent
     knows the skill exists.
  2. The agent calls ``load_skill`` to retrieve the full instructions when it
     needs to do pragmatic analysis.
  3. The agent calls ``read_skill_resource("checklist")`` to load the 40+
     question analysis checklist on demand.
  4. The agent calls ``run_skill_script("lookup_term", {"term": "..."})`` to
     retrieve a precise InterpreSure definition for any pragmatic term.

Usage::

    provider = make_interpresure_skills_provider()
    agent = Agent(..., context_providers=[provider])
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent

_DATA_DIR = Path(__file__).parent / "data"
_GLOSSARY_PATH = _DATA_DIR / "glossary.txt"
_CHECKLIST_PATH = _DATA_DIR / "checklist.txt"


def _load_checklist() -> str:
    """Load the raw checklist text."""
    if _CHECKLIST_PATH.exists():
        return _CHECKLIST_PATH.read_text(encoding="utf-8").strip()
    return "(Checklist not found — ensure checklist.txt is present in skills/data/)"


def _parse_glossary(text: str) -> dict[str, str]:
    """Parse the numbered glossary into a term → definition dict.

    Handles entries in the form::

        1. Term Name: Definition text that may span multiple sentences.
        2. Another Term: ...

    Also handles un-numbered bold-style headings like ``Is Scalar: ...``.
    """
    glossary: dict[str, str] = {}
    current_term: str | None = None
    current_def: list[str] = []

    numbered = re.compile(r"^\d+\.\s+(.+?):\s+(.*)")
    unnumbered = re.compile(r"^([A-Z][A-Za-z ,/\-]+):\s+(.*)")

    def flush():
        if current_term and current_def:
            glossary[current_term.strip()] = " ".join(current_def).strip()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = numbered.match(line) or unnumbered.match(line)
        if m:
            flush()
            current_term = m.group(1).strip()
            current_def = [m.group(2).strip()] if m.group(2).strip() else []
        elif current_term:
            current_def.append(line)

    flush()
    return glossary


def _load_glossary() -> dict[str, str]:
    if _GLOSSARY_PATH.exists():
        return _parse_glossary(_GLOSSARY_PATH.read_text(encoding="utf-8"))
    return {}


def make_interpresure_skills_provider():
    """Build and return a SkillsProvider containing the InterpreSure skill.

    Lazy-imports ``agent_framework`` so the module can be imported even in
    environments where the package is not yet installed.
    """
    from agent_framework import ClassSkill, SkillFrontmatter, SkillsProvider  # type: ignore[import]

    glossary = _load_glossary()
    checklist_text = _load_checklist()

    class InterpreSureSkill(ClassSkill):
        """InterpreSure pragmatics skill bundling glossary and analysis checklist."""

        def __init__(self) -> None:
            super().__init__(
                frontmatter=SkillFrontmatter(
                    name="interpresure-pragmatics",
                    description=dedent("""\
                        InterpreSure controlled vocabulary and systematic analysis
                        checklist for biblical pragmatic analysis. Use this skill
                        when analyzing translation fidelity with respect to
                        pragmatics — implicature, presupposition, information
                        structure, speech acts, scalarity, modality, social
                        dynamics, and discourse coherence. Use lookup_term to
                        retrieve precise definitions; use the checklist resource
                        to guide systematic verse analysis.
                    """).strip(),
                )
            )
            self._glossary = glossary
            self._checklist = checklist_text

        @property
        def instructions(self) -> str:
            return dedent("""\
                ## How to use this skill

                1. **Checklist**: Before analyzing a verse or chapter, load the
                   ``checklist`` resource to review the full set of pragmatic
                   analysis questions. Work through the checklist systematically
                   rather than relying on intuition alone.

                2. **Term lookup**: Whenever you use a pragmatic term (e.g.
                   "scalar implicature", "illocutionary force", "veridicality"),
                   use the ``lookup_term`` script to retrieve the precise
                   InterpreSure definition. This ensures your analysis uses the
                   controlled vocabulary consistently and avoids conflating
                   related but distinct concepts.

                3. **Grounding**: Base all analysis claims on observable
                   lexical, grammatical, or discourse features of the texts
                   provided. Do not import theological commentary or extra-textual
                   tradition unless it is directly relevant to pragmatic force.
            """).strip()

        @property
        @ClassSkill.resource(
            name="checklist",
            description=(
                "The full InterpreSure analysis checklist — 41 diagnostic questions "
                "covering every major dimension of pragmatic analysis. Load this "
                "resource at the start of any systematic verse analysis."
            ),
        )
        def checklist(self) -> str:
            """Return the analysis checklist as a numbered markdown list."""
            lines = ["# InterpreSure Analysis Checklist\n"]
            for i, question in enumerate(self._checklist.splitlines(), start=1):
                q = question.strip()
                if q:
                    lines.append(f"{i}. {q}")
            return "\n".join(lines)

        @ClassSkill.script(
            name="lookup-term",
            description=(
                "Look up the precise InterpreSure definition for a pragmatic term. "
                "Returns the definition and any usage notes. Use before applying a "
                "technical term to ensure correct usage."
            ),
        )
        def lookup_term(self, term: str) -> str:
            """Search the InterpreSure glossary for a term and return its definition."""
            if not term or not term.strip():
                return json.dumps({"error": "term must not be empty"})

            needle = term.strip().lower()

            # Exact match first
            for name, defn in self._glossary.items():
                if name.lower() == needle:
                    return json.dumps({"term": name, "definition": defn})

            # Substring match
            matches = [
                (name, defn)
                for name, defn in self._glossary.items()
                if needle in name.lower()
            ]
            if len(matches) == 1:
                name, defn = matches[0]
                return json.dumps({"term": name, "definition": defn})
            if matches:
                suggestions = [m[0] for m in matches[:5]]
                return json.dumps({
                    "error": f"Ambiguous term '{term}'. Did you mean one of: {suggestions}?",
                    "suggestions": suggestions,
                })

            # Word-level fallback
            needle_words = set(needle.split())
            partial = [
                (name, defn)
                for name, defn in self._glossary.items()
                if needle_words & set(name.lower().split())
            ]
            if partial:
                name, defn = partial[0]
                return json.dumps({"term": name, "definition": defn, "note": "partial match"})

            return json.dumps({
                "error": f"Term '{term}' not found in InterpreSure glossary.",
                "suggestion": "Try a shorter or differently-spelled term.",
            })

    return SkillsProvider(InterpreSureSkill())
