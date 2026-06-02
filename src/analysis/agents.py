"""Agent Framework analyst and critic agents for pragmatic analysis.

Each analysis type gets its own task description (few-shot and zero-shot
variants).  The analyst is stateless by default; callers create an
``AgentSession`` for rolling verse-by-verse conversations.  The critic is
always called statelessly — no session needed.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from .schemas import (
    CriticReview,
    DiscourseMapObservation,
    InterpreSureSuggestionsObservation,
    TranslatorQuestionsObservation,
)

# ---------------------------------------------------------------------------
# Task descriptions per analysis type
# ---------------------------------------------------------------------------

_COMMON_RULES = dedent("""\
    **Response Language:** Always respond in English regardless of source or target language.

    **Strict grounding rule:** Base every claim on observable lexical, grammatical,
    or discourse features of the texts provided.  Do not import theological commentary,
    popular tradition, or extra-textual knowledge unless it is directly relevant to
    the pragmatic force of the passage.

    **Citation rules:**
    - In reasoning sections you may cite Greek/Hebrew words or phrases from the
      original text and words/phrases from the translation.
    - In feedback sections (strengths, weaknesses, suggestions) use only plain
      English and cite the translation directly.  No Greek or Hebrew script,
      transliteration, or vocabulary in feedback.
    - Feedback must be in simple, layperson-friendly English.
""")

_INTERPRESURE_FEW_SHOT_INSTRUCTIONS = dedent("""\
    You are an expert Biblical Linguist and Translation Consultant performing a
    cross-lingual pragmatic analysis.

    You will evaluate a verse-by-verse translation from Biblical Greek or Hebrew
    into a Gateway Language.  Your focus is pragmatic fidelity: does the translation
    communicate not only what the original says, but what it implies, presupposes,
    performs, and conveys between the lines?

""") + _COMMON_RULES + dedent("""\

    **Inputs you will receive:**
    1. Full chapter text (both original and translation) — provided once at the
       start of the conversation.
    2. A discourse map for the chapter — provided once at the start.
    3. Expert InterpreSure annotations for each verse (when analyzing that verse).
    4. BART discourse annotations (Greek only, when available).
    5. MACULA syntax tokens (Greek only, when available).

    **Strict resource rule:** You MUST engage with the provided InterpreSure
    annotations, BART, and MACULA data.  If a significant pragmatic feature
    identified in the annotations is not correctly handled in the translation,
    the score MUST reflect that.

    **Tools available:**
    - Use ``lookup_verse_range`` to check any cross-reference to another verse,
      chapter, or book.  Cite the reference in ``cross_references``.
    - Use the InterpreSure skill's ``lookup_term`` to retrieve precise definitions
      before applying technical terminology.
    - Use the ``checklist`` resource at the start of each verse analysis to work
      through the full set of analysis questions systematically.
""")

_INTERPRESURE_ZERO_SHOT_INSTRUCTIONS = dedent("""\
    You are an expert Biblical Linguist and Translation Consultant performing a
    cross-lingual pragmatic analysis.

    You will evaluate a verse-by-verse translation from Biblical Greek or Hebrew
    into a Gateway Language.  Your focus is pragmatic fidelity: does the translation
    communicate not only what the original says, but what it implies, presupposes,
    performs, and conveys between the lines?

""") + _COMMON_RULES + dedent("""\

    **Inputs you will receive:**
    1. Full chapter text (both original and translation) — provided once at the
       start of the conversation.
    2. A discourse map for the chapter — provided once at the start.
    3. Each verse in sequence for individual analysis.

    **Tools available:**
    - Use ``lookup_verse_range`` to check any cross-reference to another verse,
      chapter, or book.  Cite the reference in ``cross_references``.
    - Use the InterpreSure skill's ``lookup_term`` to retrieve precise definitions
      before applying technical terminology.
    - Use the ``checklist`` resource to guide systematic analysis.
""")

_TRANSLATOR_QUESTIONS_FEW_SHOT_INSTRUCTIONS = dedent("""\
    You are an expert Biblical Linguist generating diagnostic questions for a
    Bible translator.

    Your task is to examine each verse and produce a set of plain-language
    questions that will help the translator evaluate whether their rendering
    preserves the pragmatic meaning of the original text.  Questions must be
    answerable by looking at the translation alone — no source language
    knowledge should be required.

""") + _COMMON_RULES + dedent("""\

    **Inputs you will receive:**
    1. Full chapter text and discourse map at the start.
    2. Expert InterpreSure annotations per verse.
    3. BART and MACULA data when available (Greek only).

    **Each question must:**
    - Be written in plain English, free of linguistic jargon.
    - Be answerable from the translation text alone.
    - Be grounded in a specific pragmatic feature from the annotations.
    - Include a brief rationale (one sentence) explaining what pragmatic
      feature is at stake.
""")

_TRANSLATOR_QUESTIONS_ZERO_SHOT_INSTRUCTIONS = dedent("""\
    You are an expert Biblical Linguist generating diagnostic questions for a
    Bible translator.

    Your task is to examine each verse and produce a set of plain-language
    questions that will help the translator evaluate whether their rendering
    preserves the pragmatic meaning of the original text.  Questions must be
    answerable by looking at the translation alone — no source language
    knowledge should be required.

""") + _COMMON_RULES + dedent("""\

    **Inputs you will receive:**
    1. Full chapter text and discourse map at the start.
    2. Each verse in sequence for individual analysis.

    Base questions on your expert knowledge of biblical language pragmatics.
    Use the InterpreSure skill's checklist and term lookup to ground your work.
""")

_CRITIC_INSTRUCTIONS = dedent("""\
    You are the Linguistic Critic.  Your role is to rigorously review a submitted
    translation analysis.

    Your ONLY criteria for approval:
    1. The analysis must be based on verifiable linguistic, pragmatic, or semantic
       arguments grounded in the texts provided.
    2. Every word or phrase cited MUST be present in the texts being analyzed.
    3. The analysis must be limited to how the translation handles the original
       text with respect to pragmatics.  No theological opinion, popular
       commentary, or extra-textual tradition.
    4. If expert annotations were provided, the analysis must engage with them —
       not merely acknowledge their existence.

    Respond ONLY as a JSON object with two fields:
    - ``accepted`` (boolean): true if the analysis meets the criteria.
    - ``reasoning`` (string): explanation.  If false, provide specific revision
      instructions the analyst can act on immediately.
""")

# Map analysis_type → (few_shot_instructions, zero_shot_instructions, output_schema)
_ANALYSIS_TYPE_REGISTRY: dict[str, dict[str, Any]] = {
    "interpresure_suggestions": {
        "few_shot_instructions": _INTERPRESURE_FEW_SHOT_INSTRUCTIONS,
        "zero_shot_instructions": _INTERPRESURE_ZERO_SHOT_INSTRUCTIONS,
        "output_schema": InterpreSureSuggestionsObservation,
        "discourse_schema": DiscourseMapObservation,
    },
    "translator_questions": {
        "few_shot_instructions": _TRANSLATOR_QUESTIONS_FEW_SHOT_INSTRUCTIONS,
        "zero_shot_instructions": _TRANSLATOR_QUESTIONS_ZERO_SHOT_INSTRUCTIONS,
        "output_schema": TranslatorQuestionsObservation,
        "discourse_schema": DiscourseMapObservation,
    },
}

DEFAULT_ANALYSIS_TYPE = "interpresure_suggestions"


def get_analysis_type_config(analysis_type: str) -> dict[str, Any]:
    """Return the config dict for the given analysis type.

    Raises ``ValueError`` for unknown types.
    """
    if analysis_type not in _ANALYSIS_TYPE_REGISTRY:
        available = ", ".join(_ANALYSIS_TYPE_REGISTRY)
        raise ValueError(
            f"Unknown analysis_type '{analysis_type}'. Available: {available}"
        )
    return _ANALYSIS_TYPE_REGISTRY[analysis_type]


def _force_all_required(schema: dict[str, Any]) -> None:
    """Recursively ensure every property key appears in ``required``.

    OpenAI's strict structured-output mode requires that every key listed in
    ``properties`` also appears in the ``required`` array, even for fields that
    have Pydantic defaults.  This is stricter than standard JSON Schema, where
    defaults make a field optional.  We fix this by walking the schema tree and
    rewriting ``required`` to cover every property.

    Optional fields that accept ``null`` are still valid — Pydantic represents
    them as ``anyOf: [{type: "..."}, {type: "null"}]``, which OpenAI accepts
    as long as the key is in ``required``.
    """
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        for prop_schema in schema["properties"].values():
            if isinstance(prop_schema, dict):
                _force_all_required(prop_schema)

    # Recurse into named definitions ($defs covers Pydantic v2; definitions for v1)
    for defs_key in ("$defs", "definitions"):
        if defs_key in schema:
            for sub in schema[defs_key].values():
                if isinstance(sub, dict):
                    _force_all_required(sub)

    # Recurse into array items and additionalProperties
    for key in ("items", "additionalProperties"):
        if key in schema and isinstance(schema[key], dict):
            _force_all_required(schema[key])

    # Recurse into allOf / anyOf / oneOf sub-schemas
    for combiner in ("allOf", "anyOf", "oneOf"):
        if combiner in schema:
            for sub in schema[combiner]:
                if isinstance(sub, dict):
                    _force_all_required(sub)


def _response_format_for(schema_cls) -> dict[str, Any]:
    """Build an OpenAI strict response_format dict for a Pydantic model.

    Applies ``_force_all_required`` to satisfy OpenAI's constraint that every
    property key must appear in ``required`` when ``strict: True`` is set.
    """
    schema = schema_cls.model_json_schema()
    _force_all_required(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_cls.__name__,
            "strict": True,
            "schema": schema,
        },
    }


def make_agents(
    *,
    model: str,
    critic_model: str,
    analysis_type: str,
    analysis_mode: str,
    api_key: str | None,
    base_url: str | None,
    verse_lookup_tool,
    skills_provider,
):
    """Create and return ``(analyst, critic)`` Agent Framework agents.

    Both agents are stateless by default.  Callers create an ``AgentSession``
    on the analyst for rolling verse-by-verse conversations.

    Args:
        model: Model identifier for the analyst.
        critic_model: Model identifier for the critic.
        analysis_type: One of the registered analysis type keys.
        analysis_mode: ``"few_shot"`` or ``"zero_shot"``.
        api_key: API key (None to use environment default).
        base_url: Base URL for OpenAI-compatible endpoint (None for default).
        verse_lookup_tool: The ``@tool``-decorated lookup function from ``tools.py``.
        skills_provider: A ``SkillsProvider`` instance (e.g. InterpreSure skill).
    """
    from agent_framework import Agent  # type: ignore[import]
    from agent_framework.openai import OpenAIChatCompletionClient  # type: ignore[import]

    type_config = get_analysis_type_config(analysis_type)
    instructions_key = (
        "few_shot_instructions" if analysis_mode == "few_shot" else "zero_shot_instructions"
    )
    analyst_instructions = type_config[instructions_key]
    output_schema = type_config["output_schema"]

    # --- Analyst client ---
    analyst_client_kwargs: dict[str, Any] = {"model": model}
    if api_key:
        analyst_client_kwargs["api_key"] = api_key
    if base_url:
        analyst_client_kwargs["base_url"] = base_url

    analyst_client = OpenAIChatCompletionClient(**analyst_client_kwargs)

    analyst = Agent(
        name="PRAGMATIC_ANALYST",
        client=analyst_client,
        instructions=analyst_instructions,
        tools=[verse_lookup_tool],
        context_providers=[skills_provider],
        default_options={
            "response_format": _response_format_for(output_schema),
        },
    )

    # --- Critic client ---
    critic_client_kwargs: dict[str, Any] = {"model": critic_model}
    if api_key:
        critic_client_kwargs["api_key"] = api_key
    if base_url:
        critic_client_kwargs["base_url"] = base_url

    critic_client = OpenAIChatCompletionClient(**critic_client_kwargs)

    critic = Agent(
        name="LINGUISTIC_CRITIC",
        client=critic_client,
        instructions=_CRITIC_INSTRUCTIONS,
        default_options={
            "response_format": _response_format_for(CriticReview),
        },
    )

    return analyst, critic


def make_discourse_agent(
    *,
    model: str,
    analysis_type: str,
    api_key: str | None,
    base_url: str | None,
    skills_provider,
):
    """Create a single-use stateless agent for the global discourse pass."""
    from agent_framework import Agent  # type: ignore[import]
    from agent_framework.openai import OpenAIChatCompletionClient  # type: ignore[import]

    type_config = get_analysis_type_config(analysis_type)
    discourse_schema = type_config["discourse_schema"]

    client_kwargs: dict[str, Any] = {"model": model}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIChatCompletionClient(**client_kwargs)

    return Agent(
        name="DISCOURSE_ANALYST",
        client=client,
        instructions=dedent("""\
            You are an expert Biblical Linguist.  Your task is to produce a
            structured discourse map for a chapter of scripture.

            Analyze the full chapter text provided (both original language and
            translation) along with any expert annotations, BART discourse
            structure, and MACULA syntax data.

            Identify:
            - The dominant Question(s) Under Discussion
            - The argumentative, narrative, or poetic arc
            - Discourse unit boundaries with verse ranges and functional descriptions
            - Relational and social dynamics operative across the chapter
            - Active scalar dimensions
            - Key presuppositions the discourse takes for granted
            - Genre and register notes

            Be precise and grounded in the text.  The discourse map will anchor
            all subsequent verse-level analyses, so accuracy here matters.
        """).strip(),
        context_providers=[skills_provider],
        default_options={
            "response_format": _response_format_for(discourse_schema),
        },
    )
