"""Agent factories for the translator QA pipeline.

Two agents:
  - ``make_question_gen_agent``: generates plain-language questions from
    InterpreSure annotations and biblical text.
  - ``make_qa_agent``: answers questions about a translation and returns
    pass/fail/na verdicts.

Both use Agent Framework and share the InterpreSure skill.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from analysis.agents import _force_all_required

from .schemas import InterpreSureQAObservation, QuestionGenerationResponse

# ---------------------------------------------------------------------------
# Question generation instructions
# ---------------------------------------------------------------------------

_QUESTION_GEN_INSTRUCTIONS = dedent("""\
    You are an expert Biblical Linguist helping Mother Tongue Translators (MTTs)
    evaluate their translation drafts.

    Your task is to read a chapter of scripture in the original language (Greek or
    Hebrew) and generate a set of plain-language diagnostic questions that an MTT
    can use to check whether their translation preserves the pragmatic meaning of
    the original text.

    **What good questions look like:**
    - Written in plain English — no Greek, Hebrew, transliteration, or linguistic
      jargon (terms like "scalar implicature" must not appear).
    - Answerable by reading the translation alone — the MTT does not need to know
      the source language.
    - Grounded in a specific pragmatic feature from the InterpreSure annotations:
      information structure, speech acts, implicatures, presuppositions, social
      dynamics, scales, or discourse function.
    - Precise enough that a clear pass/fail judgment is possible, or that a human
      reviewer can quickly decide.

    **Importance scoring (1–10):**
    - 9–10: The question tests a feature that is central to the main communicative
      point of the passage. Getting it wrong significantly distorts meaning.
    - 6–8: Important pragmatic nuance. Loss noticeably weakens the text.
    - 3–5: Meaningful but secondary. A reader would still grasp the main point.
    - 1–2: Stylistic or subtle nuance unlikely to affect comprehension.

    **Per-question output:**
    - ``anchor``: the U23003 reference (e.g. "PHM 1:10" or "ROM 3:10-18")
    - ``anchor_level``: always "verse" (U23003 range syntax covers multi-verse spans)
    - ``question``: the plain-language question
    - ``importance``: integer 1–10
    - ``rationale``: one sentence explaining what pragmatic feature is at stake
      (may use technical terms here since this is for the AI answering step, not
      the MTT)
    - ``annotation_topics``: which InterpreSure topics apply
      (implicature / structure / social / scales / general)

    **Strict rules:**
    - Generate questions only for verses that have InterpreSure annotations.
    - Do not generate questions about obvious surface-level translation choices
      (word choice, grammar). Focus on what is communicated *between the lines*.
    - Aim for 2–6 questions per annotated verse or verse range.
    - Do not duplicate questions that test the same feature.
    - ids must be sequential starting from 1.
""")

# ---------------------------------------------------------------------------
# QA answering instructions
# ---------------------------------------------------------------------------

_QA_INSTRUCTIONS = dedent("""\
    You are an expert Biblical Linguist evaluating a translation draft produced
    by a Mother Tongue Translator (MTT).

    You will be given:
    1. A verse or verse range from the original text (Greek or Hebrew).
    2. The MTT's translation of that passage.
    3. A diagnostic question about whether a specific pragmatic feature is
       preserved in the translation.
    4. A rationale explaining what feature is being tested.
    5. Optionally: InterpreSure expert annotations, MACULA token data, and
       BART discourse annotations.

    **Your job:**
    Answer the question based on the translation text. Use the original language
    text and annotations as interpretive context, but base your verdict on what
    the *translation* communicates.

    **Verdict options:**
    - ``pass``: The translation clearly preserves the pragmatic feature the
      question is testing. Confidence is warranted.
    - ``fail``: The translation loses, distorts, or over-explicates the feature.
      Assign severity 1–10 (0 = trivial, 10 = critical meaning loss).
    - ``na``: The question cannot be answered with confidence from the
      translation text alone; human judgment is needed. Assign severity as
      importance for review (1–10).

    **For pass: severity must be 0.**

    **Strict rules:**
    - Base your answer on the translation text. Do not assume features not
      present in the translation.
    - Do not cite Greek or Hebrew in your answer field — the answer must be
      plain English understandable to the MTT.
    - You MAY cite specific words or phrases from the translation to support
      your verdict.
    - Reasoning field: use for technical chain-of-thought before your verdict.
      May reference source language and annotations.
    - Confidence: your honest assessment of how certain you are (0–100).
      Uncertainty about the translation's communicative intent should lower
      confidence.
""")


# ---------------------------------------------------------------------------
# Response format helpers
# ---------------------------------------------------------------------------


def _response_format_for(schema_cls: Any) -> dict[str, Any]:
    """Build an OpenAI strict response_format with all properties forced into required."""
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


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def make_question_gen_agent(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    skills_provider: Any | None = None,
) -> Any:
    """Return a stateless Agent for question generation."""
    from agent_framework import Agent  # type: ignore[import]
    from agent_framework.openai import OpenAIChatCompletionClient  # type: ignore[import]

    client_kwargs: dict[str, Any] = {"model": model}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIChatCompletionClient(**client_kwargs)

    agent_kwargs: dict[str, Any] = {
        "name": "QUESTION_GENERATOR",
        "client": client,
        "instructions": _QUESTION_GEN_INSTRUCTIONS,
        "default_options": {
            "response_format": _response_format_for(QuestionGenerationResponse),
        },
    }
    if skills_provider is not None:
        agent_kwargs["context_providers"] = [skills_provider]

    return Agent(**agent_kwargs)


def make_qa_agent(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    skills_provider: Any | None = None,
) -> Any:
    """Return a stateless Agent for answering QA questions."""
    from agent_framework import Agent  # type: ignore[import]
    from agent_framework.openai import OpenAIChatCompletionClient  # type: ignore[import]

    client_kwargs: dict[str, Any] = {"model": model}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAIChatCompletionClient(**client_kwargs)

    agent_kwargs: dict[str, Any] = {
        "name": "QA_ANALYST",
        "client": client,
        "instructions": _QA_INSTRUCTIONS,
        "default_options": {
            "response_format": _response_format_for(InterpreSureQAObservation),
        },
    }
    if skills_provider is not None:
        agent_kwargs["context_providers"] = [skills_provider]

    return Agent(**agent_kwargs)
