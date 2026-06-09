"""CLI entry point for the translator QA pipeline."""

from __future__ import annotations

import click

from .answer import answer
from .generate import generate


@click.group()
def cli() -> None:
    """Translator QA: generate diagnostic questions and answer them for a translation."""


cli.add_command(generate)
cli.add_command(answer)

__all__ = ["cli", "generate", "answer"]
