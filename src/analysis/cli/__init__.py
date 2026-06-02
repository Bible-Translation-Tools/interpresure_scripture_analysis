"""Click front end for the pragmatic analysis pipeline."""

from __future__ import annotations

import click

from .build import build
from .compare import compare
from .test import test


@click.group()
def cli() -> None:
    """Pragmatic analysis — expert-guided and zero-shot modes."""


cli.add_command(build)
cli.add_command(test)
cli.add_command(compare)

__all__ = ["cli", "build", "compare", "test"]
