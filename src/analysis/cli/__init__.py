"""Click front end package for pragmatic analysis tools."""

from __future__ import annotations

import click

from .convert import convert
from .build import build
from .compare import compare
from .test import test


@click.group()
def cli() -> None:
    """Pragmatic analysis utilities for zero-shot and expert-guided outputs."""


cli.add_command(build)
cli.add_command(test)
cli.add_command(compare)
cli.add_command(convert)

__all__ = ["cli", "build", "compare", "convert", "test"]
