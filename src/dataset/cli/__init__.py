"""Click front end package for the dataset tools."""

from __future__ import annotations

import click

from .build import build
from .compare import compare
from .convert import convert
from .test import test


@click.group()
def cli() -> None:
    """Dataset utilities for schema-driven CSV and AI generation."""


cli.add_command(convert)
cli.add_command(build)
cli.add_command(test)
cli.add_command(compare)

__all__ = ["cli", "build", "compare", "convert", "test"]
