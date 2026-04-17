from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class InterpresureSource:
    book: str
    chapter: int
    path: Path
    loader_name: str


@runtime_checkable
class InterpresureContentLoader(Protocol):
    def load(self, source: InterpresureSource) -> pd.DataFrame:
        """Load an Interpresure source file into a dataframe."""


class InterpresureContentLoaderRegistry:
    def __init__(self, loaders: dict[str, InterpresureContentLoader] | None = None):
        self._loaders: dict[str, InterpresureContentLoader] = dict(loaders or {})

    def register(self, name: str, loader: InterpresureContentLoader) -> None:
        self._loaders[name] = loader

    def get(self, name: str) -> InterpresureContentLoader:
        try:
            return self._loaders[name]
        except KeyError as exc:
            raise KeyError(f"No Interpresure content loader registered for '{name}'.") from exc

    def load(self, source: InterpresureSource) -> pd.DataFrame:
        return self.get(source.loader_name).load(source)


class CsvInterpresureContentLoader:
    def __init__(
        self,
        *,
        fillna_value: str = "Not Applicable",
        lower_case_columns: bool = True,
        strip_bom: bool = True,
        rename_columns: dict[str, str] | None = None,
    ):
        self.fillna_value = fillna_value
        self.lower_case_columns = lower_case_columns
        self.strip_bom = strip_bom
        self.rename_columns = rename_columns or {}

    def load(self, source: InterpresureSource) -> pd.DataFrame:
        df = pd.read_csv(source.path)
        df = self._normalize_columns(df)
        return df.fillna(self.fillna_value)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns: list[str] = []
        for column in df.columns:
            normalized = str(column)
            if self.strip_bom:
                normalized = normalized.lstrip("\ufeff")
            normalized = normalized.strip()
            if self.lower_case_columns:
                normalized = normalized.lower()
            normalized = self.rename_columns.get(normalized, normalized)
            columns.append(normalized)
        df = df.copy()
        df.columns = columns
        return df


class JsonInterpresureContentLoader:
    def __init__(self, *, fillna_value: str = "Not Applicable"):
        self.fillna_value = fillna_value

    def load(self, source: InterpresureSource) -> pd.DataFrame:
        with open(source.path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            payload = [payload]

        if not isinstance(payload, list):
            raise ValueError(f"Unsupported Interpresure JSON payload in {source.path}")

        return pd.DataFrame(payload).fillna(self.fillna_value)
