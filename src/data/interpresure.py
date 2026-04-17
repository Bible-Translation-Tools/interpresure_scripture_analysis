from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

from pandas import DataFrame

from data.interpresure_content import (
    CsvInterpresureContentLoader,
    InterpresureContentLoader,
    InterpresureContentLoaderRegistry,
    InterpresureSource,
    JsonInterpresureContentLoader,
)


class Interpresure:

    topics = ["implicature", "structure", "social", "scales"]

    groups = {
        "implicature": {
            "title": """Implicature and Inference""",
            "description": "This group captures meaning that is communicated indirectly. It tracks what the reader must 'read between the lines' based on context and shared knowledge.",
            "columns": [
                "inference_type",
                "prejacent",
                "inferred_proposition",
                "invited_inference",
                "presupposition_type",
                "implicature_type",
                "is_cancelled"
            ],
            "goal": "Verify that the translation triggers the same cognitive inferences as the source text. It ensures that subtext remains 'subtext' and isn't lost or made too explicit, which would change the communicative strategy."
        },
        "structure": {
            "title": """Information Structure""",
            "description": "This group analyzes the 'packaging' of information—how the author organizes what is already known versus what is new or emphasized.",
            "columns": [
                "information_structure",
                "question_under_discussion",
                "predication_type",
                "veridicality",
                "modality",
                "entailment_pattern"
            ],
            "goal": "Maintain the original thematic focus and emphasis. If the source text highlights a specific concept through word order or phrasing, the translation should use equivalent target-language mechanics to ensure the same 'center of gravity' for the sentence."
        },
        "social": {
            "title": """Social and Relational Dynamics""",
            "description": "This group monitors the 'social temperature' and power balance. It tracks how the author manages their relationship with the recipient through politeness, stance, and authority.",
            "columns": [
                "face",
                "stance",
                "illocutionary_force",
                "evidentiality",
                "veridicality",
            ],
            "goal": "Preserve the interpersonal 'vibe' of the communication. This ensures that the translation accurately reflects the author's level of deference, authority, or solidarity, preventing a respectful request from sounding like a cold demand."
        },
        "scales": {
            "title": """Scales and Contrast""",
            "description": "This group deals with relative values and sets of alternatives. It tracks comparisons where things are weighed against each other on a scale of importance or magnitude.",
            "columns": [
                "is_scalar",
                "scale_type",
                "alternative",
                "is_exhausted"
            ],
            "goal": "Preserve the 'weight' and trajectory of comparisons. If the author uses a scale to show that one concept is 'even more' important than another, the translation must maintain that upward or downward intensity."
        },
        "general": {
            "title": """Pragmatic Annotations""",
            "description": "This group contains expert linguistic annotations pertaining to pragmatics.",
            "columns": [
                "annotations"
            ],
            "goal": "Preserve the pragmatics described by the expert from the original text."
        }
    }

    _source_root = Path(__file__).resolve().parents[2] / "interpresure"
    _sources = {
        ("phm", 1): InterpresureSource("PHM", 1, _source_root / "interpresure_phm.csv", "csv"),
        ("php", 1): InterpresureSource("PHP", 1, _source_root / "interpresure_php_1.csv", "csv"),
        ("psa", 145): InterpresureSource("PSA", 145, _source_root / "interpresure_psa_145.csv", "csv"),
    }
    _content_loaders = InterpresureContentLoaderRegistry(
        {
            "csv": CsvInterpresureContentLoader(),
            "json": JsonInterpresureContentLoader(),
        }
    )

    interpresure: DataFrame = None

    def __init__(
        self,
        book: str,
        chapter: int,
        *,
        source: InterpresureSource | None = None,
        content_loader: InterpresureContentLoader | None = None,
        loader_registry: InterpresureContentLoaderRegistry | None = None,
    ):
        self.book = book
        self.chapter = int(chapter)
        self.source = source or self._get_source(book, chapter)
        self.loader_registry = loader_registry or self.__class__._content_loaders
        self.content_loader = content_loader
        self.load()

    @classmethod
    def register_source(cls, book: str, chapter: int, path: Path, loader_name: str = "csv") -> None:
        cls._sources[(book.lower(), int(chapter))] = InterpresureSource(
            book.upper(),
            int(chapter),
            Path(path),
            loader_name,
        )

    @classmethod
    def register_content_loader(cls, name: str, loader: InterpresureContentLoader) -> None:
        cls._content_loaders.register(name, loader)

    def load(self, book: str | None = None, chapter: int | None = None):
        if book is not None or chapter is not None:
            self.book = book if book is not None else self.book
            self.chapter = int(chapter) if chapter is not None else self.chapter
            self.source = self._get_source(self.book, self.chapter)

        if self.content_loader is not None:
            self.interpresure = self.content_loader.load(self.source).fillna("Not Applicable")
            self.content = self.interpresure
            return self.interpresure

        self.interpresure = self.loader_registry.load(self.source).fillna("Not Applicable")
        self.content = self.interpresure
        return self.interpresure

    @classmethod
    def _get_source(cls, book: str, chapter: int) -> InterpresureSource:
        key = (book.lower(), int(chapter))
        try:
            return cls._sources[key]
        except KeyError as exc:
            available = ", ".join([f"{b.upper()} {c}" for b, c in sorted(cls._sources.keys())])
            raise KeyError(
                f"No Interpresure source registered for {book.upper()} {int(chapter)}. "
                f"Available sources: {available}"
            ) from exc

    def get_topics(self) -> list[str]:
        return self.topics
    
    def get_topic_description(self, topic):
        return self.groups[topic]["description"]
    
    def get_topic_goal(self, topic):
        return self.groups[topic]["goal"]
    
    def get_topic_title(self, topic):
        return self.groups[topic]["title"]
    
    def get_topic_categories(self, topic):
        categories = self.groups[topic]["columns"].copy()
        categories[-1] = f"and {categories[-1]}"
        if len(categories) > 2:
            return ", ".join([y.replace("_", " ") for y in categories])
        else: 
            return " ".join([y.replace("_", " ") for y in categories])
        
    def get_topic_columns(self, topic) -> list[str]:
        return self.groups[topic]["columns"].copy()

    def _normalize_topics(self, topics: str | Sequence[str] | None = None) -> list[str]:
        if topics is None:
            normalized_topics = list(self.topics)
        elif isinstance(topics, str):
            normalized_topics = [topics]
        else:
            normalized_topics = list(topics)

        if not normalized_topics:
            normalized_topics = list(self.topics)

        invalid_topics = [topic for topic in normalized_topics if topic not in self.groups]
        if invalid_topics:
            available = ", ".join(self.get_topics())
            raise KeyError(
                f"Unknown topic(s): {', '.join(invalid_topics)}. Available topics: {available}"
            )

        seen: set[str] = set()
        deduped_topics: list[str] = []
        for topic in normalized_topics:
            if topic not in seen:
                deduped_topics.append(topic)
                seen.add(topic)
        return deduped_topics

    def _annotation_columns_for_topics(self, topics: list[str], include_notes: bool) -> list[str]:
        columns: list[str] = []
        for topic in topics:
            for column in self.get_topic_columns(topic):
                if column not in columns:
                    columns.append(column)

        for column in ["token_id", "book", "chapter", "verse", "biblical_text"]:
            if column not in columns:
                columns.append(column)

        if include_notes and "notes" not in columns:
            columns.append("notes")

        return columns

    def get_annotations(
        self,
        topic: str | Sequence[str] | None = None,
        include_notes: bool = True,
    ) -> DataFrame:
        topics = self._normalize_topics(topic)
        columns = self._annotation_columns_for_topics(topics, include_notes)

        df = self.interpresure.reindex(columns=columns)[columns]
        df = df.fillna("No Annotation")
        return df

    def get_annotations_markdown(
        self,
        topic: str | Sequence[str] | None = None,
        chapter: int | None = None,
        verse: int | None = None,
        include_notes: bool = False,
    ) -> str:
        if chapter is None or verse is None:
            raise ValueError("chapter and verse are required for annotation markdown generation.")

        topics = self._normalize_topics(topic)
        pragmatic_annotations = "\n## Pragmatic Expert Annotations:\n"

        for current_topic in topics:
            df = self.get_annotations(current_topic, include_notes)
            grouped_data = df[(df["chapter"] == chapter) & (df["verse"] == verse)]

            if grouped_data.empty:
                continue

            pragmatic_annotations += f"\n### {self.get_topic_title(current_topic)}\n"

            for _, row in grouped_data.iterrows():
                pragmatic_annotations += f"#### {row['biblical_text']}\n"
                pragmatic_annotations += "\n".join(
                    [f"- {column}: {row[column]} " for column in self.get_topic_columns(current_topic)]
                )
                pragmatic_annotations += "\n"

        return pragmatic_annotations
