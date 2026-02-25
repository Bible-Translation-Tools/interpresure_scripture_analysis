from pandas import DataFrame
import pandas as pd


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
        }
    }

    interpresure: DataFrame = None

    _files = {
        "phm": {
            "1": "../interpresure/interpresure_phm.csv"
        },
        "php": {
            "1": "../interpresure/interpresure_php_1.csv"
        }
    }

    def __init__(self, book: str, chapter: int):
        self.load(book, chapter)

    def load(self, book: str, chapter: int):
        self.interpresure = pd.read_csv(self._get_file(book, chapter)).fillna("Not Applicable")

    def _get_file(self, book: str, chapter: int) -> str:
        return self._files[book.lower()][f"{chapter}"]

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

    def get_annotations(self, topic: str, include_notes = True) -> DataFrame:
        columns = self.groups[topic]["columns"].copy()
        columns += ["token_id", "book", "chapter", "verse", "biblical_text"]
        if include_notes:
            columns += ["notes"]
        
        df = self.interpresure.reindex(columns=columns)[columns]
        df = df.fillna("No Annotation")
        return df

    def get_annotations_markdown(self, topic: str, chapter: int, verse: int, include_notes = True) -> str:

        df = self.get_annotations(topic, include_notes)

        grouped_data = df[(df['chapter'] == chapter) & (df['verse'] == verse)]
        iterr = grouped_data.iterrows()

        pragmatic_annotations = "\n## Pragmatic Expert Annotations:\n"

        for _, row in iterr:
            pragmatic_annotations += f"### {row['biblical_text']}\n"
            pragmatic_annotations += "\n".join([f"- {x}: {row[x]} " for x in self.get_topic_columns(topic)])
            pragmatic_annotations += "\n"

        return pragmatic_annotations