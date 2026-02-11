from collections import defaultdict

from agents.eli5 import Eli5Agent
from agents.secretary import SecretaryAgent
from model.config import get_config_for_model
from teams.debate import LinguistTurn
from collections import defaultdict
import json

class SummarizeDebate:

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "linguist_review",
            "schema": LinguistTurn.model_json_schema()
        }
    }

    def __init__(self, secretary_model = "gpt-5.2", eli5_model = "gpt-5.2"):
        secretary_model_config = get_config_for_model(secretary_model)
        eli5_model_config = get_config_for_model(eli5_model)

        self.secretary = SecretaryAgent("SECRETARY", secretary_model_config["model"], secretary_model_config["key"], secretary_model_config["base_url"], response_format=self.response_format)
        self.eli5agent = Eli5Agent("ELI5_SUMMARIZER", eli5_model_config["model"], eli5_model_config["key"], eli5_model_config["base_url"], response_format=self.response_format)

    async def summarize(self, analysis):
        grouped = self._group_analyses_by_chapter_and_verse(analysis)
        reports = defaultdict(dict)
        for chapter, verses in grouped.items():
            print(f"Chapter {chapter}")

            for verse, analyses in verses.items():
                print(f"Verse {verse}")
                report = await self._get_summary_for_chapter(analyses)
                reports[chapter][verse] = report

        return reports

    async def _get_summary_for_chapter(self, analyses):
        reports = []
        for analysis_block in analyses:
            analysis_items = analysis_block["analysis"]

            filtered = list(filter(lambda x: x["type"] == "conclusion", analysis_items))
            if (len(filtered) > 0):
                conclusion = filtered[0]
                reports.append({
                    "pragmatic_goal": analysis_block["pragmatic_goal"]["title"],
                    "report": conclusion["summary"]
                })

        summary = await self.secretary.summarize_reports(reports)
        summary = json.loads(summary)["summary"]

        eli5 = await self.eli5agent.eli5(summary)
        eli5 = json.loads(eli5)["summary"]

        return {"summary": summary, "eli5": eli5}



    def _group_analyses_by_chapter_and_verse(self, data):
        """
        Groups all analyses by chapter and verse.

        Output structure:
        {
            chapter: {
                verse: [
                    {
                        "pragmatic_goal": {...},
                        "greek": "...",
                        "translation": "...",
                        "analysis": [...]
                    },
                    ...
                ]
            }
        }
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for evaluation in data.get("evaluation", []):
            chapter = evaluation.get("chapter")
            pragmatic_goal = evaluation.get("pragmatic_goal")

            for verse_block in evaluation.get("analysis", []):
                verse = verse_block.get("verse")

                grouped[chapter][verse].append({
                    "pragmatic_goal": pragmatic_goal,
                    "greek": verse_block.get("greek"),
                    "translation": verse_block.get("translation"),
                    "analysis": verse_block.get("analysis", [])
                })

        return grouped