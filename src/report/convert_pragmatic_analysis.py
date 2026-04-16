import pandas as pd
import json
import numpy as np

from data.interpresure import Interpresure

def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_json_parse(json_str):
    """
    Parses a string that might be a JSON list, a JSON object, 
    or a string containing double-encoded JSON.
    """
    if not isinstance(json_str, str):
        return []
    
    try:
        # Step 1: Parse the outer layer
        parsed_outer = json.loads(json_str)
        
        # Step 2: If it's a list, check if items are stringified JSON
        if isinstance(parsed_outer, list):
            cleaned_list = []
            for item in parsed_outer:
                if isinstance(item, str):
                    try:
                        # Skip instructional strings
                        if "Now we transition" in item or item.strip() == "":
                            continue
                        cleaned_item = json.loads(item)
                        cleaned_list.append(cleaned_item)
                    except json.JSONDecodeError:
                        pass 
                elif isinstance(item, dict):
                    cleaned_list.append(item)
            return cleaned_list
            
        return parsed_outer
    except (json.JSONDecodeError, TypeError):
        return []

def convert_pragmatic(
        individual_path, 
        output_path, 
        interpresure: Interpresure | None, 
        book: str, 
):
    topic = "general"

    # 1. Load the CSVs
    df_individual = pd.read_csv(individual_path)
    df_individual.columns = df_individual.columns.str.lower()

    # 2. Extract Top-Level Metadata
    book_name = book
    # Get chapter from the first row if it exists
    chapter_num = int(df_individual['chapter'].iloc[0]) if not df_individual.empty else 1
    if interpresure is None:
        try:
            interpresure = Interpresure(book_name, chapter_num)
        except Exception:
            interpresure = None
    
    # 3. Process and Group Data
    # We group by these three keys to ensure unique segments are not merged
    group_cols = ['verse', 'biblical_text'] # + annotation_columns
    
    # for col in annotation_columns:
    #     df_individual[col] = df_individual[col].fillna("Uncategorized")
    
    analysis_list = []

    # Iterate through each unique combination of Verse, Text, and Annotation
    grouped = df_individual.groupby(group_cols, sort=False)

    for (verse_num, biblical_text, *unfiltered_annotations), ind_rows in grouped:
        # convert boolean types away from np types which don't serialize
        annotations = [bool(x) if isinstance(x, np.bool_) else x for x in unfiltered_annotations]
        
        # Extract segment-specific metadata
        first_row = ind_rows.iloc[0]
        translation = first_row.get('translation', "")
        # notes = first_row.get('notes', "")

        inner_analysis = []

        # -- A. ADD INDIVIDUAL ANALYSES (One per model) --
        for _, row in ind_rows.iterrows():
            reasoning = row.get('reasoning', "")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = row.get('model_analysis', "")
            inner_analysis.append({
                "type": "individual",
                "model": row.get("model", "Unknown"),
                "score": _safe_int(row.get('score', 0)),
                "confidence": _safe_int(row.get('confidence', 0)),
                "reasoning": reasoning,
                "strengths": row.get("strengths", ""),
                "weaknesses": row.get("weaknesses", ""),
                "suggestions": row.get("suggestions", ""),
                "model_analysis": row.get('model_analysis', "")
            })

        # 4. Construct the unique Verse-Segment Object
        verse_obj = {
            "verse": int(verse_num),
            "biblical_text": biblical_text,
            "translation": translation,
            # "annotations": [{"type": col, "annotation": ann } for (col, ann) in zip(annotation_columns, annotations)],
            # "notes": notes,
            "analysis": inner_analysis
        }
        analysis_list.append(verse_obj)

    # 5. Final JSON Structure
    final_json = {
        "book": book_name,
        "chapter": chapter_num,
        "pragmatic_goal": {
            "type": topic,
            "title": interpresure.get_topic_title(topic) if interpresure else "Pragmatic Annotations",
            "goal": interpresure.get_topic_goal(topic) if interpresure else "Preserve the pragmatics described by the expert from the original text.",
            "description": interpresure.get_topic_description(topic) if interpresure else "This group contains pragmatic analysis of the current verse."
        },
        "analysis": analysis_list,
    }

    class NumpyBoolEncoder(json.JSONEncoder):
        def default(self, obj):
            # Handle NumPy booleans/numbers
            if isinstance(obj, np.bool_):
                return bool(obj)
            # Handle standard booleans if they were somehow masked
            if isinstance(obj, bool):
                return str(obj) # Or just return obj to get JSON true/false
            return super().default(obj)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False, cls=NumpyBoolEncoder)
    
    print(f"Successfully coalesced {len(analysis_list)} analysis segments into {output_path}")
    return final_json
