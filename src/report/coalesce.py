# import pandas as pd
# import json
# import ast

# def parse_list_string(s):
#     """Parses a string representation of a list (e.g. "['a', 'b']") into an actual list."""
#     try:
#         return ast.literal_eval(s)
#     except (ValueError, SyntaxError):
#         return []

# def coalesce_csvs(individual_path, debate_path, output_path):
#     # 1. Load the CSVs
#     df_individual = pd.read_csv(individual_path)
#     df_debate = pd.read_csv(debate_path)

#     # 2. Parse complex columns
#     # The 'closing_statements' column contains strings that look like lists
#     df_debate['closing_statements'] = df_debate['closing_statements'].apply(parse_list_string)

#     # 3. Extract Top-Level Metadata
#     # Assuming the "chapter" column in debate CSV actually holds the Book Name (Philemon)
#     book_name = "Philemon"
#     # Assuming the "Chapter" column in individual CSV holds the Chapter Number
#     chapter_num = int(df_individual['Chapter'].iloc[0]) if not df_individual.empty else 1
    
#     # 4. Merge Data by Verse
#     # We find all unique verses present in either file
#     all_verses = sorted(set(df_individual['Verse']).union(set(df_debate['verse'])))

#     analysis_list = []

#     for verse_num in all_verses:
#         # Filter rows for the current verse
#         ind_rows = df_individual[df_individual['Verse'] == verse_num]
#         deb_row = df_debate[df_debate['verse'] == verse_num]

#         # Extract Verse Metadata (Prioritizing the Individual CSV)
#         if not ind_rows.empty:
#             first_row = ind_rows.iloc[0]
#             greek = first_row.get('Greek_Text', "")
#             annotation = first_row.get('Face_Annotation', "")
#             notes = first_row.get('Notes', "")
#         else:
#             greek = ""
#             annotation = ""
#             notes = ""

#         # Build the 'analysis' array for this verse
#         inner_analysis = []

#         # Add "Individual" entries
#         for _, row in ind_rows.iterrows():
#             inner_analysis.append({
#                 "type": "individual",
#                 "model": row["Model"],
#                 "score": int(row['Score']),
#                 "reasoning": row['Model_Analysis']
#             })

#         # Add "Debate" entry (assuming 1 per verse)
#         if not deb_row.empty:
#             row = deb_row.iloc[0]
#             inner_analysis.append({
#                 "type": "debate",
#                 "score": int(row['final_consensus_score']),
#                 "summary": row['consensus_summary'],
#                 "closing_statements": row['closing_statements']
#             })

#         # Create the Verse Object
#         verse_obj = {
#             "verse": int(verse_num),
#             "greek": greek,
#             "annotation": annotation,
#             "notes": notes,
#             "analysis": inner_analysis
#         }
#         analysis_list.append(verse_obj)

#     # 5. Construct Final JSON Structure
#     final_json = {
#         "book": book_name,
#         "chapter": chapter_num,
#         "category": "Pauline Epistles", # Placeholder or derived logic
#         "analysis": analysis_list
#     }

#     # 6. Save to File
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_json, f, indent=2, ensure_ascii=False)
    
#     print(f"Successfully created {output_path}")

# # --- Usage ---
# # coalesce_csvs('individual.csv', 'debate.csv', 'output.json')


import pandas as pd
import json

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

def coalesce_csvs(individual_path, debate_path, output_path):
    # 1. Load the CSVs
    df_individual = pd.read_csv(individual_path)
    df_debate = pd.read_csv(debate_path)

    # Normalize column names
    df_debate.columns = [c.lower() for c in df_debate.columns]
    df_individual.columns = [c.capitalize() for c in df_individual.columns] 

    # 2. Extract Top-Level Metadata
    book_name = "Philemon"
    # Get chapter from the first row if it exists
    chapter_num = int(df_individual['Chapter'].iloc[0]) if not df_individual.empty else 1
    
    # 3. Process and Group Data
    # We group by these three keys to ensure unique segments are not merged
    group_cols = ['Verse', 'Greek_text', 'Face_annotation']
    
    # Fill NaN to avoid grouping errors
    df_individual['Face_annotation'] = df_individual['Face_annotation'].fillna("Uncategorized")
    
    analysis_list = []

    # Iterate through each unique combination of Verse, Text, and Annotation
    grouped = df_individual.groupby(group_cols, sort=False)

    for (verse_num, greek_text, annotation), ind_rows in grouped:
        # Extract segment-specific metadata
        first_row = ind_rows.iloc[0]
        translation = first_row.get('Translation', "")
        notes = first_row.get('Notes', "")

        inner_analysis = []

        # -- A. ADD INDIVIDUAL ANALYSES (One per model) --
        for _, row in ind_rows.iterrows():
            inner_analysis.append({
                "type": "individual",
                "model": row.get("Model", "Unknown"),
                "score": int(row.get('Score', 0)),
                "reasoning": row.get('Model_analysis', "")
            })

        # -- B. ADD DEBATE ANALYSIS (Matching this specific segment) --
        # We strip whitespace to ensure matching isn't broken by a stray space
        deb_rows = df_debate[
            (df_debate['verse'] == verse_num) & 
            (df_debate['greek_text'].str.strip() == greek_text.strip()) & 
            (df_debate['face_annotation'].str.strip() == annotation.strip())
        ]

        if not deb_rows.empty:
            deb_row = deb_rows.iloc[0]
            
            # Parse Closing Statements
            closing_data = safe_json_parse(deb_row.get('closing_statements', '[]'))
            closing_scores = []
            formatted_closing = []
            
            for s in closing_data:
                if isinstance(s, dict):
                    score = s.get('proposed_score')
                    if score is not None:
                        closing_scores.append(int(score))
                    formatted_closing.append({
                        "agent": s.get('agent_name', 'Unknown'),
                        "statement": s.get('argument', ""),
                        "score": score
                    })
            
            final_consensus_score = min(closing_scores) if closing_scores else 0

            # Parse Debate Transcript
            debate_raw_data = safe_json_parse(deb_row.get('debate', '[]'))
            transcript = []

            for turn in debate_raw_data:
                if not isinstance(turn, dict):
                    continue
                
                if 'agent_name' in turn:
                    transcript.append({
                        "role": "linguist",
                        "agent": turn['agent_name'],
                        "argument": turn.get('argument'),
                        "proposed_score": turn.get('proposed_score')
                    })
                elif 'intervene' in turn:
                    transcript.append({
                        "role": "moderator",
                        "agent": "Moderator",
                        "violators": turn.get('violators', [""]),
                        "intervened": turn.get('intervene'),
                        "feedback": turn.get('feedback', "")
                    })

            inner_analysis.append({
                "type": "debate",
                "score": final_consensus_score,
                "debate_transcript": transcript,
                "closing_statements": formatted_closing
            })

        # 4. Construct the unique Verse-Segment Object
        verse_obj = {
            "verse": int(verse_num),
            "greek": greek_text,
            "translation": translation,
            "annotation": annotation,
            "notes": notes,
            "analysis": inner_analysis
        }
        analysis_list.append(verse_obj)

    # 5. Final JSON Structure
    final_json = {
        "book": book_name,
        "chapter": chapter_num,
        "category": "Pauline Epistles",
        "analysis": analysis_list
    }

    # 6. Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully coalesced {len(analysis_list)} analysis segments into {output_path}")