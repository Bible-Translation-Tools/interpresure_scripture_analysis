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
import ast

def safe_json_parse(json_str):
    """
    Parses a string that might be a JSON list, a JSON object, 
    or a string containing double-encoded JSON.
    """
    if not isinstance(json_str, str):
        return []
    
    try:
        # Step 1: Parse the outer layer (the list structure)
        parsed_outer = json.loads(json_str)
        
        # Step 2: If it's a list, check if items are stringified JSON
        if isinstance(parsed_outer, list):
            cleaned_list = []
            for item in parsed_outer:
                if isinstance(item, str):
                    try:
                        # Skip instructional strings (e.g., "Now we transition...")
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

    # Normalize column names to lowercase to avoid KeyErrors
    df_debate.columns = [c.lower() for c in df_debate.columns]
    # Ensure Individual CSV columns are Capitalized (matches your likely input)
    df_individual.columns = [c.capitalize() for c in df_individual.columns] 

    # 2. Extract Top-Level Metadata
    book_name = "Philemon"
    chapter_num = int(df_individual['Chapter'].iloc[0]) if not df_individual.empty else 1
    
    # 3. Merge Data by Verse
    individual_verses = set(df_individual['Verse']) if 'Verse' in df_individual.columns else set()
    debate_verses = set(df_debate['verse']) if 'verse' in df_debate.columns else set()
    all_verses = sorted(individual_verses.union(debate_verses))

    analysis_list = []

    for verse_num in all_verses:
        # Filter rows
        ind_rows = df_individual[df_individual['Verse'] == verse_num]
        deb_rows = df_debate[df_debate['verse'] == verse_num]

        # Extract Verse Metadata
        if not ind_rows.empty:
            first_row = ind_rows.iloc[0]
            greek = first_row.get('Greek_text', "")
            annotation = first_row.get('Face_annotation', "")
            notes = first_row.get('Notes', "")
            translation = first_row.get('Translation', "")
        elif not deb_rows.empty:
            first_row = deb_rows.iloc[0]
            greek = first_row.get('greek_text', "")
            annotation = first_row.get('face_annotation', "")
            notes = ""
            translation = first_row.get('translation', "")
        else:
            greek, annotation, notes, translation = "", "", "", ""

        inner_analysis = []

        # -- PROCESS INDIVIDUAL --
        for _, row in ind_rows.iterrows():
            inner_analysis.append({
                "type": "individual",
                "model": row.get("Model", "Unknown"),
                "score": int(row.get('Score', 0)),
                "reasoning": row.get('Model_analysis', "")
            })

        # -- PROCESS DEBATE --
        if not deb_rows.empty:
            row = deb_rows.iloc[0]
            
            # A. Parse Closing Statements (to derive final score)
            closing_statements_data = safe_json_parse(row.get('closing_statements', '[]'))
            
            # Calculate Consensus Score: Min of closing scores
            closing_scores = []
            formatted_closing = []
            for s in closing_statements_data:
                if isinstance(s, dict):
                    if 'proposed_score' in s:
                        closing_scores.append(int(s['proposed_score']))
                    formatted_closing.append({
                        "agent": s.get('agent_name', 'Unknown'),
                        "statement": s.get('argument', ""),
                        "score": s.get('proposed_score')
                    })
            
            final_score = min(closing_scores) if closing_scores else 0

            # B. Parse Full Debate Transcript
            debate_raw_data = safe_json_parse(row.get('debate', '[]'))
            transcript = []

            for turn in debate_raw_data:
                if not isinstance(turn, dict):
                    continue
                
                # Handle Linguist Turn
                if 'agent_name' in turn:
                    transcript.append({
                        "role": "linguist",
                        "agent": turn['agent_name'],
                        "argument": turn.get('argument'),
                        "proposed_score": turn.get('proposed_score')
                    })
                # Handle Moderator Turn
                elif 'intervene' in turn:
                    # Include if they intervened OR gave feedback (even if empty, sometimes useful to see checks)
                    # We can filter out empty checks if desired, but here we keep "intervene: false" checks
                    # to show the moderator was listening.
                    transcript.append({
                        "role": "moderator",
                        "agent": "Moderator",
                        "violators": turn.get('violators', [""]),
                        "intervened": turn.get('intervene'),
                        "feedback": turn.get('feedback', "")
                    })

            inner_analysis.append({
                "type": "debate",
                "score": final_score,
                # summary field removed
                "debate_transcript": transcript,
                "closing_statements": formatted_closing
            })

        # Create Verse Object
        verse_obj = {
            "verse": int(verse_num),
            "greek": greek,
            "translation": translation,
            "annotation": annotation,
            "notes": notes,
            "analysis": inner_analysis
        }
        analysis_list.append(verse_obj)

    # 4. Construct Final JSON
    final_json = {
        "book": book_name,
        "chapter": chapter_num,
        "category": "Pauline Epistles",
        "analysis": analysis_list
    }

    # 5. Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_path}")