// 1. Define the specific shapes for the nested analysis items
export interface DebateEntry {
  type: "debate";
  score: number; // 0 - 10
  summary: string;
  closing_statements: string[];
}

export interface IndividualEntry {
  type: "individual";
  score: number;
  confidence?: number;
  reasoning?: string;
  strengths?: string;
  weaknesses?: string;
  suggestions?: string;
  model_analysis?: string;
}

export interface ComparisonJudgment {
  better_side: "left" | "right" | "tie";
  better_is_left: boolean;
  confidence: number;
  reasoning: string;
}

export interface AnalysisSummary {
  model?: string;
  score?: number | null;
  confidence?: number | null;
  reasoning?: string;
  strengths?: string;
  weaknesses?: string;
  suggestions?: string;
  model_analysis?: string;
}

export interface ComparisonVerseReport {
  book: string;
  chapter: number;
  verse: number;
  status: "matched" | "left_only" | "right_only";
  translation_text: string;
  biblical_text: string;
  left_present: boolean;
  right_present: boolean;
  left_model: string;
  right_model: string;
  left_analysis: AnalysisSummary;
  right_analysis: AnalysisSummary;
  left_analysis_items: AnalysisEntry[];
  right_analysis_items: AnalysisEntry[];
  field_diffs: Array<{ field: string; left: unknown; right: unknown }>;
  analysis_comparison: unknown;
  comparison_judgment: ComparisonJudgment;
  highlight_side: "left" | "right" | "tie" | null;
}

// 2. Create a Union Type (Polymorphism)
// This allows the array to contain either type
export type AnalysisEntry = DebateEntry | IndividualEntry;

// 3. Define the Verse structure
export interface VerseData {
  verse: number;
  greek: string;
  annotation: string;
  notes: string;
  // The nested analysis array containing the union type
  analysis: AnalysisEntry[];
}

// 4. Define the Root Object
export interface BookAnalysis {
  book: string;
  chapter: number;
  category: string;
  analysis: VerseData[];
}

export interface ComparisonReport {
  left_path?: string;
  right_path?: string;
  comparison_model?: string;
  left_translation?: unknown;
  right_translation?: unknown;
  translation_diffs: Array<{ field: string; left: unknown; right: unknown }>;
  left_evaluation_count: number;
  right_evaluation_count: number;
  verse_report_count: number;
  verse_reports: ComparisonVerseReport[];
}
