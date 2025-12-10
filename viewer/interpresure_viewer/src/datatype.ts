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
  reasoning: string;
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