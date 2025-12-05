import React, { useState, useEffect, useMemo } from 'react';
import { Upload, FileText, ChevronRight, AlertCircle, BookOpen, BarChart2 } from 'lucide-react';

// --- Color Scale Logic ---
const getScoreColor = (score) => {
  const s = parseInt(score);
  if (isNaN(s)) return 'bg-white hover:bg-gray-50'; // Default

  // 1-10 Scale mapping as requested
  // 10: Green, 9: Light Green, 8: Yellow-Green, 7: Light Yellow
  // 6: Dark Yellow, <6: Shades of Red
  if (s >= 10) return 'bg-green-500 text-white';
  if (s === 9) return 'bg-green-300 text-gray-900';
  if (s === 8) return 'bg-lime-300 text-gray-900';
  if (s === 7) return 'bg-yellow-200 text-gray-900';
  if (s === 6) return 'bg-yellow-500 text-white';
  if (s === 5) return 'bg-orange-300 text-gray-900';
  if (s === 4) return 'bg-orange-400 text-white';
  if (s === 3) return 'bg-red-400 text-white';
  if (s === 2) return 'bg-red-500 text-white';
  if (s <= 1) return 'bg-red-700 text-white';
  
  return 'bg-white';
};

const getScoreBadgeColor = (score) => {
  const s = parseInt(score);
  if (isNaN(s)) return 'bg-gray-100 text-gray-800';
  if (s >= 8) return 'bg-green-100 text-green-800 border-green-200';
  if (s >= 6) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
  return 'bg-red-100 text-red-800 border-red-200';
};

// --- Parsers ---

/**
 * Robust CSV Parser that handles quoted fields with newlines
 * Essential for the provided sample data which contains complex text blocks
 */
const parseCSV = (text) => {
  const rows = [];
  let currentRow = [];
  let currentField = '';
  let insideQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (char === '"') {
      if (insideQuotes && nextChar === '"') {
         currentField += '"';
         i++; // skip escaped quote
      } else {
         insideQuotes = !insideQuotes;
      }
    } else if (char === ',' && !insideQuotes) {
      currentRow.push(currentField.trim());
      currentField = '';
    } else if ((char === '\r' || char === '\n') && !insideQuotes) {
      if (char === '\r' && nextChar === '\n') i++;
      currentRow.push(currentField.trim());
      if (currentRow.length > 0 && (currentRow.length > 1 || currentRow[0] !== '')) {
        rows.push(currentRow);
      }
      currentRow = [];
      currentField = '';
    } else {
      currentField += char;
    }
  }
  if (currentField || currentRow.length > 0) {
    currentRow.push(currentField.trim());
    rows.push(currentRow);
  }
  return rows;
};

/**
 * Custom USFM Parser
 * Designed to strip metadata, footnotes, and alignment data,
 * extracting only the chapter, verse, and scripture text.
 */
const parseUSFM = (text) => {
  const book = {};
  
  // 1. Clean extraneous tags (Footnotes \f ... \f*, Word Alignment \w ... \w*, etc)
  // We utilize a regex that lazily matches content between tags
  let cleanText = text
    .replace(/\\f\s.+?\\f\*/g, '') // Remove footnotes
    .replace(/\\x\s.+?\\x\*/g, '') // Remove cross references
    .replace(/\\w\s.+?\\w\*/g, '') // Remove word alignment
    .replace(/\\r/g, '')            // Remove references
    .replace(/\\s\d/g, '')          // Remove headings
    .replace(/\\p/g, '')            // Remove paragraph markers
    .replace(/\\q\d?/g, '')         // Remove poetic markers
    .replace(/\\b/g, '')            // Remove breaks
    .replace(/\\m/g, '')            // Remove margins
    .replace(/\\nb/g, '');          // Remove no-break

  // 2. Split by Chapter
  const chapters = cleanText.split(/\\c\s+(\d+)/);
  
  // The split results in [preamble, chNum, content, chNum, content...]
  for (let i = 1; i < chapters.length; i += 2) {
    const chapterNum = parseInt(chapters[i]);
    const content = chapters[i + 1];
    
    book[chapterNum] = {};

    // 3. Split by Verse
    const verses = content.split(/\\v\s+(\d+)/);
    for (let j = 1; j < verses.length; j += 2) {
      const verseNum = parseInt(verses[j]);
      let verseText = verses[j + 1];
      
      // Final cleanup of any remaining backslash commands or excessive whitespace
      verseText = verseText.replace(/\\[a-z0-9]+\s?/g, ' ').replace(/\s+/g, ' ').trim();
      
      if (verseText) {
        book[chapterNum][verseNum] = verseText;
      }
    }
  }
  
  return book;
};


export default function BibleAnalyzer() {
  const [usfmData, setUsfmData] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [activeChapter, setActiveChapter] = useState(1);
  const [activeVerse, setActiveVerse] = useState(null); // { c, v }
  const [error, setError] = useState(null);

  // --- Handlers ---

  const handleUsfmUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const parsed = parseUSFM(evt.target.result);
        setUsfmData(parsed);
        // Default to first chapter found
        const firstCh = Object.keys(parsed)[0];
        if (firstCh) setActiveChapter(parseInt(firstCh));
      } catch (err) {
        setError("Failed to parse USFM file. Ensure it is valid UTF-8 text.");
      }
    };
    reader.readAsText(file);
  };

  const handleCsvUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const rawRows = parseCSV(evt.target.result);
        
        // Assume Header: Chapter, Verse, Face_Annotation, Score, Final_Analysis
        // We'll map headers to indices to be safe
        const headers = rawRows[0].map(h => h.toLowerCase().trim());
        const cIdx = headers.indexOf('chapter');
        const vIdx = headers.indexOf('verse');
        const sIdx = headers.indexOf('score');
        const fIdx = headers.findIndex(h => h.includes('face'));
        const aIdx = headers.findIndex(h => h.includes('analysis'));

        if (cIdx === -1 || vIdx === -1) {
          setError("CSV must have 'Chapter' and 'Verse' columns.");
          return;
        }

        const dataMap = {};
        
        // Start from index 1 to skip header
        for (let i = 1; i < rawRows.length; i++) {
          const row = rawRows[i];
          if (row.length < 2) continue;

          const c = parseInt(row[cIdx]);
          const v = parseInt(row[vIdx]);
          
          if (!dataMap[c]) dataMap[c] = {};
          
          dataMap[c][v] = {
            face: fIdx > -1 ? row[fIdx] : '',
            score: sIdx > -1 ? row[sIdx] : '',
            analysis: aIdx > -1 ? row[aIdx] : '',
            fullRow: row
          };
        }
        setCsvData(dataMap);
      } catch (err) {
        setError("Failed to parse CSV file.");
      }
    };
    reader.readAsText(file);
  };

  // --- Derived State ---

  const chapters = useMemo(() => {
    if (!usfmData) return [];
    return Object.keys(usfmData).map(Number).sort((a, b) => a - b);
  }, [usfmData]);

  const currentVerses = useMemo(() => {
    if (!usfmData || !activeChapter) return [];
    const chData = usfmData[activeChapter];
    if (!chData) return [];
    
    return Object.keys(chData)
      .map(Number)
      .sort((a, b) => a - b)
      .map(vNum => {
        const text = chData[vNum];
        // Check for CSV match
        const analysis = csvData?.[activeChapter]?.[vNum];
        return {
          vNum,
          text,
          analysis
        };
      });
  }, [usfmData, csvData, activeChapter]);

  const selectedVerseData = useMemo(() => {
    if (!activeVerse || !usfmData) return null;
    const { c, v } = activeVerse;
    
    return {
      c, v,
      text: usfmData[c]?.[v],
      analysis: csvData?.[c]?.[v]
    };
  }, [activeVerse, usfmData, csvData]);


  // --- Render ---

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 font-sans">
      
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-600 rounded-lg text-white">
            <BookOpen size={20} />
          </div>
          <h1 className="text-xl font-bold text-gray-800 tracking-tight">Scripture Linguistic Analysis</h1>
        </div>

        <div className="flex items-center gap-4">
          {/* USFM Upload */}
          <div className="relative group">
            <input 
              type="file" 
              accept=".usfm,.txt" 
              onChange={handleUsfmUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <button className={`flex items-center gap-2 px-4 py-2 rounded-md border transition-colors ${usfmData ? 'bg-green-50 border-green-200 text-green-700' : 'bg-white border-gray-300 text-gray-600 hover:bg-gray-50'}`}>
              <FileText size={16} />
              <span className="text-sm font-medium">{usfmData ? 'USFM Loaded' : 'Upload USFM'}</span>
            </button>
          </div>

          {/* CSV Upload */}
          <div className="relative group">
            <input 
              type="file" 
              accept=".csv" 
              onChange={handleCsvUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <button className={`flex items-center gap-2 px-4 py-2 rounded-md border transition-colors ${csvData ? 'bg-blue-50 border-blue-200 text-blue-700' : 'bg-white border-gray-300 text-gray-600 hover:bg-gray-50'}`}>
              <BarChart2 size={16} />
              <span className="text-sm font-medium">{csvData ? 'Data Loaded' : 'Upload CSV'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {usfmData ? (
        <div className="flex flex-1 overflow-hidden">
          
          {/* Left Column: Scripture Viewer */}
          <div className="flex-1 flex flex-col min-w-0 border-r border-gray-200 bg-white">
            
            {/* Chapter Toolbar */}
            <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-white sticky top-0 z-10">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-gray-500 uppercase tracking-wider">Chapter</span>
                <select 
                  value={activeChapter} 
                  onChange={(e) => setActiveChapter(parseInt(e.target.value))}
                  className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md focus:ring-blue-500 focus:border-blue-500 block p-2"
                >
                  {chapters.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
              <div className="text-xs text-gray-400">
                {currentVerses.length} verses found
              </div>
            </div>

            {/* Verse List */}
            <div className="flex-1 overflow-y-auto p-6 space-y-3">
              {currentVerses.map(({ vNum, text, analysis }) => {
                const score = analysis ? analysis.score : null;
                const bgColorClass = score ? getScoreColor(score) : 'bg-white border border-gray-100 hover:border-blue-300';
                const isSelected = activeVerse?.c === activeChapter && activeVerse?.v === vNum;

                return (
                  <div 
                    key={vNum}
                    onClick={() => setActiveVerse({ c: activeChapter, v: vNum })}
                    className={`
                      relative p-4 rounded-lg cursor-pointer transition-all duration-200
                      ${bgColorClass}
                      ${isSelected ? 'ring-4 ring-blue-500/30 scale-[1.01] shadow-lg z-10' : 'shadow-sm hover:shadow-md'}
                    `}
                  >
                    <div className="flex gap-4">
                      <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-full bg-black/10 text-sm font-bold opacity-70">
                        {vNum}
                      </span>
                      <p className={`text-lg leading-relaxed ${score ? '' : 'text-gray-700'}`}>
                        {text}
                      </p>
                      {score && (
                        <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-white/20 text-xs font-bold backdrop-blur-sm">
                          {score}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Analysis Panel */}
          <div className="w-1/3 min-w-[400px] bg-gray-50 flex flex-col border-l border-gray-200">
            {selectedVerseData ? (
              <div className="flex flex-col h-full overflow-hidden">
                <div className="p-6 border-b border-gray-200 bg-white shadow-sm">
                  <div className="flex items-center gap-2 mb-2">
                     <span className="text-xs font-bold uppercase text-gray-400 tracking-wider">Selected Verse</span>
                     <div className="flex-1 h-px bg-gray-100"></div>
                  </div>
                  <h2 className="text-2xl font-serif text-gray-800 mb-2">
                    Chapter {selectedVerseData.c}, Verse {selectedVerseData.v}
                  </h2>
                  <p className="text-gray-600 italic border-l-4 border-blue-500 pl-4 py-1">
                    "{selectedVerseData.text}"
                  </p>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                  {selectedVerseData.analysis ? (
                    <div className="space-y-6">
                      
                      {/* Score Card */}
                      <div className={`p-5 rounded-xl border ${getScoreBadgeColor(selectedVerseData.analysis.score)}`}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-bold uppercase tracking-wide opacity-80">Linguistic Score</span>
                          <span className="text-3xl font-black">{selectedVerseData.analysis.score}</span>
                        </div>
                        <div className="w-full bg-black/10 h-2 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-current opacity-50" 
                            style={{ width: `${Math.min(selectedVerseData.analysis.score * 10, 100)}%` }}
                          ></div>
                        </div>
                      </div>

                      {/* Face Annotation */}
                      {selectedVerseData.analysis.face && (
                        <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                          <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Face Annotation</h3>
                          <p className="text-gray-800 font-medium">
                            {selectedVerseData.analysis.face}
                          </p>
                        </div>
                      )}

                      {/* Detailed Analysis */}
                      <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                        <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Final Analysis</h3>
                        <div className="prose prose-sm prose-blue max-w-none text-gray-700 whitespace-pre-wrap">
                          {selectedVerseData.analysis.analysis}
                        </div>
                      </div>

                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center text-gray-400">
                      <BarChart2 size={48} className="mb-4 opacity-20" />
                      <p>No linguistic data available for this verse.</p>
                      <p className="text-sm mt-2">Upload a CSV file to see analysis.</p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 p-8">
                <ChevronRight size={48} className="mb-4 opacity-20" />
                <h3 className="text-lg font-medium text-gray-500">No Verse Selected</h3>
                <p className="text-sm mt-2 max-w-xs">
                  Click on any verse row in the left panel to view its detailed linguistic analysis.
                </p>
              </div>
            )}
          </div>

        </div>
      ) : (
        /* Empty State */
        <div className="flex-1 flex flex-col items-center justify-center bg-gray-50 p-6">
          <div className="max-w-md w-full bg-white p-8 rounded-2xl shadow-xl border border-gray-100 text-center">
            <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <Upload size={32} />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Start your Analysis</h2>
            <p className="text-gray-500 mb-8">
              Upload a standard USFM file to parse scripture, then attach your CSV data for deep linguistic insights.
            </p>
            
            <div className="space-y-3">
              <label className="block w-full">
                <div className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg cursor-pointer transition-colors flex items-center justify-center gap-2">
                  <FileText size={18} />
                  Choose USFM File
                </div>
                <input type="file" accept=".usfm,.txt" onChange={handleUsfmUpload} className="hidden" />
              </label>
              
              <div className="text-xs text-gray-400 pt-2">
                Supported formats: .usfm, .txt (USFM formatted)
              </div>
            </div>

            {error && (
              <div className="mt-6 p-4 bg-red-50 text-red-700 text-sm rounded-lg flex items-start gap-2 text-left">
                <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                {error}
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}