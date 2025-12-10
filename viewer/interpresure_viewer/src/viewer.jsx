import React, { useState, useMemo } from 'react';
import { 
  Upload, 
  FileText, 
  ChevronRight, 
  ChevronDown,
  ChevronUp,
  AlertCircle, 
  BookOpen, 
  BarChart2, 
  Users, 
  User, 
  MessageSquare,
  Gavel
} from 'lucide-react';
import Markdown from 'react-markdown';

// --- Color Scale Logic ---
const getScoreColor = (score) => {
  const s = parseInt(score);
  if (isNaN(s)) return 'bg-white hover:bg-gray-50'; // Default

  // 1-10 Scale mapping
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
 * Custom USFM Parser
 * Strips metadata, footnotes, extracting chapter/verse/text.
 */
const parseUSFM = (text) => {
  const book = {};
  
  let cleanText = text
    .replace(/\\f\s.+?\\f\*/g, '') 
    .replace(/\\x\s.+?\\x\*/g, '') 
    .replace(/\\w\s.+?\\w\*/g, '') 
    .replace(/\\r/g, '')            
    .replace(/\\s\d/g, '')          
    .replace(/\\p/g, '')            
    .replace(/\\q\d?/g, '')         
    .replace(/\\b/g, '')            
    .replace(/\\m/g, '')            
    .replace(/\\nb/g, '');          

  const chapters = cleanText.split(/\\c\s+(\d+)/);
  
  for (let i = 1; i < chapters.length; i += 2) {
    const chapterNum = parseInt(chapters[i]);
    const content = chapters[i + 1];
    book[chapterNum] = {};

    const verses = content.split(/\\v\s+(\d+)/);
    for (let j = 1; j < verses.length; j += 2) {
      const verseNum = parseInt(verses[j]);
      let verseText = verses[j + 1];
      verseText = verseText.replace(/\\[a-z0-9]+\s?/g, ' ').replace(/\s+/g, ' ').trim();
      
      if (verseText) {
        book[chapterNum][verseNum] = verseText;
      }
    }
  }
  return book;
};

// --- Helper Components ---

const CollapsibleCard = ({ title, icon: Icon, children, defaultOpen = false, score = null }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-4">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon size={18} className="text-gray-500" />}
          <span className="font-semibold text-gray-700">{title}</span>
        </div>
        <div className="flex items-center gap-3">
          {score !== null && (
            <span className={`text-xs font-bold px-2 py-1 rounded-full ${score >= 8 ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
              Score: {score}
            </span>
          )}
          {isOpen ? <ChevronUp size={18} className="text-gray-400" /> : <ChevronDown size={18} className="text-gray-400" />}
        </div>
      </button>
      
      {isOpen && (
        <div className="p-5 border-t border-gray-100">
          {children}
        </div>
      )}
    </div>
  );
};

// --- Main Application ---

export default function BibleAnalyzer() {
  const [usfmData, setUsfmData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null); // Changed from csvData
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
        const firstCh = Object.keys(parsed)[0];
        if (firstCh) setActiveChapter(parseInt(firstCh));
        setError(null);
      } catch (err) {
        setError("Failed to parse USFM file.");
      }
    };
    reader.readAsText(file);
  };

  const handleJsonUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const json = JSON.parse(evt.target.result);
        
        // Validation based on schema
        if (!json.chapter || !Array.isArray(json.analysis)) {
          throw new Error("Invalid JSON schema: Missing 'chapter' or 'analysis' array.");
        }

        const chapter = json.chapter;
        const newMap = { ...analysisData }; // Preserve existing data if any

        if (!newMap[chapter]) newMap[chapter] = {};

        // Map the array to a verse-lookup object
        json.analysis.forEach(item => {
          if (item.verse) {
            newMap[chapter][item.verse] = item;
          }
        });

        setAnalysisData(newMap);
        setError(null);
      } catch (err) {
        setError("Failed to parse JSON file. Ensure it matches the schema.");
      }
    };
    reader.readAsText(file);
  };

  // --- Derived Data Helpers ---

  const getDebateInfo = (verseAnalysis) => {
    if (!verseAnalysis || !verseAnalysis.analysis) return null;
    return verseAnalysis.analysis.find(a => a.type === 'debate');
  };

  const getIndividualAnalyses = (verseAnalysis) => {
    if (!verseAnalysis || !verseAnalysis.analysis) return [];
    return verseAnalysis.analysis.filter(a => a.type === 'individual');
  };

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
        // Check for Analysis match
        const verseData = analysisData?.[activeChapter]?.[vNum];
        const debate = getDebateInfo(verseData);
        
        return {
          vNum,
          text,
          debateScore: debate ? debate.score : null
        };
      });
  }, [usfmData, analysisData, activeChapter]);

  const selectedVerseData = useMemo(() => {
    if (!activeVerse || !usfmData) return null;
    const { c, v } = activeVerse;
    
    const rawAnalysis = analysisData?.[c]?.[v];

    return {
      c, v,
      text: usfmData[c]?.[v],
      rawAnalysis,
      debate: getDebateInfo(rawAnalysis),
      individuals: getIndividualAnalyses(rawAnalysis)
    };
  }, [activeVerse, usfmData, analysisData]);


  // --- Sub-components for Right Panel ---

  const IndividualAnalysisSection = ({ models }) => {
    const [selectedModelIndex, setSelectedModelIndex] = useState(0);

    if (!models || models.length === 0) return <p className="text-gray-500 italic">No individual models found.</p>;

    const currentModel = models[selectedModelIndex];

    return (
      <div className="space-y-4">
        <div>
          <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Select Model</label>
          <select 
            value={selectedModelIndex} 
            onChange={(e) => setSelectedModelIndex(parseInt(e.target.value))}
            className="w-full bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5"
          >
            {models.map((m, idx) => (
              <option key={idx} value={idx}>{m.model} (Score: {m.score})</option>
            ))}
          </select>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-3">
             <span className="font-bold text-gray-700">{currentModel.model}</span>
             <span className={`text-xs font-bold px-2 py-1 rounded ${getScoreBadgeColor(currentModel.score)}`}>
               Score: {currentModel.score}
             </span>
          </div>
          <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
            <Markdown>
                {currentModel.reasoning}
            </Markdown>
          </p>
        </div>
      </div>
    );
  };

  const DebateAnalysisSection = ({ debate }) => {
    if (!debate) return <p className="text-gray-500 italic">No debate data available.</p>;

    return (
      <div className="space-y-2">
        {/* Debate Transcript */}
        <CollapsibleCard title="Debate Transcript" icon={MessageSquare} defaultOpen={true}>
          <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
            {debate.debate_transcript.filter((debate_item) => ( (debate_item.role !== "moderator" || debate_item.intervened === true)? true : false)).map((turn, idx) => (
              <div key={idx} className={`flex gap-3 ${turn.role === 'moderator' ? 'bg-blue-50 p-3 rounded-lg border border-blue-100' : ''}`}>
                <div className={`mt-1 flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${turn.role === 'moderator' ? 'bg-blue-200 text-blue-800' : 'bg-gray-200 text-gray-600'}`}>
                  {turn.role === 'moderator' ? 'M' : turn.agent.charAt(0)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold uppercase text-gray-500">{turn.agent} ({turn.role})</span>
                    {turn.proposed_score && (
                       <span className="text-xs font-mono bg-gray-100 px-1 rounded">Score: {turn.proposed_score}</span>
                    )}
                  </div>
                  <p className="text-sm text-gray-800">
                    <Markdown>
                        {turn.argument || turn.feedback}
                    </Markdown>
                  </p>
                  {turn.violators && turn.violators.length > 0 && (
                    <div className="mt-2 text-xs text-red-600 flex items-center gap-1">
                      <AlertCircle size={12} />
                      Violations: {turn.violators.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleCard>

        {/* Closing Statements */}
        <CollapsibleCard title="Closing Statements" icon={Gavel}>
          <div className="space-y-4">
            {debate.closing_statements.map((stmt, idx) => (
              <div key={idx} className="bg-gray-50 p-3 rounded-lg border border-gray-100">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-sm text-gray-700">{stmt.agent}</span>
                  <span className={`text-xs font-bold px-2 py-0.5 rounded ${getScoreBadgeColor(stmt.score)}`}>
                    Final: {stmt.score}
                  </span>
                </div>
                <p className="text-sm text-gray-600">
                    <Markdown>
                        {stmt.statement}
                    </Markdown>
                </p>
              </div>
            ))}
          </div>
        </CollapsibleCard>
      </div>
    );
  };


  // --- Render ---

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 font-sans">
      
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg text-white">
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

          {/* JSON Upload */}
          <div className="relative group">
            <input 
              type="file" 
              accept=".json" 
              onChange={handleJsonUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <button className={`flex items-center gap-2 px-4 py-2 rounded-md border transition-colors ${analysisData ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-white border-gray-300 text-gray-600 hover:bg-gray-50'}`}>
              <BarChart2 size={16} />
              <span className="text-sm font-medium">{analysisData ? 'Data Loaded' : 'Upload JSON'}</span>
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
              {currentVerses.map(({ vNum, text, debateScore }) => {
                const bgColorClass = debateScore ? getScoreColor(debateScore) : 'bg-white border border-gray-100 hover:border-blue-300';
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
                      <p className={`text-lg leading-relaxed ${debateScore ? '' : 'text-gray-700'}`}>
                        {text}
                      </p>
                      {debateScore && (
                        <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-white/20 text-xs font-bold backdrop-blur-sm">
                          {debateScore}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Analysis Panel */}
          <div className="w-1/3 min-w-[450px] bg-gray-50 flex flex-col border-l border-gray-200">
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
                  <p className="text-gray-600 italic border-l-4 border-indigo-500 pl-4 py-1 mb-2">
                    "{selectedVerseData.text}"
                  </p>
                  
                  {/* Metadata if available */}
                  {selectedVerseData.rawAnalysis && (
                    <div>
                      <p className="text-gray-600 italic border-l-4 border-cyan-500 pl-4 py-1 mb-2">
                        "{selectedVerseData.rawAnalysis.greek}"
                      </p>
                     <div className="flex flex-wrap gap-2 text-xs mt-3">
                        <span className="px-2 py-1 bg-blue-50 text-blue-700 rounded border border-blue-100">
                           {selectedVerseData.rawAnalysis.annotation}
                        </span>
                     </div>
                    </div>
                  )}
                </div>

                <div className="flex-1 overflow-y-auto p-6 scrollbar-thin">
                  {selectedVerseData.rawAnalysis ? (
                    <div className="space-y-6">
                      
                      {/* Top Score Card (Debate Score) */}
                      {selectedVerseData.debate && (
                        <div className={`p-5 rounded-xl border ${getScoreBadgeColor(selectedVerseData.debate.score)}`}>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-bold uppercase tracking-wide opacity-80">Consensus Score</span>
                            <span className="text-3xl font-black">{selectedVerseData.debate.score}</span>
                          </div>
                          <div className="w-full bg-black/10 h-2 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-current opacity-50" 
                              style={{ width: `${Math.min(selectedVerseData.debate.score * 10, 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      )}

                      {/* Card 1: Individual Analysis */}
                      <CollapsibleCard title="Individual Analysis" icon={User}>
                        <IndividualAnalysisSection models={selectedVerseData.individuals} />
                      </CollapsibleCard>

                      {/* Card 2: Debate Analysis */}
                      <CollapsibleCard title="Debate Analysis" icon={Users} defaultOpen={true}>
                         <DebateAnalysisSection debate={selectedVerseData.debate} />
                      </CollapsibleCard>

                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center text-gray-400">
                      <BarChart2 size={48} className="mb-4 opacity-20" />
                      <p>No linguistic data available for this verse.</p>
                      <p className="text-sm mt-2">Upload a JSON file to see analysis.</p>
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
            <div className="w-16 h-16 bg-indigo-100 text-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <Upload size={32} />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Start your Analysis</h2>
            <p className="text-gray-500 mb-8">
              Upload a standard USFM file to parse scripture, then attach your JSON data for deep linguistic insights.
            </p>
            
            <div className="space-y-3">
              <label className="block w-full">
                <div className="w-full px-4 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg cursor-pointer transition-colors flex items-center justify-center gap-2">
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