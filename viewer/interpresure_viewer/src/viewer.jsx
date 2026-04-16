import React, { useState, useMemo, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
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
  Gavel,
  Layers,
  AlignLeft,
  Settings,
  Brain,
  MessageCircle,
  Layout,
  Scale,
  Smile,
  Tag,
  ArrowRight,
  GripVertical
} from 'lucide-react';

// --- Markdown Styling Components ---
const markdownComponents = {
  ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
  ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3 space-y-1" {...props} />,
  li: ({node, ...props}) => <li className="pl-1" {...props} />,
  h1: ({node, ...props}) => <h1 className="text-xl font-bold text-gray-900 mt-4 mb-2" {...props} />,
  h2: ({node, ...props}) => <h2 className="text-lg font-bold text-gray-900 mt-3 mb-2" {...props} />,
  h3: ({node, ...props}) => <h3 className="text-base font-bold text-gray-900 mt-2 mb-1" {...props} />,
  h4: ({node, ...props}) => <h4 className="text-sm font-bold text-gray-900 mt-2 mb-1" {...props} />,
  p: ({node, ...props}) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
  strong: ({node, ...props}) => <strong className="font-bold text-gray-900" {...props} />,
  em: ({node, ...props}) => <em className="italic text-gray-800" {...props} />,
  blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-gray-200 pl-4 italic text-gray-600 my-2" {...props} />,
  code: ({node, ...props}) => <code className="bg-gray-100 rounded px-1 py-0.5 text-xs font-mono text-gray-800" {...props} />,
};

// --- Color Scale Logic ---
const getScoreColor = (score) => {
  const s = parseFloat(score);
  if (isNaN(s)) return 'bg-gray-100 text-gray-500'; 

  // 1-10 Scale mapping
  if (s >= 10) return 'bg-green-500 text-white';
  if (s >= 9) return 'bg-green-300 text-gray-900';
  if (s >= 8) return 'bg-lime-300 text-gray-900';
  if (s >= 7) return 'bg-yellow-200 text-gray-900';
  if (s >= 6) return 'bg-yellow-500 text-white';
  if (s >= 5) return 'bg-orange-300 text-gray-900';
  if (s >= 4) return 'bg-orange-400 text-white';
  if (s >= 3) return 'bg-red-400 text-white';
  if (s >= 2) return 'bg-red-500 text-white';
  if (s <= 1) return 'bg-red-700 text-white';
  
  return 'bg-gray-100 text-gray-500';
};

const getScoreBadgeColor = (score) => {
  const s = parseFloat(score);
  if (isNaN(s)) return 'bg-gray-100 text-gray-800';
  if (s >= 8) return 'bg-green-100 text-green-800 border-green-200';
  if (s >= 6) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
  return 'bg-red-100 text-red-800 border-red-200';
};

const getConfidenceBadgeColor = (confidence) => {
  const c = parseFloat(confidence);
  if (isNaN(c)) return 'bg-gray-100 text-gray-800 border-gray-200';
  if (c >= 80) return 'bg-emerald-100 text-emerald-800 border-emerald-200';
  if (c >= 60) return 'bg-amber-100 text-amber-800 border-amber-200';
  return 'bg-rose-100 text-rose-800 border-rose-200';
};

const buildSummaryMarkdown = (analysis) => {
  if (!analysis) return '';
  const parts = [];
  if (analysis.strengths) parts.push(`### Strengths\n${analysis.strengths}`);
  if (analysis.weaknesses) parts.push(`### Weaknesses\n${analysis.weaknesses}`);
  if (analysis.suggestions) parts.push(`### Suggestions\n${analysis.suggestions}`);
  return parts.join('\n\n');
};

const getScoreDotColor = (score) => {
  const s = parseFloat(score);
  if (isNaN(s)) return 'bg-gray-300';
  if (s >= 8) return 'bg-green-500';
  if (s >= 6) return 'bg-yellow-400';
  return 'bg-red-500';
};

// --- Helper Functions ---

const getGoalIcon = (type) => {
  switch (type?.toLowerCase()) {
    case 'logical': return Brain;
    case 'implicature': return MessageCircle;
    case 'structure': return Layout;
    case 'social': return Users;
    case 'scalar': return Scale;
    default: return Layers;
  }
};

const getGoalShortName = (type) => {
   switch (type?.toLowerCase()) {
    case 'logical': return 'LOG';
    case 'implicature': return 'IMP';
    case 'structure': return 'STR';
    case 'social': return 'SOC';
    case 'scalar': return 'SCL';
    default: return type?.substring(0, 3).toUpperCase() || '???';
  }
};

// --- Parsers ---

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
      if (verseText) book[chapterNum][verseNum] = verseText;
    }
  }
  return book;
};

// --- Score & Goal Helpers ---

const getLowestScore = (verseData) => {
  if (!verseData || !verseData.analysis) return null;
  const scores = verseData.analysis
    .map(a => {
        if (a.type === 'debate') return a.score;
        if (a.type === 'individual') return a.score;
        return null;
    })
    .filter(s => s != null && !isNaN(s));
  
  if (scores.length === 0) return null;
  return Math.min(...scores);
};

const getGoalsAndScoreForVerse = (allContexts) => {
  const groupedGoals = new Map();
  
  allContexts.forEach(ctx => {
      const type = ctx.goal.type;
      if (!groupedGoals.has(type)) {
          groupedGoals.set(type, {
              type: type,
              title: ctx.goal.title,
              variants: [],
              minScore: null
          });
      }
      const group = groupedGoals.get(type);
      group.variants.push(ctx);
      
      const s = getLowestScore(ctx.verseData);
      if (s !== null) {
          if (group.minScore === null || s < group.minScore) {
              group.minScore = s;
          }
      }
  });

  const goals = Array.from(groupedGoals.values());
  
  // Calculate total score as an average of the minimum scores
  let totalScoreSum = 0;
  let validScoreCount = 0;
  goals.forEach(g => {
      if (g.minScore !== null && !isNaN(g.minScore)) {
          totalScoreSum += g.minScore;
          validScoreCount++;
      }
  });
  
  const totalScore = validScoreCount > 0 
    ? Math.round((totalScoreSum / validScoreCount) * 10) / 10 
    : null;

  return { goals, totalScore };
};

// --- Sub-Components ---

const ScoreBar = ({ score, label, max = 10, isTotal = false, onClick }) => {
    if (score === null || isNaN(score)) return null;
    const percentage = Math.min((score / max) * 100, 100);
    
    let color = 'bg-green-500';
    if (score < 6) color = 'bg-red-500';
    else if (score < 8) color = 'bg-yellow-400';

    const Container = onClick ? 'button' : 'div';
    const interactiveStyles = onClick 
        ? 'hover:bg-gray-50 p-2 -ml-2 rounded-lg transition-colors cursor-pointer group hover:shadow-sm border border-transparent hover:border-gray-100' 
        : '';

    return (
        <Container 
            onClick={onClick}
            className={`flex items-center gap-3 w-full text-left ${isTotal ? 'mb-2' : ''} ${interactiveStyles}`}
        >
            <span className={`w-32 ${isTotal ? 'text-sm font-bold text-gray-800' : 'text-xs font-medium text-gray-600'} ${onClick ? 'group-hover:text-indigo-600 transition-colors' : ''}`}>
                {label}
            </span>
            <div className={`flex-1 ${isTotal ? 'h-3' : 'h-2'} bg-gray-100 rounded-full overflow-hidden`}>
                <div 
                    className={`h-full ${color} rounded-full transition-all duration-500`} 
                    style={{ width: `${percentage}%` }}
                ></div>
            </div>
            <span className={`w-8 text-right ${isTotal ? 'text-sm font-bold' : 'text-xs font-medium'} text-gray-700`}>
                {score}
            </span>
            {onClick && (
                <div className="w-4 flex justify-end">
                    <ChevronRight size={14} className="text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
            )}
        </Container>
    );
};

const CollapsibleCard = ({ title, icon: Icon, children, defaultOpen = false, score = null, className = "" }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden ${className}`}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon size={18} className="text-gray-500" />}
          <span className="font-semibold text-gray-700 text-left">{title}</span>
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

const AnnotationBadge = ({ type, value }) => (
  <div className="flex flex-col bg-gray-50 border border-gray-200 rounded p-2 text-xs">
    <span className="font-bold text-gray-400 uppercase tracking-wider mb-1 text-[10px]">{type.replace(/_/g, ' ')}</span>
    <span className="font-medium text-gray-800">{value}</span>
  </div>
);

const VerseContextCard = ({ greek, translation }) => (
    <div className="bg-blue-50/50 p-5 rounded-xl border border-blue-100 shadow-sm">
        <div className="grid grid-cols-1 gap-4">
            {greek && (
                <div>
                    <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-blue-400 tracking-wider mb-1">
                        <BookOpen size={10} /> Source Text
                    </span>
                    <p className="font-serif text-lg text-gray-800 leading-snug bg-white/60 p-2 rounded border border-blue-100/50">
                        {greek}
                    </p>
                </div>
            )}
            {translation && (
                <div>
                    <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-blue-400 tracking-wider mb-1">
                        <FileText size={10} /> Target Translation
                    </span>
                    <p className="text-gray-700 leading-relaxed italic bg-white/60 p-2 rounded border border-blue-100/50">
                        "{translation}"
                    </p>
                </div>
            )}
        </div>
    </div>
);

// --- Main Application ---

export default function BibleAnalyzer() {
  const [usfmData, setUsfmData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null); 
  const [verseReports, setVerseReports] = useState(null);
  const [activeChapter, setActiveChapter] = useState(1);
  const [activeSelection, setActiveSelection] = useState(null); 
  const [expandedVerses, setExpandedVerses] = useState(new Set());
  const [sidebarWidth, setSidebarWidth] = useState(450); 
  const sidebarRef = useRef(null);
  const isResizing = useRef(false);
  const [error, setError] = useState(null);

  // --- Handlers ---

  const handleJsonUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const json = JSON.parse(evt.target.result);
        
        // 1. Handle Embedded USFM
        if (json.translation && json.translation.usfm) {
          const parsedUsfm = parseUSFM(json.translation.usfm);
          setUsfmData(parsedUsfm);
          const firstCh = Object.keys(parsedUsfm)[0];
          if (firstCh) setActiveChapter(parseInt(firstCh));
        }

        // 2. Handle Reports (Verse Summaries)
        if (json.reports) {
            setVerseReports(json.reports);
        }

        // 3. Handle Evaluation/Analysis Data
        let rawData = [];
        const evaluationData = json.evaluation || json.evauluation;

        if (evaluationData && Array.isArray(evaluationData)) {
          rawData = evaluationData;
        } else if (json.analysis && Array.isArray(json.analysis)) {
           // Backwards compatibility
           rawData = [{
             chapter: json.chapter,
             pragmatic_goal: { type: 'general', title: 'General Analysis', goal: 'General', description: '' },
             analysis: json.analysis
           }];
        } else {
           if (!json.translation) throw new Error("Invalid Schema: Missing 'evaluation' data.");
        }

        const refinedMap = {};
        rawData.forEach(evalEntry => {
           const ch = evalEntry.chapter;
           if (!refinedMap[ch]) refinedMap[ch] = {};

           evalEntry.analysis.forEach(verseItem => {
              const v = verseItem.verse;
              if (!refinedMap[ch][v]) refinedMap[ch][v] = [];
              
              refinedMap[ch][v].push({
                 goal: evalEntry.pragmatic_goal,
                 verseData: verseItem 
              });
           });
        });

        setAnalysisData(refinedMap);
        setError(null);
      } catch (err) {
        console.error(err);
        setError("Failed to parse JSON file. " + err.message);
      }
    };
    reader.readAsText(file);
  };

  const toggleVerseExpansion = (verseNum) => {
    setExpandedVerses(prev => {
      const next = new Set(prev);
      if (next.has(verseNum)) {
        next.delete(verseNum);
      } else {
        next.add(verseNum);
      }
      return next;
    });
  };

  // --- Resize Handlers ---
  const startResizing = React.useCallback(() => {
    isResizing.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  const stopResizing = React.useCallback(() => {
    isResizing.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  const resize = React.useCallback((mouseMoveEvent) => {
    if (isResizing.current) {
        const newWidth = mouseMoveEvent.clientX;
        if (newWidth > 300 && newWidth < window.innerWidth - 300) {
            setSidebarWidth(newWidth);
        }
    }
  }, []);

  useEffect(() => {
    window.addEventListener("mousemove", resize);
    window.addEventListener("mouseup", stopResizing);
    return () => {
      window.removeEventListener("mousemove", resize);
      window.removeEventListener("mouseup", stopResizing);
    };
  }, [resize, stopResizing]);

  // --- Derived Data Helpers ---

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
        const allContexts = analysisData?.[activeChapter]?.[vNum] || [];
        
        const { goals, totalScore } = getGoalsAndScoreForVerse(allContexts);

        // Check if report exists for this verse
        const hasReport = verseReports?.[activeChapter]?.[vNum] !== undefined;

        return {
          vNum,
          text,
          goals,
          totalScore,
          hasReport
        };
      });
  }, [usfmData, analysisData, verseReports, activeChapter]);

  const selectedContent = useMemo(() => {
    if (!activeSelection) return null;
    const { c, v, type } = activeSelection;

    // Common verse text
    const verseText = usfmData?.[c]?.[v] || '';

    if (type === 'summary') {
        const allContexts = analysisData?.[c]?.[v] || [];
        if (allContexts.length === 0) return null;

        const activeContext = allContexts[0];
        const verseData = activeContext.verseData || {};
        const analysisItems = (verseData.analysis || []).filter((item) => item && item.type === 'individual');
        const scores = analysisItems
          .map((item) => item.score)
          .filter((score) => score != null && !isNaN(score));
        const totalScore = scores.length > 0 ? Math.min(...scores) : null;

        return {
            type: 'verse',
            c, v,
            verseText,
            greekSource: verseData.biblical_text || '',
            verseData,
            analysisItems,
            totalScore,
        };
    }

    return null;
  }, [activeSelection, analysisData, verseReports, usfmData]);


  // --- Sub-components ---

  const VerseSummaryView = ({ selection }) => {
    const [showDetails, setShowDetails] = useState(false);
    const [modelIdx, setModelIdx] = useState(0);
    const { c, v, verseText, greekSource, verseData, analysisItems, totalScore } = selection;
    const current = analysisItems[modelIdx] || null;
    const summaryMarkdown = buildSummaryMarkdown(current);
    const reasoningMarkdown = current?.reasoning || current?.model_analysis || '';

    return (
        <div className="flex flex-col h-full overflow-hidden">
            <div className="p-6 border-b border-gray-200 bg-white shadow-sm flex-shrink-0">
                <div className="flex items-center gap-2 mb-3">
                     <span className="px-2 py-0.5 rounded-md bg-gray-100 text-gray-500 text-xs font-bold uppercase tracking-wider">
                       Chapter {c} : {v}
                     </span>
                     <ChevronRight size={14} className="text-gray-300" />
                     <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-green-50 text-green-700 text-xs font-bold uppercase tracking-wider">
                       <AlignLeft size={12} />
                       Verse Analysis
                     </span>
                </div>
                <h2 className="text-2xl font-serif text-gray-800 mb-1 leading-tight">
                    Verse Overview
                </h2>
            </div>

            <div className="flex-1 overflow-y-auto p-6 scrollbar-thin">
                <div className="space-y-6">
                    <VerseContextCard greek={greekSource} translation={verseText} />

                    {totalScore !== null && (
                        <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                            <div className="flex items-center justify-between mb-4">
                                <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                                    <BarChart2 size={14} />
                                    Analysis Score
                                </h4>
                                {analysisItems.length > 1 && (
                                    <select
                                        className="text-xs border border-gray-300 rounded-md px-2 py-1"
                                        value={modelIdx}
                                        onChange={(e) => setModelIdx(Number(e.target.value))}
                                    >
                                        {analysisItems.map((item, idx) => (
                                            <option key={idx} value={idx}>
                                                {item.model || `Model ${idx + 1}`}
                                            </option>
                                        ))}
                                    </select>
                                )}
                            </div>

                            <div className="flex items-center gap-2 flex-wrap mb-4">
                                <span className={`text-xs px-2 py-1 rounded font-bold border ${getScoreBadgeColor(current?.score)}`}>
                                    Score: {current?.score ?? totalScore}
                                </span>
                                {current?.confidence != null && (
                                    <span className={`text-xs px-2 py-1 rounded font-bold border ${getConfidenceBadgeColor(current.confidence)}`}>
                                        Confidence: {current.confidence}%
                                    </span>
                                )}
                            </div>

                            <ScoreBar score={current?.score ?? totalScore} label="Score" isTotal={true} />
                        </div>
                    )}

                    <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                        <div className="flex items-center justify-between mb-3">
                            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                                <AlignLeft size={14} />
                                Simplified Summary
                            </h4>
                            {current?.confidence != null && (
                                <span className="text-[11px] text-gray-500">
                                    Analyst confidence: {current.confidence}%
                                </span>
                            )}
                        </div>
                        <div className="text-gray-800 leading-relaxed text-sm prose prose-sm max-w-none">
                            {summaryMarkdown ? (
                                <ReactMarkdown components={markdownComponents}>{summaryMarkdown}</ReactMarkdown>
                            ) : (
                                <p className="text-gray-400 italic">No summary available.</p>
                            )}
                        </div>
                    </div>

                    <div className="bg-white p-5 rounded-xl shadow-sm border border-gray-200">
                        <div className="flex items-center justify-between mb-3">
                            <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                                <FileText size={14} />
                                Details
                            </h4>
                            <button
                                onClick={() => setShowDetails(!showDetails)}
                                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${showDetails ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                            >
                                <Settings size={12} />
                                {showDetails ? "Hide Details" : "Show Details"}
                            </button>
                        </div>
                        {showDetails ? (
                            <div className="text-gray-800 leading-relaxed text-sm prose prose-sm max-w-none">
                                {reasoningMarkdown ? (
                                    <ReactMarkdown components={markdownComponents}>{reasoningMarkdown}</ReactMarkdown>
                                ) : (
                                    <p className="text-gray-400 italic">No reasoning available.</p>
                                )}
                            </div>
                        ) : (
                            <p className="text-sm text-gray-500 italic">
                                Select "Show Details" to read the full reasoning behind the analysis.
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
  };

  // --- Render ---

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900 font-sans overflow-hidden">
      
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between shadow-sm flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-600 rounded-lg text-white">
            <BookOpen size={20} />
          </div>
          <h1 className="text-xl font-bold text-gray-800 tracking-tight">Scripture Linguistic Analysis</h1>
        </div>

        <div className="flex items-center gap-4">
          <div className="relative group">
            <input 
              type="file" 
              accept=".json" 
              onChange={handleJsonUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <button className={`flex items-center gap-2 px-4 py-2 rounded-md border transition-colors ${analysisData ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-white border-gray-300 text-gray-600 hover:bg-gray-50'}`}>
              <Upload size={16} />
              <span className="text-sm font-medium">{analysisData ? 'Data Loaded' : 'Upload JSON Analysis'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {usfmData ? (
        <div className="flex flex-1 overflow-hidden" onMouseUp={stopResizing}>
          
          {/* Left Column: Verse List & Goal Accordion */}
          <div 
             className="flex flex-col min-w-[300px] border-r border-gray-200 bg-white"
             style={{ width: sidebarWidth }}
          >
            
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

            {/* Verse Navigation List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {currentVerses.map(({ vNum, text, totalScore }) => {
                const hasSelectionInThisVerse = activeSelection?.c === activeChapter && activeSelection?.v === vNum;
                return (
                  <div key={vNum} className="border border-gray-100 rounded-lg overflow-hidden bg-white hover:border-gray-300 transition-colors shadow-sm">
                    <div 
                      onClick={() => setActiveSelection({ c: activeChapter, v: vNum, type: 'summary' })}
                      className={`
                        p-3 cursor-pointer flex gap-3 items-start relative transition-colors
                        ${hasSelectionInThisVerse ? 'bg-indigo-50 border-l-4 border-indigo-500' : 'hover:bg-gray-50 border-l-4 border-transparent'}
                      `}
                    >
                      <span className={`flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-full text-xs font-bold mt-0.5 ${hasSelectionInThisVerse ? 'bg-indigo-200 text-indigo-700' : 'bg-gray-100 text-gray-500'}`}>
                        {vNum}
                      </span>
                      
                      <div className="flex-1 min-w-0">
                         <div className="flex items-start justify-between gap-4">
                             <p className={`text-sm text-gray-800 whitespace-normal ${hasSelectionInThisVerse ? 'font-medium' : ''}`}>
                                {text}
                             </p>

                             <div className="flex flex-col items-end gap-2 flex-shrink-0">
                                <div className="flex flex-wrap gap-1 justify-end">
                                    {totalScore !== null ? (
                                        <span 
                                          className={`px-2 py-0.5 flex items-center justify-center text-xs font-bold rounded ${getScoreBadgeColor(totalScore)}`}
                                          title={`Total Average Score: ${totalScore}`}
                                        >
                                          {totalScore}
                                        </span>
                                    ) : (
                                        <span className="text-xs text-gray-400">-</span>
                                    )}
                                </div>
                             </div>
                         </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          
          {/* Drag Handle */}
          <div
            className="w-1 bg-gray-200 hover:bg-indigo-400 cursor-col-resize flex items-center justify-center transition-colors z-20"
            onMouseDown={startResizing}
          >
            <GripVertical size={12} className="text-gray-400" />
          </div>

          {/* Right Column: Analysis Detail Panel */}
          <div className="flex-1 bg-gray-50 flex flex-col min-w-[400px]">
            {selectedContent ? (
                <VerseSummaryView selection={selectedContent} />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 p-8">
                <div className="bg-white p-6 rounded-full shadow-sm mb-4">
                     <Layers size={48} className="opacity-20 text-indigo-500" />
                </div>
                <h3 className="text-lg font-medium text-gray-600">Select Verse</h3>
                <p className="text-sm mt-2 max-w-xs leading-relaxed">
                  Click a verse row to see the analysis summary and reasoning.
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
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Analysis Dashboard</h2>
            <p className="text-gray-500 mb-8">
              Upload your JSON Analysis file. It should contain both the translation text (USFM) and the linguistic evaluation data.
            </p>
            
            <div className="space-y-3">
              <label className="block w-full">
                <div className="w-full px-4 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg cursor-pointer transition-colors flex items-center justify-center gap-2">
                  <FileText size={18} />
                  Upload JSON Analysis
                </div>
                <input type="file" accept=".json" onChange={handleJsonUpload} className="hidden" />
              </label>
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
