// ============================================================
// AgentChat Component - FDC Analysis Agent Chat Interface
// with RAG Knowledge Search Integration
// ============================================================
import React, { useState, useRef, useEffect } from 'react';
import classNames from 'classnames';

interface ToolCall {
  tool: string;
  params: Record<string, any>;
  success: boolean;
  message: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  executionTime?: number;
  ragSources?: RagSource[];
  toolCalls?: ToolCall[];
}

interface RagSource {
  id: string;
  doc_type: string;
  source: string;
  similarity: number;
  preview: string;
}

interface RagStats {
  total_documents: number;
  documents_by_type: Record<string, number>;
}

interface AgentChatProps {
  className?: string;
}

const API_URL = 'http://localhost:8010';

const EXAMPLE_QUESTIONS = [
  'ETCH-001 온도 알람 발생. 원인은?',
  'Cell 용량 불량이 발생했습니다. Roll 공정부터 점검 순서를 알려주세요.',
  'Module에서 발열 이상이 감지되었습니다. 원인과 점검 방법은?',
  'Pack EOL 테스트에서 절연저항 불량이 발생했습니다.',
];

const DOC_TYPE_LABELS: Record<string, string> = {
  domain_knowledge: '도메인 지식',
  causal_relationship: '인과관계',
  alarm_cause: '알람 원인',
  leading_indicator: '선행 지표',
  impossible_relationship: '불가능 관계',
  discovered_relationship: '발견된 관계',
  discovery_summary: '발견 요약',
};

const DOC_TYPE_COLORS: Record<string, string> = {
  domain_knowledge: 'bg-blue-500/20 text-blue-400',
  causal_relationship: 'bg-purple-500/20 text-purple-400',
  alarm_cause: 'bg-red-500/20 text-red-400',
  leading_indicator: 'bg-amber-500/20 text-amber-400',
  impossible_relationship: 'bg-slate-500/20 text-slate-400',
  discovered_relationship: 'bg-emerald-500/20 text-emerald-400',
  discovery_summary: 'bg-cyan-500/20 text-cyan-400',
};

const TOOL_LABELS: Record<string, string> = {
  ontology_search: '온톨로지 검색',
  time_series_analysis: '시계열 분석',
  pattern_mining: '패턴 마이닝',
  root_cause_analysis: '근본원인 분석',
  alarm_history: '알람 이력',
};

const TOOL_ICONS: Record<string, string> = {
  ontology_search: '🔍',
  time_series_analysis: '📊',
  pattern_mining: '🔗',
  root_cause_analysis: '🎯',
  alarm_history: '🔔',
};

export const AgentChat: React.FC<AgentChatProps> = ({ className }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [ragStats, setRagStats] = useState<RagStats | null>(null);
  const [showRagPanel, setShowRagPanel] = useState(true);
  const [currentRagSources, setCurrentRagSources] = useState<RagSource[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Check agent status on mount
  useEffect(() => {
    checkAgentStatus();
    checkRagStatus();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkAgentStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/agent/status`);
      const data = await response.json();
      setIsConnected(data.available);
    } catch (error) {
      setIsConnected(false);
    }
  };

  const checkRagStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/agent/rag/status`);
      const data = await response.json();
      if (data.available) {
        setRagStats(data.stats);
      }
    } catch (error) {
      console.error('Failed to check RAG status:', error);
    }
  };

  const searchRag = async (query: string): Promise<RagSource[]> => {
    try {
      const response = await fetch(`${API_URL}/api/agent/rag/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, n_results: 5 }),
      });
      const data = await response.json();
      return data.sources || [];
    } catch (error) {
      console.error('RAG search failed:', error);
      return [];
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setCurrentRagSources([]);

    try {
      // Parallel: RAG search and Agent analysis
      const [ragSources, agentResponse] = await Promise.all([
        searchRag(userMessage.content),
        fetch(`${API_URL}/api/agent/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: userMessage.content }),
        }).then(res => res.json()),
      ]);

      setCurrentRagSources(ragSources);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: agentResponse.success ? agentResponse.answer : `Error: ${agentResponse.error || 'Unknown error'}`,
        timestamp: new Date(),
        executionTime: agentResponse.execution_time_ms,
        ragSources: ragSources,
        toolCalls: agentResponse.tool_calls || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Failed to connect to Agent server. Please check if the server is running.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExampleClick = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentRagSources([]);
  };

  const formatContent = (content: string) => {
    return content
      .split('\n')
      .map((line, i) => {
        if (line.startsWith('### ')) {
          return <h3 key={i} className="text-lg font-bold mt-4 mb-2 text-blue-400">{line.slice(4)}</h3>;
        }
        if (line.startsWith('## ')) {
          return <h2 key={i} className="text-xl font-bold mt-4 mb-2 text-blue-300">{line.slice(3)}</h2>;
        }
        if (line.includes('**')) {
          const parts = line.split(/\*\*(.*?)\*\*/g);
          return (
            <p key={i} className="my-1">
              {parts.map((part, j) => (j % 2 === 1 ? <strong key={j} className="text-white">{part}</strong> : part))}
            </p>
          );
        }
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return <li key={i} className="ml-4 my-1">{line.slice(2)}</li>;
        }
        if (line.match(/^\d+\./)) {
          return <li key={i} className="ml-4 my-1 list-decimal">{line.replace(/^\d+\./, '')}</li>;
        }
        if (line.startsWith('```')) {
          return null;
        }
        if (!line.trim()) {
          return <br key={i} />;
        }
        return <p key={i} className="my-1">{line}</p>;
      })
      .filter(Boolean);
  };

  return (
    <div className={classNames('flex h-full bg-slate-900 rounded-xl', className)}>
      {/* Main Chat Area */}
      <div className="flex flex-col flex-1">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">FDC Analysis Agent</h2>
              <p className="text-sm text-slate-400">EXAONE 3.5 + RAG Knowledge System</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className={classNames(
              'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm',
              isConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
            )}>
              <div className={classNames(
                'w-2 h-2 rounded-full',
                isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
              )} />
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
            <button
              onClick={() => setShowRagPanel(!showRagPanel)}
              className={classNames(
                'p-2 rounded-lg transition-colors',
                showRagPanel ? 'bg-purple-500/20 text-purple-400' : 'text-slate-400 hover:text-white hover:bg-slate-700'
              )}
              title="Toggle RAG Panel"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </button>
            <button
              onClick={clearChat}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
              title="Clear chat"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-slate-300 mb-2">Start a conversation</h3>
              <p className="text-slate-500 mb-6 max-w-md">
                Ask about equipment alarms, quality issues, or manufacturing process problems.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full max-w-2xl">
                {EXAMPLE_QUESTIONS.map((question, i) => (
                  <button
                    key={i}
                    onClick={() => handleExampleClick(question)}
                    className="text-left px-4 py-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm transition-colors border border-slate-700"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={classNames(
                  'flex gap-3',
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                {message.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                )}
                <div
                  className={classNames(
                    'max-w-[80%] rounded-xl px-4 py-3',
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300'
                  )}
                >
                  {message.role === 'user' ? (
                    <p>{message.content}</p>
                  ) : (
                    <>
                      {/* Tool Calls Display */}
                      {message.toolCalls && message.toolCalls.length > 0 && (
                        <div className="mb-3 p-3 bg-slate-900/50 rounded-lg border border-slate-700">
                          <p className="text-xs text-slate-500 mb-2 font-medium">도구 호출 결과</p>
                          <div className="flex flex-wrap gap-2">
                            {message.toolCalls.map((tool, idx) => (
                              <div
                                key={idx}
                                className={classNames(
                                  'flex items-center gap-1.5 px-2 py-1 rounded-md text-xs',
                                  tool.success
                                    ? 'bg-emerald-500/20 text-emerald-400'
                                    : 'bg-red-500/20 text-red-400'
                                )}
                              >
                                <span>{TOOL_ICONS[tool.tool] || '🔧'}</span>
                                <span className="font-medium">{TOOL_LABELS[tool.tool] || tool.tool}</span>
                                <span className={tool.success ? 'text-emerald-300' : 'text-red-300'}>
                                  {tool.success ? '✓' : '✗'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="prose prose-invert prose-sm max-w-none">
                        {formatContent(message.content)}
                      </div>
                    </>
                  )}
                  {message.executionTime && (
                    <p className="text-xs text-slate-500 mt-2">
                      Response time: {(message.executionTime / 1000).toFixed(1)}s
                      {message.toolCalls && message.toolCalls.length > 0 && (
                        <span className="ml-2">
                          | {message.toolCalls.length} tools called
                        </span>
                      )}
                    </p>
                  )}
                </div>
                {message.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="bg-slate-800 rounded-xl px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                  <span className="text-slate-400 text-sm ml-2">Analyzing with RAG... (30-60 seconds)</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-slate-700">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about equipment issues, quality problems, or manufacturing processes..."
              className="flex-1 px-4 py-3 rounded-xl bg-slate-800 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 resize-none"
              rows={1}
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={classNames(
                'px-6 py-3 rounded-xl font-medium transition-all',
                input.trim() && !isLoading
                  ? 'bg-blue-600 hover:bg-blue-500 text-white'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
              )}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </form>
        </div>
      </div>

      {/* RAG Panel */}
      {showRagPanel && (
        <div className="w-80 border-l border-slate-700 flex flex-col">
          {/* RAG Header */}
          <div className="px-4 py-3 border-b border-slate-700">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              <h3 className="font-semibold text-white">RAG Knowledge</h3>
            </div>
            {ragStats && (
              <p className="text-xs text-slate-400 mt-1">
                {ragStats.total_documents} documents indexed
              </p>
            )}
          </div>

          {/* RAG Stats */}
          {ragStats && (
            <div className="px-4 py-3 border-b border-slate-700">
              <p className="text-xs text-slate-500 mb-2">Knowledge Base</p>
              <div className="flex flex-wrap gap-1">
                {Object.entries(ragStats.documents_by_type).map(([type, count]) => (
                  <span
                    key={type}
                    className={classNames(
                      'px-2 py-0.5 rounded text-xs',
                      DOC_TYPE_COLORS[type] || 'bg-slate-500/20 text-slate-400'
                    )}
                  >
                    {DOC_TYPE_LABELS[type] || type}: {count}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Retrieved Sources */}
          <div className="flex-1 overflow-y-auto p-4">
            {currentRagSources.length > 0 ? (
              <div className="space-y-3">
                <p className="text-xs text-slate-500 font-medium">Retrieved Sources</p>
                {currentRagSources.map((source, i) => (
                  <div
                    key={source.id}
                    className="bg-slate-800 rounded-lg p-3 border border-slate-700"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={classNames(
                        'px-2 py-0.5 rounded text-xs',
                        DOC_TYPE_COLORS[source.doc_type] || 'bg-slate-500/20 text-slate-400'
                      )}>
                        {DOC_TYPE_LABELS[source.doc_type] || source.doc_type}
                      </span>
                      <span className="text-xs text-emerald-400 font-medium">
                        {Math.round(source.similarity * 100)}%
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 line-clamp-3">
                      {source.preview}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <svg className="w-12 h-12 text-slate-700 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm text-slate-500">
                  {isLoading ? 'Searching knowledge base...' : 'Ask a question to see related knowledge'}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
