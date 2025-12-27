// ============================================================
// Path Finder Component for Ontology Graph
// ============================================================
import React, { useState } from 'react';
import classNames from 'classnames';

interface PathFinderProps {
  nodeTypes: string[];
  onFindPath: (from: { type: string; id: string }, to: { type: string; id: string }) => void;
  onClearPath: () => void;
  isSearching: boolean;
  pathResult?: {
    found: boolean;
    length: number;
    nodes: string[];
  } | null;
}

export const PathFinder: React.FC<PathFinderProps> = ({
  nodeTypes,
  onFindPath,
  onClearPath,
  isSearching,
  pathResult,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [fromType, setFromType] = useState('Equipment');
  const [fromId, setFromId] = useState('');
  const [toType, setToType] = useState('Alarm');
  const [toId, setToId] = useState('');

  const handleSearch = () => {
    if (fromId.trim() && toId.trim()) {
      onFindPath(
        { type: fromType, id: fromId },
        { type: toType, id: toId }
      );
    }
  };

  return (
    <div className="absolute top-4 right-16 bg-slate-800/95 backdrop-blur-sm rounded-xl border border-slate-700 shadow-xl z-20">
      {/* Toggle Button */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={classNames(
          'flex items-center space-x-2 px-4 py-2 transition-colors',
          expanded ? 'border-b border-slate-700' : ''
        )}
      >
        <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
        </svg>
        <span className="text-sm font-medium text-white">Path Finder</span>
        <svg
          className={classNames('w-4 h-4 text-slate-400 transition-transform', expanded && 'rotate-180')}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="p-4 w-72 space-y-4">
          {/* From Node */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wide">From</label>
            <div className="flex space-x-2">
              <select
                value={fromType}
                onChange={(e) => setFromType(e.target.value)}
                className="select flex-1 text-sm"
              >
                {nodeTypes.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
              <input
                type="text"
                value={fromId}
                onChange={(e) => setFromId(e.target.value)}
                placeholder="ID"
                className="input flex-1 text-sm"
              />
            </div>
          </div>

          {/* Direction Icon */}
          <div className="flex justify-center">
            <svg className="w-6 h-6 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>

          {/* To Node */}
          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wide">To</label>
            <div className="flex space-x-2">
              <select
                value={toType}
                onChange={(e) => setToType(e.target.value)}
                className="select flex-1 text-sm"
              >
                {nodeTypes.map((type) => (
                  <option key={type} value={type}>{type}</option>
                ))}
              </select>
              <input
                type="text"
                value={toId}
                onChange={(e) => setToId(e.target.value)}
                placeholder="ID"
                className="input flex-1 text-sm"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex space-x-2">
            <button
              onClick={handleSearch}
              disabled={isSearching || !fromId.trim() || !toId.trim()}
              className={classNames(
                'flex-1 btn btn-primary text-sm',
                (isSearching || !fromId.trim() || !toId.trim()) && 'opacity-50 cursor-not-allowed'
              )}
            >
              {isSearching ? (
                <span className="flex items-center justify-center">
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                  Searching...
                </span>
              ) : (
                'Find Path'
              )}
            </button>
            <button
              onClick={() => {
                onClearPath();
                setFromId('');
                setToId('');
              }}
              className="btn btn-secondary text-sm"
            >
              Clear
            </button>
          </div>

          {/* Result */}
          {pathResult && (
            <div className={classNames(
              'p-3 rounded-lg border',
              pathResult.found
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-red-500/10 border-red-500/30 text-red-400'
            )}>
              {pathResult.found ? (
                <>
                  <div className="flex items-center space-x-2 mb-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="font-medium">Path Found!</span>
                  </div>
                  <p className="text-sm text-slate-300">
                    {pathResult.length} hops: {pathResult.nodes.join(' → ')}
                  </p>
                </>
              ) : (
                <div className="flex items-center space-x-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="font-medium">No path found</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
