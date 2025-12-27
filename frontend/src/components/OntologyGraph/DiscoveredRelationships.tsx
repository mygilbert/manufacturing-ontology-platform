// ============================================================
// Discovered Relationships Panel
// 관계 발견 엔진에서 발견된 암묵적 관계 시각화
// ============================================================
import React, { useState } from 'react';
import classNames from 'classnames';

export interface DiscoveredRelation {
  id: string;
  source: string;
  target: string;
  type: 'correlation' | 'causality' | 'pattern';
  method: string;
  confidence: number;
  lag?: number;
  status: 'pending' | 'verified' | 'rejected';
  properties?: Record<string, unknown>;
}

interface DiscoveredRelationshipsProps {
  relations: DiscoveredRelation[];
  onAddToGraph: (relation: DiscoveredRelation) => void;
  onVerify: (id: string) => void;
  onReject: (id: string) => void;
  onClose: () => void;
}

const TYPE_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  correlation: { bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30' },
  causality: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30' },
  pattern: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
};

const STATUS_COLORS: Record<string, string> = {
  pending: 'bg-yellow-500/20 text-yellow-400',
  verified: 'bg-green-500/20 text-green-400',
  rejected: 'bg-red-500/20 text-red-400',
};

export const DiscoveredRelationships: React.FC<DiscoveredRelationshipsProps> = ({
  relations,
  onAddToGraph,
  onVerify,
  onReject,
  onClose,
}) => {
  const [filter, setFilter] = useState<'all' | 'correlation' | 'causality' | 'pattern'>('all');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'verified' | 'rejected'>('all');
  const [sortBy, setSortBy] = useState<'confidence' | 'type'>('confidence');

  const filteredRelations = relations
    .filter((r) => filter === 'all' || r.type === filter)
    .filter((r) => statusFilter === 'all' || r.status === statusFilter)
    .sort((a, b) => {
      if (sortBy === 'confidence') return b.confidence - a.confidence;
      return a.type.localeCompare(b.type);
    });

  const stats = {
    total: relations.length,
    correlation: relations.filter((r) => r.type === 'correlation').length,
    causality: relations.filter((r) => r.type === 'causality').length,
    pattern: relations.filter((r) => r.type === 'pattern').length,
    pending: relations.filter((r) => r.status === 'pending').length,
    verified: relations.filter((r) => r.status === 'verified').length,
  };

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-slate-800 border-l border-slate-700 shadow-2xl z-50 flex flex-col">
      {/* Header */}
      <div className="px-4 py-4 border-b border-slate-700 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Discovered Relationships</h2>
          <p className="text-xs text-slate-400 mt-1">
            {stats.total} relationships found
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-slate-700"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Stats */}
      <div className="px-4 py-3 border-b border-slate-700/50 grid grid-cols-3 gap-2">
        <div className="text-center p-2 bg-blue-500/10 rounded-lg">
          <div className="text-lg font-bold text-blue-400">{stats.correlation}</div>
          <div className="text-xs text-slate-400">Correlation</div>
        </div>
        <div className="text-center p-2 bg-purple-500/10 rounded-lg">
          <div className="text-lg font-bold text-purple-400">{stats.causality}</div>
          <div className="text-xs text-slate-400">Causality</div>
        </div>
        <div className="text-center p-2 bg-amber-500/10 rounded-lg">
          <div className="text-lg font-bold text-amber-400">{stats.pattern}</div>
          <div className="text-xs text-slate-400">Pattern</div>
        </div>
      </div>

      {/* Filters */}
      <div className="px-4 py-3 border-b border-slate-700/50 space-y-2">
        <div className="flex space-x-2">
          {(['all', 'correlation', 'causality', 'pattern'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              className={classNames(
                'px-3 py-1 text-xs font-medium rounded-full transition-colors',
                filter === type
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
              )}
            >
              {type === 'all' ? 'All' : type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex items-center justify-between">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
            className="select text-xs"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="verified">Verified</option>
            <option value="rejected">Rejected</option>
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
            className="select text-xs"
          >
            <option value="confidence">Sort by Confidence</option>
            <option value="type">Sort by Type</option>
          </select>
        </div>
      </div>

      {/* Relations List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {filteredRelations.length === 0 ? (
          <div className="text-center py-8 text-slate-500">
            <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p>No relationships match the filter</p>
          </div>
        ) : (
          filteredRelations.map((relation) => {
            const typeStyle = TYPE_COLORS[relation.type];
            return (
              <div
                key={relation.id}
                className={classNames(
                  'p-3 rounded-xl border transition-all hover:shadow-lg',
                  typeStyle.bg,
                  typeStyle.border
                )}
              >
                {/* Relationship */}
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-sm font-medium text-white truncate">{relation.source}</span>
                  <svg className="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                  <span className="text-sm font-medium text-white truncate">{relation.target}</span>
                </div>

                {/* Tags */}
                <div className="flex items-center space-x-2 mb-2">
                  <span className={classNames('px-2 py-0.5 text-xs font-medium rounded-full', typeStyle.text, typeStyle.bg)}>
                    {relation.type}
                  </span>
                  <span className={classNames('px-2 py-0.5 text-xs font-medium rounded-full', STATUS_COLORS[relation.status])}>
                    {relation.status}
                  </span>
                  <span className="text-xs text-slate-400">{relation.method}</span>
                </div>

                {/* Confidence */}
                <div className="mb-3">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-slate-400">Confidence</span>
                    <span className={classNames(
                      'font-medium',
                      relation.confidence >= 0.8 ? 'text-green-400' :
                      relation.confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                    )}>
                      {(relation.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={classNames(
                        'h-full rounded-full transition-all',
                        relation.confidence >= 0.8 ? 'bg-green-500' :
                        relation.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                      )}
                      style={{ width: `${relation.confidence * 100}%` }}
                    />
                  </div>
                </div>

                {/* Lag if exists */}
                {relation.lag !== undefined && (
                  <div className="text-xs text-slate-400 mb-2">
                    Time lag: {relation.lag}s
                  </div>
                )}

                {/* Actions */}
                <div className="flex space-x-2">
                  <button
                    onClick={() => onAddToGraph(relation)}
                    className="flex-1 btn btn-primary text-xs py-1.5"
                  >
                    <svg className="w-4 h-4 mr-1 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    Add to Graph
                  </button>
                  {relation.status === 'pending' && (
                    <>
                      <button
                        onClick={() => onVerify(relation.id)}
                        className="p-1.5 rounded-lg bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-colors"
                        title="Verify"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </button>
                      <button
                        onClick={() => onReject(relation.id)}
                        className="p-1.5 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                        title="Reject"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
