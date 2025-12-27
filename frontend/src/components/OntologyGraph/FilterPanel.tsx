// ============================================================
// Filter Panel for Ontology Graph
// ============================================================
import React from 'react';
import classNames from 'classnames';

export interface NodeTypeFilter {
  type: string;
  label: string;
  color: string;
  visible: boolean;
  count: number;
}

export interface RelationFilter {
  type: string;
  visible: boolean;
  count: number;
}

interface FilterPanelProps {
  nodeFilters: NodeTypeFilter[];
  relationFilters: RelationFilter[];
  onNodeFilterChange: (type: string, visible: boolean) => void;
  onRelationFilterChange: (type: string, visible: boolean) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export const FilterPanel: React.FC<FilterPanelProps> = ({
  nodeFilters,
  relationFilters,
  onNodeFilterChange,
  onRelationFilterChange,
  onSelectAll,
  onDeselectAll,
}) => {
  const [expanded, setExpanded] = React.useState(true);
  const [activeTab, setActiveTab] = React.useState<'nodes' | 'relations'>('nodes');

  return (
    <div className="absolute top-4 left-4 bg-slate-800/95 backdrop-blur-sm rounded-xl border border-slate-700 shadow-xl z-20 w-64">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-slate-700 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center space-x-2">
          <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
          </svg>
          <span className="font-medium text-white">Filters</span>
        </div>
        <svg
          className={classNames('w-5 h-5 text-slate-400 transition-transform', expanded && 'rotate-180')}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>

      {expanded && (
        <>
          {/* Tabs */}
          <div className="flex border-b border-slate-700">
            <button
              className={classNames(
                'flex-1 py-2 text-sm font-medium transition-colors',
                activeTab === 'nodes'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-300'
              )}
              onClick={() => setActiveTab('nodes')}
            >
              Node Types
            </button>
            <button
              className={classNames(
                'flex-1 py-2 text-sm font-medium transition-colors',
                activeTab === 'relations'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-300'
              )}
              onClick={() => setActiveTab('relations')}
            >
              Relations
            </button>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-slate-700/50">
            <button
              onClick={onSelectAll}
              className="text-xs text-blue-400 hover:text-blue-300"
            >
              Select All
            </button>
            <button
              onClick={onDeselectAll}
              className="text-xs text-slate-400 hover:text-slate-300"
            >
              Deselect All
            </button>
          </div>

          {/* Content */}
          <div className="max-h-64 overflow-y-auto p-3 space-y-2">
            {activeTab === 'nodes' ? (
              nodeFilters.map((filter) => (
                <label
                  key={filter.type}
                  className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-700/50 cursor-pointer transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={filter.visible}
                      onChange={(e) => onNodeFilterChange(filter.type, e.target.checked)}
                      className="w-4 h-4 rounded border-slate-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-slate-800"
                    />
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: filter.color }}
                    />
                    <span className="text-sm text-slate-200">{filter.label}</span>
                  </div>
                  <span className="text-xs text-slate-500 bg-slate-700 px-2 py-0.5 rounded-full">
                    {filter.count}
                  </span>
                </label>
              ))
            ) : (
              relationFilters.map((filter) => (
                <label
                  key={filter.type}
                  className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-700/50 cursor-pointer transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={filter.visible}
                      onChange={(e) => onRelationFilterChange(filter.type, e.target.checked)}
                      className="w-4 h-4 rounded border-slate-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-slate-800"
                    />
                    <span className="text-sm text-slate-200">{filter.type}</span>
                  </div>
                  <span className="text-xs text-slate-500 bg-slate-700 px-2 py-0.5 rounded-full">
                    {filter.count}
                  </span>
                </label>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
};
