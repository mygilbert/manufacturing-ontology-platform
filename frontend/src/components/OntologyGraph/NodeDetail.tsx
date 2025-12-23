// ============================================================
// Node Detail Panel Component
// ============================================================
import React from 'react';
import type { GraphNode } from '@/types';
import classNames from 'classnames';

interface NodeDetailProps {
  node: GraphNode | null;
  onClose: () => void;
  onExpand?: (node: GraphNode) => void;
}

const NODE_TYPE_LABELS: Record<string, string> = {
  Equipment: '설비',
  Lot: '로트',
  Wafer: '웨이퍼',
  Process: '공정',
  Recipe: '레시피',
  Measurement: '측정값',
  Alarm: '알람',
  Parameter: '파라미터',
};

export const NodeDetail: React.FC<NodeDetailProps> = ({ node, onClose, onExpand }) => {
  if (!node) return null;

  const properties = node.properties || {};

  return (
    <div className="fixed right-6 top-24 w-80 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl overflow-hidden animate-slide-up z-50">
      {/* Header */}
      <div className="px-4 py-3 bg-slate-700/50 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div
            className={classNames(
              'w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold',
              node.type === 'Equipment' && 'bg-blue-600',
              node.type === 'Lot' && 'bg-emerald-600',
              node.type === 'Wafer' && 'bg-indigo-600',
              node.type === 'Process' && 'bg-amber-600',
              node.type === 'Recipe' && 'bg-purple-600',
              node.type === 'Measurement' && 'bg-cyan-600',
              node.type === 'Alarm' && 'bg-red-600',
              !['Equipment', 'Lot', 'Wafer', 'Process', 'Recipe', 'Measurement', 'Alarm'].includes(node.type) && 'bg-slate-600'
            )}
          >
            {node.type.charAt(0)}
          </div>
          <div>
            <h3 className="font-semibold text-white">{node.label || node.id}</h3>
            <p className="text-xs text-slate-400">{NODE_TYPE_LABELS[node.type] || node.type}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 text-slate-400 hover:text-white transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4 max-h-[60vh] overflow-y-auto">
        {/* ID */}
        <div>
          <label className="text-xs font-medium text-slate-400 uppercase tracking-wide">ID</label>
          <p className="mt-1 text-sm text-slate-200 font-mono bg-slate-700/50 px-2 py-1 rounded">
            {node.id}
          </p>
        </div>

        {/* Properties */}
        {Object.keys(properties).length > 0 && (
          <div>
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wide">
              Properties
            </label>
            <div className="mt-2 space-y-2">
              {Object.entries(properties).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="text-slate-400">{key}</span>
                  <span className="text-slate-200 font-medium truncate max-w-[60%]">
                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Status for Equipment */}
        {node.type === 'Equipment' && properties.status && (
          <div>
            <label className="text-xs font-medium text-slate-400 uppercase tracking-wide">Status</label>
            <div className="mt-2">
              <span
                className={classNames(
                  'status-badge',
                  properties.status === 'RUNNING' && 'status-running',
                  properties.status === 'IDLE' && 'status-idle',
                  properties.status === 'ERROR' && 'status-error',
                  properties.status === 'MAINTENANCE' && 'status-maintenance',
                  !['RUNNING', 'IDLE', 'ERROR', 'MAINTENANCE'].includes(String(properties.status)) && 'status-unknown'
                )}
              >
                {String(properties.status)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-4 py-3 bg-slate-700/30 border-t border-slate-700 flex space-x-2">
        {onExpand && (
          <button
            onClick={() => onExpand(node)}
            className="flex-1 btn btn-primary text-sm"
          >
            <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
            Expand
          </button>
        )}
        <button className="flex-1 btn btn-secondary text-sm">
          <svg className="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
          Details
        </button>
      </div>
    </div>
  );
};
