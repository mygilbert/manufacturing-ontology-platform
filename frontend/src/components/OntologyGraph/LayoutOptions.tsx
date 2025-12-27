// ============================================================
// Layout Options Component for Ontology Graph
// ============================================================
import React from 'react';
import classNames from 'classnames';

export type LayoutType = 'force' | 'hierarchical' | 'radial' | 'grid';

interface LayoutOptionsProps {
  currentLayout: LayoutType;
  onLayoutChange: (layout: LayoutType) => void;
}

const LAYOUTS: { type: LayoutType; label: string; icon: React.ReactNode }[] = [
  {
    type: 'force',
    label: 'Force',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
  },
  {
    type: 'hierarchical',
    label: 'Hierarchy',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
  },
  {
    type: 'radial',
    label: 'Radial',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12a3 3 0 116 0 3 3 0 01-6 0z" />
      </svg>
    ),
  },
  {
    type: 'grid',
    label: 'Grid',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
      </svg>
    ),
  },
];

export const LayoutOptions: React.FC<LayoutOptionsProps> = ({
  currentLayout,
  onLayoutChange,
}) => {
  return (
    <div className="absolute bottom-4 right-4 flex bg-slate-800/90 backdrop-blur-sm rounded-lg border border-slate-700 overflow-hidden">
      {LAYOUTS.map((layout) => (
        <button
          key={layout.type}
          onClick={() => onLayoutChange(layout.type)}
          className={classNames(
            'flex flex-col items-center px-3 py-2 transition-colors',
            currentLayout === layout.type
              ? 'bg-blue-500 text-white'
              : 'text-slate-400 hover:bg-slate-700 hover:text-white'
          )}
          title={layout.label}
        >
          {layout.icon}
          <span className="text-xs mt-1">{layout.label}</span>
        </button>
      ))}
    </div>
  );
};
