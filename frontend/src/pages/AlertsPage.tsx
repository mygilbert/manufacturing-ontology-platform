// ============================================================
// Alerts Page
// ============================================================
import React from 'react';
import { AlertPanel } from '@/components/AlertPanel';
import { useStore } from '@/hooks';

export const AlertsPage: React.FC = () => {
  const { activeAlarms, recentAnomalies, clearAlarms } = useStore();

  const stats = {
    total: activeAlarms.length + recentAnomalies.length,
    critical: activeAlarms.filter((a) => a.severity === 'CRITICAL').length +
              recentAnomalies.filter((a) => a.severity === 'CRITICAL').length,
    warning: activeAlarms.filter((a) => a.severity === 'WARNING').length +
             recentAnomalies.filter((a) => a.severity === 'WARNING').length,
    info: activeAlarms.filter((a) => a.severity === 'INFO').length +
          recentAnomalies.filter((a) => a.severity === 'INFO').length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Alerts & Anomalies</h1>
          <p className="text-slate-400 text-sm mt-1">Real-time monitoring alerts</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={clearAlarms}
            className="btn btn-secondary"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear All
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400">Total Alerts</p>
            <p className="text-3xl font-bold text-white mt-1">{stats.total}</p>
          </div>
          <div className="p-3 bg-blue-500/20 rounded-lg">
            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          </div>
        </div>
        <div className="card p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400">Critical</p>
            <p className="text-3xl font-bold text-red-400 mt-1">{stats.critical}</p>
          </div>
          <div className="p-3 bg-red-500/20 rounded-lg">
            <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
        </div>
        <div className="card p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400">Warnings</p>
            <p className="text-3xl font-bold text-amber-400 mt-1">{stats.warning}</p>
          </div>
          <div className="p-3 bg-amber-500/20 rounded-lg">
            <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        </div>
        <div className="card p-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-400">Info</p>
            <p className="text-3xl font-bold text-blue-400 mt-1">{stats.info}</p>
          </div>
          <div className="p-3 bg-blue-500/20 rounded-lg">
            <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        </div>
      </div>

      {/* Alert Panel - Full Width */}
      <div className="h-[calc(100vh-20rem)]">
        <AlertPanel maxItems={100} />
      </div>
    </div>
  );
};
