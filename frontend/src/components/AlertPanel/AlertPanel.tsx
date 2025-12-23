// ============================================================
// Alert Panel Component for Real-time Alerts
// ============================================================
import React, { useState } from 'react';
import type { Alarm, Anomaly } from '@/types';
import { useStore } from '@/hooks';
import classNames from 'classnames';
import { format } from 'date-fns';

type AlertType = 'all' | 'alarm' | 'anomaly';
type SeverityFilter = 'all' | 'CRITICAL' | 'WARNING' | 'INFO';

interface AlertPanelProps {
  maxItems?: number;
}

export const AlertPanel: React.FC<AlertPanelProps> = ({ maxItems = 50 }) => {
  const { activeAlarms, recentAnomalies, removeAlarm } = useStore();
  const [alertType, setAlertType] = useState<AlertType>('all');
  const [severity, setSeverity] = useState<SeverityFilter>('all');

  // Combine and sort alerts
  const combinedAlerts = [
    ...activeAlarms.map((a) => ({ ...a, type: 'alarm' as const })),
    ...recentAnomalies.map((a) => ({
      ...a,
      type: 'anomaly' as const,
      alarm_id: a.anomaly_id,
      occurred_at: a.detected_at,
    })),
  ].sort((a, b) => {
    const dateA = new Date(a.type === 'alarm' ? a.occurred_at : a.detected_at);
    const dateB = new Date(b.type === 'alarm' ? b.occurred_at : b.detected_at);
    return dateB.getTime() - dateA.getTime();
  });

  // Apply filters
  const filteredAlerts = combinedAlerts
    .filter((a) => alertType === 'all' || a.type === alertType)
    .filter((a) => severity === 'all' || a.severity === severity)
    .slice(0, maxItems);

  const severityCounts = {
    CRITICAL: combinedAlerts.filter((a) => a.severity === 'CRITICAL').length,
    WARNING: combinedAlerts.filter((a) => a.severity === 'WARNING').length,
    INFO: combinedAlerts.filter((a) => a.severity === 'INFO').length,
  };

  return (
    <div className="card h-full flex flex-col">
      {/* Header */}
      <div className="card-header flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <h3 className="font-medium text-white">Alerts</h3>
          <div className="flex items-center space-x-2">
            <span
              className={classNames(
                'px-2 py-0.5 rounded-full text-xs font-medium',
                severityCounts.CRITICAL > 0 ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-400'
              )}
            >
              {severityCounts.CRITICAL} Critical
            </span>
            <span
              className={classNames(
                'px-2 py-0.5 rounded-full text-xs font-medium',
                severityCounts.WARNING > 0 ? 'bg-amber-500/20 text-amber-400' : 'bg-slate-700 text-slate-400'
              )}
            >
              {severityCounts.WARNING} Warning
            </span>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center space-x-2">
          <select
            value={alertType}
            onChange={(e) => setAlertType(e.target.value as AlertType)}
            className="px-2 py-1 bg-slate-700 border border-slate-600 rounded text-xs text-slate-200"
          >
            <option value="all">All Types</option>
            <option value="alarm">Alarms</option>
            <option value="anomaly">Anomalies</option>
          </select>
          <select
            value={severity}
            onChange={(e) => setSeverity(e.target.value as SeverityFilter)}
            className="px-2 py-1 bg-slate-700 border border-slate-600 rounded text-xs text-slate-200"
          >
            <option value="all">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="WARNING">Warning</option>
            <option value="INFO">Info</option>
          </select>
        </div>
      </div>

      {/* Alert List */}
      <div className="flex-1 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400">
            <div className="text-center">
              <svg
                className="w-12 h-12 mx-auto mb-3 text-slate-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-sm">No active alerts</p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-slate-700/50">
            {filteredAlerts.map((alert) => (
              <AlertItem
                key={alert.alarm_id}
                alert={alert}
                onDismiss={() => {
                  if (alert.type === 'alarm') {
                    removeAlarm(alert.alarm_id);
                  }
                }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

interface AlertItemProps {
  alert: (Alarm | Anomaly) & { type: 'alarm' | 'anomaly' };
  onDismiss?: () => void;
}

const AlertItem: React.FC<AlertItemProps> = ({ alert, onDismiss }) => {
  const timestamp = alert.type === 'alarm'
    ? (alert as Alarm).occurred_at
    : (alert as Anomaly).detected_at;

  const formattedTime = format(new Date(timestamp), 'HH:mm:ss');
  const formattedDate = format(new Date(timestamp), 'MM/dd');

  return (
    <div
      className={classNames(
        'px-4 py-3 hover:bg-slate-700/30 transition-colors animate-fade-in',
        alert.severity === 'CRITICAL' && 'border-l-4 border-red-500',
        alert.severity === 'WARNING' && 'border-l-4 border-amber-500',
        alert.severity === 'INFO' && 'border-l-4 border-blue-500'
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-2">
            <span
              className={classNames(
                'px-1.5 py-0.5 rounded text-xs font-medium uppercase',
                alert.severity === 'CRITICAL' && 'bg-red-500/20 text-red-400',
                alert.severity === 'WARNING' && 'bg-amber-500/20 text-amber-400',
                alert.severity === 'INFO' && 'bg-blue-500/20 text-blue-400'
              )}
            >
              {alert.severity}
            </span>
            <span className="text-xs text-slate-500 uppercase">{alert.type}</span>
          </div>

          <p className="mt-1 text-sm text-white">
            {alert.type === 'alarm'
              ? (alert as Alarm).message || (alert as Alarm).alarm_code
              : (alert as Anomaly).message || `Anomaly detected (score: ${(alert as Anomaly).anomaly_score.toFixed(2)})`}
          </p>

          <div className="mt-1 flex items-center space-x-3 text-xs text-slate-400">
            <span>{alert.equipment_id}</span>
            <span>|</span>
            <span>
              {formattedDate} {formattedTime}
            </span>
            {alert.type === 'anomaly' && (
              <>
                <span>|</span>
                <span>Score: {(alert as Anomaly).anomaly_score.toFixed(3)}</span>
              </>
            )}
          </div>
        </div>

        {onDismiss && alert.type === 'alarm' && (
          <button
            onClick={onDismiss}
            className="p-1 text-slate-500 hover:text-white transition-colors"
            title="Dismiss"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};
