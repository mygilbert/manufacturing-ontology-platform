// ============================================================
// Alert Panel Component for Real-time Alerts with Acknowledge
// ============================================================
import React, { useState } from 'react';
import type { Alarm, Anomaly, AlarmStatus } from '@/types';
import { useStore } from '@/hooks';
import classNames from 'classnames';
import { format } from 'date-fns';

type TabType = 'active' | 'acknowledged';
type AlertType = 'all' | 'alarm' | 'anomaly';
type SeverityFilter = 'all' | 'CRITICAL' | 'WARNING' | 'INFO';

interface AlertPanelProps {
  maxItems?: number;
}

export const AlertPanel: React.FC<AlertPanelProps> = ({ maxItems = 50 }) => {
  const {
    activeAlarms,
    acknowledgedAlarms,
    recentAnomalies,
    acknowledgeAlarm,
    escalateAlarm
  } = useStore();

  const [tab, setTab] = useState<TabType>('active');
  const [alertType, setAlertType] = useState<AlertType>('all');
  const [severity, setSeverity] = useState<SeverityFilter>('all');
  const [acknowledgeModalId, setAcknowledgeModalId] = useState<string | null>(null);

  const currentAlarms = tab === 'active' ? activeAlarms : acknowledgedAlarms;

  const combinedAlerts = tab === 'active'
    ? [
        ...currentAlarms.map((a) => ({ ...a, type: 'alarm' as const })),
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
      })
    : currentAlarms.map((a) => ({ ...a, type: 'alarm' as const }));

  const filteredAlerts = combinedAlerts
    .filter((a) => alertType === 'all' || a.type === alertType)
    .filter((a) => severity === 'all' || a.severity === severity)
    .slice(0, maxItems);

  const severityCounts = {
    CRITICAL: activeAlarms.filter((a) => a.severity === 'CRITICAL').length,
    WARNING: activeAlarms.filter((a) => a.severity === 'WARNING').length,
    INFO: activeAlarms.filter((a) => a.severity === 'INFO').length,
  };

  const handleAcknowledge = (alarmId: string, note?: string) => {
    acknowledgeAlarm(alarmId, note);
    setAcknowledgeModalId(null);
  };

  return (
    <div className="card h-full flex flex-col">
      <div className="card-header">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <h3 className="font-medium text-white">Alerts</h3>
            <div className="flex items-center space-x-2">
              <span
                className={classNames(
                  'px-2 py-0.5 rounded-full text-xs font-medium',
                  severityCounts.CRITICAL > 0
                    ? 'bg-red-500/20 text-red-400 animate-pulse'
                    : 'bg-slate-700 text-slate-400'
                )}
              >
                {severityCounts.CRITICAL} Critical
              </span>
              <span
                className={classNames(
                  'px-2 py-0.5 rounded-full text-xs font-medium',
                  severityCounts.WARNING > 0
                    ? 'bg-amber-500/20 text-amber-400'
                    : 'bg-slate-700 text-slate-400'
                )}
              >
                {severityCounts.WARNING} Warning
              </span>
            </div>
          </div>
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

        <div className="flex border-b border-slate-700">
          <button
            className={classNames(
              'px-4 py-2 text-sm font-medium transition-colors relative',
              tab === 'active' ? 'text-blue-400' : 'text-slate-400 hover:text-slate-200'
            )}
            onClick={() => setTab('active')}
          >
            Active
            <span className="ml-2 px-1.5 py-0.5 rounded-full bg-slate-700 text-xs">
              {activeAlarms.length + recentAnomalies.length}
            </span>
            {tab === 'active' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-400" />}
          </button>
          <button
            className={classNames(
              'px-4 py-2 text-sm font-medium transition-colors relative',
              tab === 'acknowledged' ? 'text-green-400' : 'text-slate-400 hover:text-slate-200'
            )}
            onClick={() => setTab('acknowledged')}
          >
            Acknowledged
            <span className="ml-2 px-1.5 py-0.5 rounded-full bg-slate-700 text-xs">
              {acknowledgedAlarms.length}
            </span>
            {tab === 'acknowledged' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-green-400" />}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-400">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-3 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-sm">{tab === 'active' ? 'No active alerts' : 'No acknowledged alerts'}</p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-slate-700/50">
            {filteredAlerts.map((alert) => (
              <AlertItem
                key={alert.alarm_id}
                alert={alert}
                isAcknowledged={tab === 'acknowledged'}
                onAcknowledge={() => setAcknowledgeModalId(alert.alarm_id)}
                onEscalate={() => escalateAlarm(alert.alarm_id)}
              />
            ))}
          </div>
        )}
      </div>

      {acknowledgeModalId && (
        <QuickAcknowledgeModal
          onConfirm={(note) => handleAcknowledge(acknowledgeModalId, note)}
          onCancel={() => setAcknowledgeModalId(null)}
        />
      )}
    </div>
  );
};

interface AlertItemProps {
  alert: (Alarm | Anomaly) & { type: 'alarm' | 'anomaly' };
  isAcknowledged?: boolean;
  onAcknowledge?: () => void;
  onEscalate?: () => void;
}

const AlertItem: React.FC<AlertItemProps> = ({ alert, isAcknowledged, onAcknowledge, onEscalate }) => {
  const timestamp = alert.type === 'alarm' ? (alert as Alarm).occurred_at : (alert as Anomaly).detected_at;
  const formattedTime = format(new Date(timestamp), 'HH:mm:ss');
  const formattedDate = format(new Date(timestamp), 'MM/dd');
  const alarmData = alert as Alarm;
  const status = alarmData.status || 'ACTIVE';

  return (
    <div
      className={classNames(
        'px-4 py-3 hover:bg-slate-700/30 transition-colors animate-fade-in',
        alert.severity === 'CRITICAL' && 'border-l-4 border-red-500',
        alert.severity === 'WARNING' && 'border-l-4 border-amber-500',
        alert.severity === 'INFO' && 'border-l-4 border-blue-500',
        status === 'ESCALATED' && 'bg-red-900/20'
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
            <StatusBadge status={status} />
          </div>

          <p className="mt-1 text-sm text-white">
            {alert.type === 'alarm'
              ? alarmData.message || alarmData.alarm_code
              : (alert as Anomaly).message || `Anomaly detected (score: ${(alert as Anomaly).anomaly_score.toFixed(2)})`}
          </p>

          <div className="mt-1 flex items-center space-x-3 text-xs text-slate-400">
            <span>{alert.equipment_id}</span>
            <span>|</span>
            <span>{formattedDate} {formattedTime}</span>
            {alert.type === 'anomaly' && (
              <>
                <span>|</span>
                <span>Score: {(alert as Anomaly).anomaly_score.toFixed(3)}</span>
              </>
            )}
          </div>

          {isAcknowledged && alarmData.acknowledged_at && (
            <div className="mt-2 px-2 py-1 bg-green-900/20 rounded text-xs text-green-400">
              <span className="font-medium">Acknowledged:</span>{' '}
              {format(new Date(alarmData.acknowledged_at), 'MM/dd HH:mm:ss')}
              {alarmData.acknowledge_note && (
                <span className="ml-2 text-slate-400">- {alarmData.acknowledge_note}</span>
              )}
            </div>
          )}
        </div>

        {!isAcknowledged && alert.type === 'alarm' && (
          <div className="flex items-center space-x-1 ml-2">
            <button
              onClick={onAcknowledge}
              className="p-1.5 text-green-500 hover:bg-green-500/20 rounded transition-colors"
              title="Acknowledge"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </button>
            {alert.severity === 'CRITICAL' && status !== 'ESCALATED' && (
              <button
                onClick={onEscalate}
                className="p-1.5 text-amber-500 hover:bg-amber-500/20 rounded transition-colors"
                title="Escalate"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                </svg>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const StatusBadge: React.FC<{ status: AlarmStatus }> = ({ status }) => {
  const config: Record<AlarmStatus, { bg: string; text: string; label: string }> = {
    ACTIVE: { bg: 'bg-blue-500/20', text: 'text-blue-400', label: 'Active' },
    ACKNOWLEDGED: { bg: 'bg-green-500/20', text: 'text-green-400', label: 'ACK' },
    RESOLVED: { bg: 'bg-slate-500/20', text: 'text-slate-400', label: 'Resolved' },
    ESCALATED: { bg: 'bg-orange-500/20', text: 'text-orange-400', label: 'Escalated' },
  };
  const { bg, text, label } = config[status] || config.ACTIVE;
  return (
    <span className={classNames('px-1.5 py-0.5 rounded text-xs font-medium', bg, text)}>
      {label}
    </span>
  );
};

interface QuickAcknowledgeModalProps {
  onConfirm: (note?: string) => void;
  onCancel: () => void;
}

const QuickAcknowledgeModal: React.FC<QuickAcknowledgeModalProps> = ({ onConfirm, onCancel }) => {
  const [note, setNote] = useState('');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-slate-800 rounded-lg p-6 w-96 shadow-xl animate-fade-in">
        <h3 className="text-lg font-medium text-white mb-4">Acknowledge Alarm</h3>
        <textarea
          className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm resize-none"
          placeholder="Optional note..."
          rows={3}
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
        <div className="flex justify-end space-x-3 mt-4">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(note || undefined)}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors"
          >
            Acknowledge
          </button>
        </div>
      </div>
    </div>
  );
};
