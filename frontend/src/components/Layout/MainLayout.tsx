// ============================================================
// Main Layout Component
// ============================================================
import React, { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useStore, useWebSocket, useWebSocketMessage } from '@/hooks';
import { CriticalAlertModal } from '@/components/AlertModal';
import { alertSound } from '@/utils/alertSound';
import classNames from 'classnames';
import type { Alarm, Anomaly, Equipment } from '@/types';

export const MainLayout: React.FC = () => {
  const { 
    sidebarOpen, 
    soundEnabled,
    pendingCriticalAlarm,
    addAlarm, 
    addAnomaly, 
    updateEquipmentStatus,
    acknowledgeAlarm,
    escalateAlarm,
  } = useStore();

  // Connect to WebSocket
  useWebSocket({
    autoConnect: true,
    channels: ['alerts', 'anomalies', 'equipment_status'],
  });

  // Handle real-time alarm messages
  useWebSocketMessage<Alarm>('alert', (data: any) => {
    const alarm: Alarm = {
      alarm_id: data.alarm_id || 'alarm_' + Date.now(),
      equipment_id: data.equipment_id || 'UNKNOWN',
      alarm_code: data.alarm_code,
      severity: data.severity || 'INFO',
      message: data.message,
      occurred_at: data.timestamp || new Date().toISOString(),
    };
    
    addAlarm(alarm);
    
    // Play sound based on severity
    if (soundEnabled) {
      const severity = (alarm.severity || 'info').toLowerCase() as 'critical' | 'warning' | 'info';
      alertSound.play(severity);
    }
  });

  // Handle real-time anomaly messages
  useWebSocketMessage<Anomaly>('anomaly', (data: any) => {
    const anomaly: Anomaly = {
      anomaly_id: data.anomaly_id || 'anomaly_' + Date.now(),
      equipment_id: data.equipment_id || 'UNKNOWN',
      detected_at: data.timestamp || new Date().toISOString(),
      severity: data.severity || 'WARNING',
      anomaly_score: data.score || 0,
      message: data.message,
    };
    
    addAnomaly(anomaly);
    
    if (soundEnabled) {
      alertSound.play('warning');
    }
  });

  // Handle equipment status updates
  useWebSocketMessage<{ equipment_id: string; status: Equipment['status'] }>(
    'equipment_status',
    (data) => {
      updateEquipmentStatus(data.equipment_id, data.status);
    }
  );

  // Handle acknowledge
  const handleAcknowledge = (alarmId: string, note?: string) => {
    acknowledgeAlarm(alarmId, note);
  };

  // Handle escalate
  const handleEscalate = (alarmId: string) => {
    escalateAlarm(alarmId);
  };

  return (
    <div className="min-h-screen bg-slate-900">
      <Sidebar />
      <Header />
      <main
        className={classNames(
          'pt-16 min-h-screen transition-all duration-300',
          sidebarOpen ? 'ml-64' : 'ml-20'
        )}
      >
        <div className="p-6">
          <Outlet />
        </div>
      </main>

      {/* Critical Alarm Modal */}
      {pendingCriticalAlarm && (
        <CriticalAlertModal
          alarm={pendingCriticalAlarm}
          onAcknowledge={handleAcknowledge}
          onEscalate={handleEscalate}
        />
      )}
    </div>
  );
};
