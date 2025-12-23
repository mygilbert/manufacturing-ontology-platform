// ============================================================
// Main Layout Component
// ============================================================
import React, { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useStore, useWebSocket, useWebSocketMessage } from '@/hooks';
import classNames from 'classnames';
import type { Alarm, Anomaly, Equipment } from '@/types';

export const MainLayout: React.FC = () => {
  const { sidebarOpen, addAlarm, addAnomaly, updateEquipmentStatus } = useStore();

  // Connect to WebSocket
  useWebSocket({
    autoConnect: true,
    channels: ['alarms', 'anomalies', 'equipment_status'],
  });

  // Handle real-time alarm messages
  useWebSocketMessage<Alarm>('alarm', (alarm) => {
    addAlarm(alarm);
  });

  // Handle real-time anomaly messages
  useWebSocketMessage<Anomaly>('anomaly', (anomaly) => {
    addAnomaly(anomaly);
  });

  // Handle equipment status updates
  useWebSocketMessage<{ equipment_id: string; status: Equipment['status'] }>(
    'equipment_status',
    (data) => {
      updateEquipmentStatus(data.equipment_id, data.status);
    }
  );

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
    </div>
  );
};
