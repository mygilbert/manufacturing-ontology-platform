// ============================================================
// Dashboard Page
// ============================================================
import React, { useEffect, useState } from 'react';
import { StatCard, EquipmentStatusGrid, TrendChart } from '@/components/Dashboard';
import { AlertPanel } from '@/components/AlertPanel';
import { api } from '@/services/api';
import { useStore } from '@/hooks';
import type { Equipment } from '@/types';

export const DashboardPage: React.FC = () => {
  const { equipmentList, setEquipmentList } = useStore();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalEquipment: 0,
    runningEquipment: 0,
    activeAlarms: 0,
    averageCpk: 0,
  });

  // Mock trend data for demonstration
  const [trendData] = useState(() => {
    const now = Date.now();
    const data = [];
    for (let i = 30; i >= 0; i--) {
      data.push({
        timestamp: new Date(now - i * 60000).toISOString(),
        value: 100 + Math.random() * 20 - 10,
      });
    }
    return data;
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const equipment = await api.listEquipment({ limit: 100 });
        setEquipmentList(equipment);

        // Calculate stats
        const running = equipment.filter((e) => e.status === 'RUNNING').length;
        setStats({
          totalEquipment: equipment.length,
          runningEquipment: running,
          activeAlarms: Math.floor(Math.random() * 10),
          averageCpk: 1.2 + Math.random() * 0.5,
        });
      } catch (error) {
        console.error('Failed to fetch equipment:', error);
        // Use mock data on error
        const mockEquipment: Equipment[] = [
          { equipment_id: 'EQP001', name: 'Etcher-01', equipment_type: 'DRY_ETCH', status: 'RUNNING' },
          { equipment_id: 'EQP002', name: 'Etcher-02', equipment_type: 'DRY_ETCH', status: 'IDLE' },
          { equipment_id: 'EQP003', name: 'CVD-01', equipment_type: 'CVD', status: 'RUNNING' },
          { equipment_id: 'EQP004', name: 'CVD-02', equipment_type: 'CVD', status: 'MAINTENANCE' },
          { equipment_id: 'EQP005', name: 'Litho-01', equipment_type: 'LITHO', status: 'RUNNING' },
          { equipment_id: 'EQP006', name: 'CMP-01', equipment_type: 'CMP', status: 'ERROR' },
        ];
        setEquipmentList(mockEquipment);
        setStats({
          totalEquipment: mockEquipment.length,
          runningEquipment: mockEquipment.filter((e) => e.status === 'RUNNING').length,
          activeAlarms: 3,
          averageCpk: 1.45,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [setEquipmentList]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
        <div className="loader" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400 text-sm mt-1">Manufacturing Overview</p>
        </div>
        <div className="flex items-center space-x-3">
          <span className="text-sm text-slate-400">Last updated: Just now</span>
          <button className="btn btn-primary text-sm">
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Equipment"
          value={stats.totalEquipment}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          }
          color="blue"
        />
        <StatCard
          title="Running"
          value={stats.runningEquipment}
          change={5.2}
          changeLabel="vs yesterday"
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          color="green"
        />
        <StatCard
          title="Active Alarms"
          value={stats.activeAlarms}
          change={-12.5}
          changeLabel="vs yesterday"
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          }
          color="amber"
        />
        <StatCard
          title="Avg Cpk"
          value={stats.averageCpk.toFixed(2)}
          change={2.3}
          changeLabel="vs last week"
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          color="purple"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Equipment Status */}
        <div className="lg:col-span-2">
          <EquipmentStatusGrid equipment={equipmentList} />
        </div>

        {/* Alert Panel */}
        <div className="lg:col-span-1 h-[400px]">
          <AlertPanel maxItems={20} />
        </div>
      </div>

      {/* Trend Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TrendChart
          title="Production Yield Trend"
          series={[
            { name: 'Yield', data: trendData, color: '#10b981' },
          ]}
          yAxisLabel="%"
        />
        <TrendChart
          title="Equipment Utilization"
          series={[
            {
              name: 'Utilization',
              data: trendData.map((d) => ({ ...d, value: 70 + Math.random() * 20 })),
              color: '#3b82f6',
            },
          ]}
          yAxisLabel="%"
        />
      </div>
    </div>
  );
};
