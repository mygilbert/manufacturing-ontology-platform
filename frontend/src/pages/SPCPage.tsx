// ============================================================
// SPC Charts Page
// ============================================================
import React, { useState, useEffect } from 'react';
import { SPCChart, CapabilityGauge } from '@/components/SPCChart';
import { api } from '@/services/api';
import type { SPCDataPoint, CapabilityResult, Equipment } from '@/types';

export const SPCPage: React.FC = () => {
  const [selectedEquipment, setSelectedEquipment] = useState<string>('EQP001');
  const [selectedParam, setSelectedParam] = useState<string>('temperature');
  const [chartType, setChartType] = useState<'individual' | 'xbar' | 'range'>('individual');
  const [loading, setLoading] = useState(true);

  // Mock data for demonstration
  const [spcData, setSpcData] = useState<SPCDataPoint[]>([]);
  const [capability, setCapability] = useState<CapabilityResult | null>(null);
  const [controlLimits, setControlLimits] = useState({ ucl: 105, cl: 100, lcl: 95, usl: 110, lsl: 90 });

  // Mock equipment list
  const equipmentList: Equipment[] = [
    { equipment_id: 'EQP001', name: 'Etcher-01', equipment_type: 'DRY_ETCH', status: 'RUNNING' },
    { equipment_id: 'EQP002', name: 'CVD-01', equipment_type: 'CVD', status: 'RUNNING' },
    { equipment_id: 'EQP003', name: 'Litho-01', equipment_type: 'LITHO', status: 'RUNNING' },
  ];

  const parameterList = [
    { id: 'temperature', name: 'Temperature', unit: 'C' },
    { id: 'pressure', name: 'Pressure', unit: 'mTorr' },
    { id: 'power', name: 'RF Power', unit: 'W' },
    { id: 'flow_rate', name: 'Gas Flow', unit: 'sccm' },
  ];

  useEffect(() => {
    const generateMockData = () => {
      const now = Date.now();
      const data: SPCDataPoint[] = [];

      for (let i = 50; i >= 0; i--) {
        const baseValue = 100;
        let value = baseValue + (Math.random() - 0.5) * 8;
        let status: 'normal' | 'warning' | 'ooc' = 'normal';
        const violations: number[] = [];

        // Add some out-of-control points
        if (Math.random() < 0.1) {
          value = baseValue + (Math.random() > 0.5 ? 1 : -1) * (6 + Math.random() * 3);
          status = 'ooc';
          violations.push(1); // Rule 1: Point beyond 3 sigma
        } else if (Math.random() < 0.15) {
          value = baseValue + (Math.random() > 0.5 ? 1 : -1) * (4 + Math.random() * 2);
          status = 'warning';
        }

        data.push({
          timestamp: new Date(now - i * 60000).toISOString(),
          value,
          status,
          violations,
        });
      }

      return data;
    };

    const loadData = async () => {
      setLoading(true);
      try {
        // Try API first
        const result = await api.analyzeSPC({
          equipment_id: selectedEquipment,
          item_id: selectedParam,
          chart_type: chartType,
        });

        setControlLimits({
          ucl: result.ucl,
          cl: result.cl,
          lcl: result.lcl,
          usl: result.ucl + 5,
          lsl: result.lcl - 5,
        });

        // Generate mock data since we don't have real measurement data
        setSpcData(generateMockData());

        const capResult = await api.analyzeCapability({
          equipment_id: selectedEquipment,
          item_id: selectedParam,
          usl: result.ucl + 5,
          lsl: result.lcl - 5,
        });
        setCapability(capResult);
      } catch (error) {
        // Use mock data
        setSpcData(generateMockData());
        setCapability({
          equipment_id: selectedEquipment,
          item_id: selectedParam,
          cp: 1.2 + Math.random() * 0.5,
          cpk: 1.0 + Math.random() * 0.5,
          pp: 1.1 + Math.random() * 0.4,
          ppk: 0.9 + Math.random() * 0.4,
          ppm_total: Math.random() * 1000,
          level: ['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR'][Math.floor(Math.random() * 4)] as CapabilityResult['level'],
        });
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedEquipment, selectedParam, chartType]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">SPC Control Charts</h1>
          <p className="text-slate-400 text-sm mt-1">Statistical Process Control Analysis</p>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Equipment</label>
            <select
              value={selectedEquipment}
              onChange={(e) => setSelectedEquipment(e.target.value)}
              className="select"
            >
              {equipmentList.map((eq) => (
                <option key={eq.equipment_id} value={eq.equipment_id}>
                  {eq.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Parameter</label>
            <select
              value={selectedParam}
              onChange={(e) => setSelectedParam(e.target.value)}
              className="select"
            >
              {parameterList.map((param) => (
                <option key={param.id} value={param.id}>
                  {param.name} ({param.unit})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Chart Type</label>
            <select
              value={chartType}
              onChange={(e) => setChartType(e.target.value as typeof chartType)}
              className="select"
            >
              <option value="individual">Individual (I-MR)</option>
              <option value="xbar">X-bar</option>
              <option value="range">Range (R)</option>
            </select>
          </div>
          <div className="flex items-end">
            <button className="btn btn-primary w-full">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh Data
            </button>
          </div>
        </div>
      </div>

      {/* Charts */}
      {loading ? (
        <div className="flex items-center justify-center h-96">
          <div className="loader" />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main SPC Chart */}
          <div className="lg:col-span-2">
            <SPCChart
              title={`${parameterList.find((p) => p.id === selectedParam)?.name || selectedParam} Control Chart`}
              data={spcData}
              ucl={controlLimits.ucl}
              cl={controlLimits.cl}
              lcl={controlLimits.lcl}
              usl={controlLimits.usl}
              lsl={controlLimits.lsl}
              chartType={chartType}
              height={400}
            />
          </div>

          {/* Capability Gauge */}
          <div>
            {capability && (
              <CapabilityGauge
                result={capability}
                title="Process Capability"
              />
            )}
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <div className="card p-4">
        <h3 className="font-medium text-white mb-4">Analysis Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-xl font-bold text-white">{spcData.length}</div>
            <div className="text-xs text-slate-400 mt-1">Data Points</div>
          </div>
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-xl font-bold text-red-400">
              {spcData.filter((d) => d.status === 'ooc').length}
            </div>
            <div className="text-xs text-slate-400 mt-1">Out of Control</div>
          </div>
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-xl font-bold text-amber-400">
              {spcData.filter((d) => d.status === 'warning').length}
            </div>
            <div className="text-xs text-slate-400 mt-1">Warnings</div>
          </div>
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-xl font-bold text-white">
              {(spcData.reduce((sum, d) => sum + d.value, 0) / spcData.length).toFixed(2)}
            </div>
            <div className="text-xs text-slate-400 mt-1">Mean</div>
          </div>
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-xl font-bold text-white">{controlLimits.ucl.toFixed(2)}</div>
            <div className="text-xs text-slate-400 mt-1">UCL</div>
          </div>
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-xl font-bold text-white">{controlLimits.lcl.toFixed(2)}</div>
            <div className="text-xs text-slate-400 mt-1">LCL</div>
          </div>
        </div>
      </div>
    </div>
  );
};
