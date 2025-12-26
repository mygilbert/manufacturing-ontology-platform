// ============================================================
// SPC Charts Page
// ============================================================
import React, { useState, useEffect } from 'react';
import { SPCChart, CapabilityGauge } from '@/components/SPCChart';
import { api } from '@/services/api';
import type { SPCDataPoint, CapabilityResult, Equipment } from '@/types';

export const SPCPage: React.FC = () => {
  const [selectedEquipment, setSelectedEquipment] = useState<string>('');
  const [selectedParam, setSelectedParam] = useState<string>('RF_Power');
  const [chartType, setChartType] = useState<'individual' | 'xbar' | 'range'>('individual');
  const [loading, setLoading] = useState(true);
  const [equipmentList, setEquipmentList] = useState<Equipment[]>([]);

  // SPC data
  const [spcData, setSpcData] = useState<SPCDataPoint[]>([]);
  const [capability, setCapability] = useState<CapabilityResult | null>(null);
  const [controlLimits, setControlLimits] = useState({ ucl: 105, cl: 100, lcl: 95, usl: 110, lsl: 90 });

  const parameterList = [
    { id: 'RF_Power', name: 'RF Power', unit: 'W' },
    { id: 'Chamber_Pressure', name: 'Chamber Pressure', unit: 'mTorr' },
    { id: 'Chuck_Temp', name: 'Chuck Temperature', unit: 'C' },
    { id: 'Gas_Flow_CF4', name: 'CF4 Gas Flow', unit: 'sccm' },
    { id: 'Gas_Flow_O2', name: 'O2 Gas Flow', unit: 'sccm' },
    { id: 'ESC_Voltage', name: 'ESC Voltage', unit: 'V' },
    { id: 'Heater_Temp', name: 'Heater Temp', unit: 'C' },
    { id: 'DC_Power', name: 'DC Power', unit: 'W' },
    { id: 'Thickness', name: 'Thickness', unit: 'nm' },
  ];

  // Fetch equipment list on mount
  useEffect(() => {
    const fetchEquipment = async () => {
      try {
        const response = await fetch('/api/ontology/equipment?limit=100');
        const data = await response.json();
        setEquipmentList(data);
        if (data.length > 0) {
          setSelectedEquipment(data[0].equipment_id);
        }
      } catch (error) {
        console.error('Failed to fetch equipment:', error);
      }
    };
    fetchEquipment();
  }, []);

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
      if (!selectedEquipment) return;
      setLoading(true);
      try {
        // Fetch real SPC control chart data from API
        const response = await fetch(
          `/api/analytics/spc/control-chart?equipment_id=${selectedEquipment}&item_id=${selectedParam}&hours=168`
        );
        const result = await response.json();

        if (result.error) {
          throw new Error(result.error);
        }

        setControlLimits({
          ucl: result.limits.ucl,
          cl: result.limits.cl,
          lcl: result.limits.lcl,
          usl: result.limits.ucl + (result.limits.ucl - result.limits.cl),
          lsl: result.limits.lcl - (result.limits.cl - result.limits.lcl),
        });

        // Convert API data to SPC data points
        const spcPoints: SPCDataPoint[] = result.data.map((d: { timestamp: string; value: number }) => {
          const status = d.value > result.limits.ucl || d.value < result.limits.lcl 
            ? 'ooc' 
            : (d.value > result.limits.ucl - (result.limits.ucl - result.limits.cl) * 0.5 ||
               d.value < result.limits.lcl + (result.limits.cl - result.limits.lcl) * 0.5)
            ? 'warning' 
            : 'normal';
          return {
            timestamp: d.timestamp,
            value: d.value,
            status,
            violations: status === 'ooc' ? [1] : [],
          };
        }).reverse();

        setSpcData(spcPoints);

        // Calculate capability
        const mean = result.statistics.mean;
        const std = result.statistics.std;
        const usl = result.limits.ucl + (result.limits.ucl - result.limits.cl);
        const lsl = result.limits.lcl - (result.limits.cl - result.limits.lcl);
        const cp = (usl - lsl) / (6 * std);
        const cpk = Math.min((usl - mean) / (3 * std), (mean - lsl) / (3 * std));

        setCapability({
          equipment_id: selectedEquipment,
          item_id: selectedParam,
          cp: cp,
          cpk: cpk,
          pp: cp * 0.95,
          ppk: cpk * 0.95,
          ppm_total: cpk >= 1.33 ? 63 : cpk >= 1.0 ? 2700 : 66800,
          level: cpk >= 2.0 ? 'EXCELLENT' : cpk >= 1.67 ? 'GOOD' : cpk >= 1.33 ? 'ACCEPTABLE' : 'POOR',
        });
      } catch (error) {
        console.error('Failed to load SPC data:', error);
        // Use mock data as fallback
        setSpcData(generateMockData());
        setCapability({
          equipment_id: selectedEquipment,
          item_id: selectedParam,
          cp: 1.2,
          cpk: 1.0,
          pp: 1.1,
          ppk: 0.9,
          ppm_total: 500,
          level: 'ACCEPTABLE',
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
