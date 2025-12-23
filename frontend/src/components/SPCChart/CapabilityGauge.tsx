// ============================================================
// Process Capability Gauge Component
// ============================================================
import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { CapabilityResult } from '@/types';
import classNames from 'classnames';

interface CapabilityGaugeProps {
  result: CapabilityResult;
  title?: string;
}

const LEVEL_COLORS: Record<string, { color: string; bg: string }> = {
  EXCELLENT: { color: '#10b981', bg: 'bg-emerald-500/20' },
  GOOD: { color: '#22c55e', bg: 'bg-green-500/20' },
  ACCEPTABLE: { color: '#f59e0b', bg: 'bg-amber-500/20' },
  POOR: { color: '#f97316', bg: 'bg-orange-500/20' },
  CRITICAL: { color: '#ef4444', bg: 'bg-red-500/20' },
};

export const CapabilityGauge: React.FC<CapabilityGaugeProps> = ({ result, title }) => {
  const levelStyle = LEVEL_COLORS[result.level] || LEVEL_COLORS.CRITICAL;

  const gaugeOptions: EChartsOption = useMemo(() => {
    // Cpk value capped at 2 for display purposes
    const displayValue = Math.min(result.cpk, 2);

    return {
      backgroundColor: 'transparent',
      series: [
        {
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          min: 0,
          max: 2,
          splitNumber: 4,
          radius: '100%',
          center: ['50%', '75%'],
          axisLine: {
            lineStyle: {
              width: 20,
              color: [
                [0.33, '#ef4444'],  // 0-0.67: Critical/Poor
                [0.5, '#f97316'],   // 0.67-1.0: Poor
                [0.67, '#f59e0b'],  // 1.0-1.33: Acceptable
                [0.83, '#22c55e'],  // 1.33-1.67: Good
                [1, '#10b981'],     // 1.67-2.0: Excellent
              ],
            },
          },
          pointer: {
            icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
            length: '60%',
            width: 10,
            offsetCenter: [0, '-10%'],
            itemStyle: {
              color: '#e2e8f0',
            },
          },
          axisTick: {
            length: 8,
            lineStyle: {
              color: 'auto',
              width: 1,
            },
          },
          splitLine: {
            length: 15,
            lineStyle: {
              color: 'auto',
              width: 2,
            },
          },
          axisLabel: {
            color: '#94a3b8',
            fontSize: 10,
            distance: -30,
            formatter: (value: number) => {
              if (value === 0) return '0';
              if (value === 0.67) return '0.67';
              if (value === 1) return '1.0';
              if (value === 1.33) return '1.33';
              if (value === 2) return '2.0';
              return '';
            },
          },
          title: {
            offsetCenter: [0, '20%'],
            fontSize: 12,
            color: '#94a3b8',
          },
          detail: {
            fontSize: 28,
            offsetCenter: [0, '-5%'],
            valueAnimation: true,
            formatter: (value: number) => value.toFixed(2),
            color: levelStyle.color,
          },
          data: [{ value: displayValue, name: 'Cpk' }],
        },
      ],
    };
  }, [result.cpk, levelStyle.color]);

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between">
        <h3 className="font-medium text-white">{title || 'Process Capability'}</h3>
        <span
          className={classNames(
            'px-2 py-1 rounded text-xs font-medium',
            levelStyle.bg
          )}
          style={{ color: levelStyle.color }}
        >
          {result.level}
        </span>
      </div>
      <div className="card-body">
        <ReactECharts
          option={gaugeOptions}
          style={{ height: '200px', width: '100%' }}
          opts={{ renderer: 'canvas' }}
        />

        {/* Statistics Grid */}
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-2xl font-bold text-white">{result.cp.toFixed(2)}</div>
            <div className="text-xs text-slate-400 mt-1">Cp</div>
          </div>
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-2xl font-bold" style={{ color: levelStyle.color }}>
              {result.cpk.toFixed(2)}
            </div>
            <div className="text-xs text-slate-400 mt-1">Cpk</div>
          </div>
          {result.pp !== undefined && (
            <div className="text-center p-3 bg-slate-700/30 rounded-lg">
              <div className="text-2xl font-bold text-white">{result.pp.toFixed(2)}</div>
              <div className="text-xs text-slate-400 mt-1">Pp</div>
            </div>
          )}
          {result.ppk !== undefined && (
            <div className="text-center p-3 bg-slate-700/30 rounded-lg">
              <div className="text-2xl font-bold text-white">{result.ppk.toFixed(2)}</div>
              <div className="text-xs text-slate-400 mt-1">Ppk</div>
            </div>
          )}
        </div>

        {/* PPM */}
        <div className="mt-4 p-3 bg-slate-700/30 rounded-lg text-center">
          <div className="text-lg font-bold text-white">
            {result.ppm_total.toFixed(1)} <span className="text-sm font-normal text-slate-400">PPM</span>
          </div>
          <div className="text-xs text-slate-400 mt-1">Expected Defects per Million</div>
        </div>
      </div>
    </div>
  );
};
