// ============================================================
// SPC Control Chart Component using ECharts
// ============================================================
import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { SPCDataPoint } from '@/types';

interface SPCChartProps {
  title: string;
  data: SPCDataPoint[];
  ucl: number;
  cl: number;
  lcl: number;
  usl?: number;
  lsl?: number;
  chartType?: 'individual' | 'xbar' | 'range';
  height?: number;
}

export const SPCChart: React.FC<SPCChartProps> = ({
  title,
  data,
  ucl,
  cl,
  lcl,
  usl,
  lsl,
  chartType = 'individual',
  height = 350,
}) => {
  const options: EChartsOption = useMemo(() => {
    const timestamps = data.map((d) => d.timestamp);
    const values = data.map((d) => d.value);
    const statuses = data.map((d) => d.status);

    // Create series data with colors based on status
    const seriesData = values.map((value, index) => ({
      value,
      itemStyle: {
        color:
          statuses[index] === 'ooc'
            ? '#ef4444'
            : statuses[index] === 'warning'
            ? '#f59e0b'
            : '#10b981',
      },
    }));

    return {
      backgroundColor: 'transparent',
      title: {
        text: title,
        left: 'center',
        textStyle: {
          color: '#e2e8f0',
          fontSize: 14,
          fontWeight: 500,
        },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1e293b',
        borderColor: '#475569',
        textStyle: {
          color: '#e2e8f0',
        },
        formatter: (params: any) => {
          const param = params[0];
          const dataPoint = data[param.dataIndex];
          let html = `<div class="font-medium">${param.axisValue}</div>`;
          html += `<div class="text-sm">Value: ${param.value.toFixed(3)}</div>`;
          if (dataPoint.violations && dataPoint.violations.length > 0) {
            html += `<div class="text-xs text-red-400">Rules: ${dataPoint.violations.join(', ')}</div>`;
          }
          return html;
        },
      },
      grid: {
        left: '10%',
        right: '5%',
        top: '15%',
        bottom: '15%',
      },
      xAxis: {
        type: 'category',
        data: timestamps,
        axisLine: {
          lineStyle: { color: '#475569' },
        },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 10,
          rotate: 45,
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
          },
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLine: {
          lineStyle: { color: '#475569' },
        },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 10,
        },
        splitLine: {
          lineStyle: { color: '#334155', type: 'dashed' },
        },
      },
      series: [
        // Main data series
        {
          name: 'Value',
          type: 'line',
          data: seriesData,
          symbol: 'circle',
          symbolSize: 8,
          lineStyle: {
            color: '#3b82f6',
            width: 2,
          },
          emphasis: {
            scale: 1.5,
          },
        },
        // UCL line
        {
          name: 'UCL',
          type: 'line',
          data: new Array(data.length).fill(ucl),
          symbol: 'none',
          lineStyle: {
            color: '#ef4444',
            width: 2,
            type: 'dashed',
          },
          markLine: {
            silent: true,
            symbol: 'none',
            label: {
              show: true,
              position: 'end',
              formatter: 'UCL',
              color: '#ef4444',
            },
          },
        },
        // CL line
        {
          name: 'CL',
          type: 'line',
          data: new Array(data.length).fill(cl),
          symbol: 'none',
          lineStyle: {
            color: '#10b981',
            width: 2,
          },
        },
        // LCL line
        {
          name: 'LCL',
          type: 'line',
          data: new Array(data.length).fill(lcl),
          symbol: 'none',
          lineStyle: {
            color: '#ef4444',
            width: 2,
            type: 'dashed',
          },
        },
        // Optional USL line
        ...(usl !== undefined
          ? [
              {
                name: 'USL',
                type: 'line' as const,
                data: new Array(data.length).fill(usl),
                symbol: 'none' as const,
                lineStyle: {
                  color: '#8b5cf6',
                  width: 1,
                  type: 'dotted' as const,
                },
              },
            ]
          : []),
        // Optional LSL line
        ...(lsl !== undefined
          ? [
              {
                name: 'LSL',
                type: 'line' as const,
                data: new Array(data.length).fill(lsl),
                symbol: 'none' as const,
                lineStyle: {
                  color: '#8b5cf6',
                  width: 1,
                  type: 'dotted' as const,
                },
              },
            ]
          : []),
      ],
      legend: {
        show: true,
        bottom: 0,
        textStyle: {
          color: '#94a3b8',
          fontSize: 10,
        },
        data: ['Value', 'UCL', 'CL', 'LCL', ...(usl ? ['USL'] : []), ...(lsl ? ['LSL'] : [])],
      },
    };
  }, [title, data, ucl, cl, lcl, usl, lsl]);

  return (
    <div className="card">
      <ReactECharts
        option={options}
        style={{ height: `${height}px`, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
      {/* Status Summary */}
      <div className="px-4 py-2 border-t border-slate-700 flex items-center justify-between text-sm">
        <div className="flex items-center space-x-4">
          <span className="text-slate-400">
            Points: <span className="text-white font-medium">{data.length}</span>
          </span>
          <span className="text-slate-400">
            OOC:{' '}
            <span className="text-red-400 font-medium">
              {data.filter((d) => d.status === 'ooc').length}
            </span>
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-500">
            UCL: {ucl.toFixed(2)} | CL: {cl.toFixed(2)} | LCL: {lcl.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
};
