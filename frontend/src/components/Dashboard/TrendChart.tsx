// ============================================================
// Trend Chart Component using ECharts
// ============================================================
import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';

interface DataSeries {
  name: string;
  data: Array<{ timestamp: string; value: number }>;
  color?: string;
}

interface TrendChartProps {
  title: string;
  series: DataSeries[];
  height?: number;
  yAxisLabel?: string;
  showLegend?: boolean;
}

const DEFAULT_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'];

export const TrendChart: React.FC<TrendChartProps> = ({
  title,
  series,
  height = 300,
  yAxisLabel,
  showLegend = true,
}) => {
  const options: EChartsOption = useMemo(() => {
    // Get all unique timestamps
    const allTimestamps = [...new Set(series.flatMap((s) => s.data.map((d) => d.timestamp)))].sort();

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
      },
      legend: showLegend
        ? {
            bottom: 0,
            textStyle: {
              color: '#94a3b8',
              fontSize: 10,
            },
            data: series.map((s) => s.name),
          }
        : undefined,
      grid: {
        left: '10%',
        right: '5%',
        top: '15%',
        bottom: showLegend ? '15%' : '10%',
      },
      xAxis: {
        type: 'category',
        data: allTimestamps,
        axisLine: {
          lineStyle: { color: '#475569' },
        },
        axisLabel: {
          color: '#94a3b8',
          fontSize: 10,
          rotate: 30,
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
          },
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        name: yAxisLabel,
        nameTextStyle: {
          color: '#94a3b8',
          fontSize: 10,
        },
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
      series: series.map((s, index) => ({
        name: s.name,
        type: 'line',
        smooth: true,
        data: allTimestamps.map((ts) => {
          const point = s.data.find((d) => d.timestamp === ts);
          return point?.value ?? null;
        }),
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: {
          color: s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
          width: 2,
        },
        itemStyle: {
          color: s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length],
        },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              {
                offset: 0,
                color: `${s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}33`,
              },
              {
                offset: 1,
                color: `${s.color || DEFAULT_COLORS[index % DEFAULT_COLORS.length]}05`,
              },
            ],
          },
        },
      })),
    };
  }, [title, series, yAxisLabel, showLegend]);

  return (
    <div className="card">
      <ReactECharts
        option={options}
        style={{ height: `${height}px`, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  );
};
