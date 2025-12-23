// ============================================================
// Statistics Card Component
// ============================================================
import React from 'react';
import classNames from 'classnames';

interface StatCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon?: React.ReactNode;
  color?: 'blue' | 'green' | 'amber' | 'red' | 'purple';
}

const COLOR_CLASSES = {
  blue: {
    bg: 'bg-blue-500/20',
    icon: 'text-blue-400',
    ring: 'ring-blue-500/50',
  },
  green: {
    bg: 'bg-emerald-500/20',
    icon: 'text-emerald-400',
    ring: 'ring-emerald-500/50',
  },
  amber: {
    bg: 'bg-amber-500/20',
    icon: 'text-amber-400',
    ring: 'ring-amber-500/50',
  },
  red: {
    bg: 'bg-red-500/20',
    icon: 'text-red-400',
    ring: 'ring-red-500/50',
  },
  purple: {
    bg: 'bg-purple-500/20',
    icon: 'text-purple-400',
    ring: 'ring-purple-500/50',
  },
};

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  change,
  changeLabel,
  icon,
  color = 'blue',
}) => {
  const colorClasses = COLOR_CLASSES[color];

  return (
    <div className="card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400">{title}</p>
          <p className="text-3xl font-bold text-white mt-2">{value}</p>
          {change !== undefined && (
            <div className="flex items-center mt-2">
              <span
                className={classNames(
                  'text-sm font-medium',
                  change >= 0 ? 'text-emerald-400' : 'text-red-400'
                )}
              >
                {change >= 0 ? '+' : ''}
                {change}%
              </span>
              {changeLabel && (
                <span className="text-xs text-slate-500 ml-2">{changeLabel}</span>
              )}
            </div>
          )}
        </div>
        {icon && (
          <div
            className={classNames(
              'p-3 rounded-lg ring-1',
              colorClasses.bg,
              colorClasses.ring
            )}
          >
            <div className={colorClasses.icon}>{icon}</div>
          </div>
        )}
      </div>
    </div>
  );
};
