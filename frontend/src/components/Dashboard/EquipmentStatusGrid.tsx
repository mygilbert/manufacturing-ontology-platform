// ============================================================
// Equipment Status Grid Component
// ============================================================
import React from 'react';
import type { Equipment } from '@/types';
import classNames from 'classnames';

interface EquipmentStatusGridProps {
  equipment: Equipment[];
  onEquipmentClick?: (equipment: Equipment) => void;
}

const STATUS_COLORS: Record<string, string> = {
  RUNNING: 'bg-emerald-500',
  IDLE: 'bg-amber-500',
  ERROR: 'bg-red-500',
  MAINTENANCE: 'bg-purple-500',
  UNKNOWN: 'bg-slate-500',
};

const STATUS_BG: Record<string, string> = {
  RUNNING: 'bg-emerald-500/10 hover:bg-emerald-500/20 border-emerald-500/30',
  IDLE: 'bg-amber-500/10 hover:bg-amber-500/20 border-amber-500/30',
  ERROR: 'bg-red-500/10 hover:bg-red-500/20 border-red-500/30',
  MAINTENANCE: 'bg-purple-500/10 hover:bg-purple-500/20 border-purple-500/30',
  UNKNOWN: 'bg-slate-500/10 hover:bg-slate-500/20 border-slate-500/30',
};

export const EquipmentStatusGrid: React.FC<EquipmentStatusGridProps> = ({
  equipment,
  onEquipmentClick,
}) => {
  // Group equipment by status
  const statusCounts = equipment.reduce((acc, eq) => {
    acc[eq.status] = (acc[eq.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="card">
      <div className="card-header flex items-center justify-between">
        <h3 className="font-medium text-white">Equipment Status</h3>
        <div className="flex items-center space-x-3">
          {Object.entries(statusCounts).map(([status, count]) => (
            <div key={status} className="flex items-center space-x-1">
              <div className={classNames('w-2 h-2 rounded-full', STATUS_COLORS[status])} />
              <span className="text-xs text-slate-400">
                {status}: {count}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="card-body">
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
          {equipment.map((eq) => (
            <button
              key={eq.equipment_id}
              onClick={() => onEquipmentClick?.(eq)}
              className={classNames(
                'p-3 rounded-lg border transition-all duration-200 text-center',
                STATUS_BG[eq.status]
              )}
              title={`${eq.name}\nStatus: ${eq.status}\nType: ${eq.equipment_type}`}
            >
              <div className={classNames('w-3 h-3 rounded-full mx-auto mb-2', STATUS_COLORS[eq.status])} />
              <div className="text-xs text-white font-medium truncate">{eq.name}</div>
              <div className="text-xs text-slate-400 truncate">{eq.equipment_type}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
