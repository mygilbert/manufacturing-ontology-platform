// ============================================================
// Zustand Store for Global State Management
// ============================================================
import { create } from 'zustand';
import type { Equipment, Lot, Alarm, Anomaly } from '@/types';

interface AppState {
  // Equipment
  selectedEquipment: Equipment | null;
  equipmentList: Equipment[];
  setSelectedEquipment: (equipment: Equipment | null) => void;
  setEquipmentList: (list: Equipment[]) => void;
  updateEquipmentStatus: (equipmentId: string, status: Equipment['status']) => void;

  // Lots
  selectedLot: Lot | null;
  lotList: Lot[];
  setSelectedLot: (lot: Lot | null) => void;
  setLotList: (list: Lot[]) => void;

  // Alarms
  activeAlarms: Alarm[];
  acknowledgedAlarms: Alarm[];
  pendingCriticalAlarm: Alarm | null;
  addAlarm: (alarm: Alarm) => void;
  removeAlarm: (alarmId: string) => void;
  acknowledgeAlarm: (alarmId: string, note?: string) => void;
  escalateAlarm: (alarmId: string) => void;
  dismissCriticalModal: () => void;
  clearAlarms: () => void;

  // Anomalies
  recentAnomalies: Anomaly[];
  addAnomaly: (anomaly: Anomaly) => void;

  // UI State
  sidebarOpen: boolean;
  soundEnabled: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setSoundEnabled: (enabled: boolean) => void;

  // Notifications
  unreadCount: number;
  incrementUnread: () => void;
  clearUnread: () => void;
}

export const useStore = create<AppState>((set, get) => ({
  // Equipment
  selectedEquipment: null,
  equipmentList: [],
  setSelectedEquipment: (equipment) => set({ selectedEquipment: equipment }),
  setEquipmentList: (list) => set({ equipmentList: list }),
  updateEquipmentStatus: (equipmentId, status) =>
    set((state) => ({
      equipmentList: state.equipmentList.map((eq) =>
        eq.equipment_id === equipmentId ? { ...eq, status } : eq
      ),
      selectedEquipment:
        state.selectedEquipment?.equipment_id === equipmentId
          ? { ...state.selectedEquipment, status }
          : state.selectedEquipment,
    })),

  // Lots
  selectedLot: null,
  lotList: [],
  setSelectedLot: (lot) => set({ selectedLot: lot }),
  setLotList: (list) => set({ lotList: list }),

  // Alarms
  activeAlarms: [],
  acknowledgedAlarms: [],
  pendingCriticalAlarm: null,

  addAlarm: (alarm) =>
    set((state) => {
      const newAlarm = { ...alarm, status: 'ACTIVE' as const };
      const isCritical = alarm.severity === 'CRITICAL';
      
      return {
        activeAlarms: [newAlarm, ...state.activeAlarms].slice(0, 100),
        unreadCount: state.unreadCount + 1,
        // Critical 알람이면 모달 표시
        pendingCriticalAlarm: isCritical && !state.pendingCriticalAlarm 
          ? newAlarm 
          : state.pendingCriticalAlarm,
      };
    }),

  removeAlarm: (alarmId) =>
    set((state) => ({
      activeAlarms: state.activeAlarms.filter((a) => a.alarm_id !== alarmId),
    })),

  acknowledgeAlarm: (alarmId, note) =>
    set((state) => {
      const alarm = state.activeAlarms.find((a) => a.alarm_id === alarmId);
      if (!alarm) return state;

      const acknowledgedAlarm: Alarm = {
        ...alarm,
        status: 'ACKNOWLEDGED',
        acknowledged_at: new Date().toISOString(),
        acknowledged_by: 'current_user', // TODO: 실제 사용자 정보로 대체
        acknowledge_note: note,
      };

      return {
        activeAlarms: state.activeAlarms.filter((a) => a.alarm_id !== alarmId),
        acknowledgedAlarms: [acknowledgedAlarm, ...state.acknowledgedAlarms].slice(0, 100),
        pendingCriticalAlarm: 
          state.pendingCriticalAlarm?.alarm_id === alarmId 
            ? null 
            : state.pendingCriticalAlarm,
      };
    }),

  escalateAlarm: (alarmId) =>
    set((state) => {
      const alarm = state.activeAlarms.find((a) => a.alarm_id === alarmId);
      if (!alarm) return state;

      const escalatedAlarm: Alarm = {
        ...alarm,
        status: 'ESCALATED',
        escalated_at: new Date().toISOString(),
        escalated_to: 'supervisor', // TODO: 실제 에스컬레이션 대상
      };

      return {
        activeAlarms: state.activeAlarms.map((a) =>
          a.alarm_id === alarmId ? escalatedAlarm : a
        ),
        pendingCriticalAlarm:
          state.pendingCriticalAlarm?.alarm_id === alarmId
            ? null
            : state.pendingCriticalAlarm,
      };
    }),

  dismissCriticalModal: () => set({ pendingCriticalAlarm: null }),

  clearAlarms: () => set({ activeAlarms: [], acknowledgedAlarms: [] }),

  // Anomalies
  recentAnomalies: [],
  addAnomaly: (anomaly) =>
    set((state) => ({
      recentAnomalies: [anomaly, ...state.recentAnomalies].slice(0, 50),
      unreadCount: state.unreadCount + 1,
    })),

  // UI State
  sidebarOpen: true,
  soundEnabled: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  setSoundEnabled: (enabled) => set({ soundEnabled: enabled }),

  // Notifications
  unreadCount: 0,
  incrementUnread: () => set((state) => ({ unreadCount: state.unreadCount + 1 })),
  clearUnread: () => set({ unreadCount: 0 }),
}));
