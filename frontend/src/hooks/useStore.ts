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
  addAlarm: (alarm: Alarm) => void;
  removeAlarm: (alarmId: string) => void;
  clearAlarms: () => void;

  // Anomalies
  recentAnomalies: Anomaly[];
  addAnomaly: (anomaly: Anomaly) => void;

  // UI State
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Notifications
  unreadCount: number;
  incrementUnread: () => void;
  clearUnread: () => void;
}

export const useStore = create<AppState>((set) => ({
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
  addAlarm: (alarm) =>
    set((state) => ({
      activeAlarms: [alarm, ...state.activeAlarms].slice(0, 100),
      unreadCount: state.unreadCount + 1,
    })),
  removeAlarm: (alarmId) =>
    set((state) => ({
      activeAlarms: state.activeAlarms.filter((a) => a.alarm_id !== alarmId),
    })),
  clearAlarms: () => set({ activeAlarms: [] }),

  // Anomalies
  recentAnomalies: [],
  addAnomaly: (anomaly) =>
    set((state) => ({
      recentAnomalies: [anomaly, ...state.recentAnomalies].slice(0, 50),
      unreadCount: state.unreadCount + 1,
    })),

  // UI State
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  // Notifications
  unreadCount: 0,
  incrementUnread: () => set((state) => ({ unreadCount: state.unreadCount + 1 })),
  clearUnread: () => set({ unreadCount: 0 }),
}));
