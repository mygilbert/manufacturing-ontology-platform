// ============================================================
// Manufacturing Ontology Platform - Type Definitions
// ============================================================

// =========================================================
// Equipment Types
// =========================================================
export type EquipmentStatus = 'RUNNING' | 'IDLE' | 'ERROR' | 'MAINTENANCE' | 'UNKNOWN';
export type EquipmentType = 'DRY_ETCH' | 'WET_ETCH' | 'CVD' | 'PVD' | 'LITHO' | 'CMP' | 'METROLOGY' | 'OTHER';

export interface Equipment {
  equipment_id: string;
  name: string;
  equipment_type: EquipmentType;
  status: EquipmentStatus;
  location?: string;
  properties?: Record<string, unknown>;
}

// =========================================================
// Lot & Wafer Types
// =========================================================
export type LotStatus = 'CREATED' | 'IN_PROCESS' | 'HOLD' | 'COMPLETED' | 'SCRAPPED';
export type WaferStatus = 'IN_PROCESS' | 'COMPLETED' | 'SCRAPPED';

export interface Lot {
  lot_id: string;
  product_code: string;
  quantity: number;
  status: LotStatus;
  current_step?: string;
  start_time?: string;
  end_time?: string;
  wafer_count: number;
}

export interface Wafer {
  wafer_id: string;
  lot_id: string;
  slot_no: number;
  status: WaferStatus;
}

// =========================================================
// Process & Recipe Types
// =========================================================
export interface Process {
  process_id: string;
  name: string;
  sequence: number;
  description?: string;
}

export interface Recipe {
  recipe_id: string;
  name: string;
  version: string;
  parameters?: Record<string, unknown>;
}

// =========================================================
// Alarm Types
// =========================================================
export type AlarmSeverity = 'CRITICAL' | 'WARNING' | 'INFO';
export type AlarmStatus = 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED' | 'ESCALATED';

export interface Alarm {
  alarm_id: string;
  equipment_id: string;
  alarm_code?: string;
  severity: AlarmSeverity;
  message?: string;
  occurred_at: string;
  resolved_at?: string;
  // Acknowledge workflow
  status?: AlarmStatus;
  acknowledged_at?: string;
  acknowledged_by?: string;
  acknowledge_note?: string;
  // Escalation
  escalated_at?: string;
  escalated_to?: string;
}

// =========================================================
// Measurement Types
// =========================================================
export interface Measurement {
  measurement_id: string;
  equipment_id: string;
  param_id: string;
  value: number;
  timestamp: string;
  status: string;
  unit?: string;
}

// =========================================================
// Anomaly Detection Types
// =========================================================
export interface Anomaly {
  anomaly_id: string;
  equipment_id: string;
  detected_at: string;
  severity: AlarmSeverity;
  anomaly_score: number;
  features?: Record<string, number>;
  message?: string;
}

// =========================================================
// SPC Types
// =========================================================
export type ChartType = 'individual' | 'xbar' | 'range' | 'xbar_r';

export interface SPCResult {
  equipment_id: string;
  item_id: string;
  chart_type: ChartType;
  ucl: number;
  cl: number;
  lcl: number;
  cpk?: number;
  ooc_count: number;
  status: 'NORMAL' | 'WARNING' | 'OOC';
}

export interface SPCDataPoint {
  timestamp: string;
  value: number;
  status: 'normal' | 'warning' | 'ooc';
  violations?: number[];
}

// =========================================================
// Capability Types
// =========================================================
export interface CapabilityResult {
  equipment_id: string;
  item_id: string;
  cp: number;
  cpk: number;
  pp?: number;
  ppk?: number;
  ppm_total: number;
  level: 'EXCELLENT' | 'GOOD' | 'ACCEPTABLE' | 'POOR' | 'CRITICAL';
}

// =========================================================
// Prediction Types
// =========================================================
export interface FailurePrediction {
  equipment_id: string;
  predicted_at: string;
  probability: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  remaining_useful_life?: number;
  contributing_factors?: Array<{ factor: string; contribution: number }>;
}

export interface QualityPrediction {
  process_id: string;
  lot_id?: string;
  predicted_at: string;
  predicted_yield: number;
  confidence_interval?: [number, number];
}

// =========================================================
// Graph Types (for Ontology Visualization)
// =========================================================
export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, unknown>;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  label: string;
  properties?: Record<string, unknown>;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// =========================================================
// Real-time Types
// =========================================================
export type WebSocketMessageType =
  | 'alarm'
  | 'measurement'
  | 'equipment_status'
  | 'anomaly'
  | 'spc_violation';

export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  timestamp: string;
  data: T;
}

// =========================================================
// API Response Types
// =========================================================
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
}
