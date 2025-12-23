// ============================================================
// REST API Service
// ============================================================
import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  Equipment,
  Lot,
  Alarm,
  Anomaly,
  SPCResult,
  CapabilityResult,
  FailurePrediction,
  QualityPrediction,
  GraphData,
  ApiResponse,
  PaginatedResponse,
} from '@/types';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          localStorage.removeItem('token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // =========================================================
  // Ontology APIs
  // =========================================================

  async getEquipment(equipmentId: string): Promise<Equipment> {
    const response = await this.client.get<Equipment>(`/ontology/equipment/${equipmentId}`);
    return response.data;
  }

  async listEquipment(params?: {
    equipment_type?: string;
    status?: string;
    limit?: number;
  }): Promise<Equipment[]> {
    const response = await this.client.get<Equipment[]>('/ontology/equipment', { params });
    return response.data;
  }

  async getLot(lotId: string): Promise<Lot> {
    const response = await this.client.get<Lot>(`/ontology/lots/${lotId}`);
    return response.data;
  }

  async listLots(params?: {
    product_code?: string;
    status?: string;
    limit?: number;
  }): Promise<Lot[]> {
    const response = await this.client.get<Lot[]>('/ontology/lots', { params });
    return response.data;
  }

  async getEquipmentAlarms(
    equipmentId: string,
    params?: { severity?: string; limit?: number }
  ): Promise<Alarm[]> {
    const response = await this.client.get<Alarm[]>(
      `/ontology/equipment/${equipmentId}/alarms`,
      { params }
    );
    return response.data;
  }

  // =========================================================
  // Graph APIs
  // =========================================================

  async traverseGraph(params: {
    start_type: string;
    start_id: string;
    direction?: 'in' | 'out' | 'both';
    depth?: number;
  }): Promise<GraphData> {
    const response = await this.client.get<GraphData>('/ontology/graph/traverse', { params });
    return response.data;
  }

  async findPath(params: {
    from_type: string;
    from_id: string;
    to_type: string;
    to_id: string;
    max_depth?: number;
  }): Promise<GraphData & { found: boolean; length: number }> {
    const response = await this.client.get('/ontology/graph/path', { params });
    return response.data;
  }

  // =========================================================
  // Analytics APIs
  // =========================================================

  async getAnomalies(params?: {
    equipment_id?: string;
    severity?: string;
    limit?: number;
  }): Promise<Anomaly[]> {
    const response = await this.client.get<Anomaly[]>('/analytics/anomalies', { params });
    return response.data;
  }

  async analyzeSPC(params: {
    equipment_id: string;
    item_id: string;
    chart_type?: string;
  }): Promise<SPCResult> {
    const response = await this.client.get<SPCResult>('/analytics/spc', { params });
    return response.data;
  }

  async analyzeCapability(params: {
    equipment_id: string;
    item_id: string;
    usl: number;
    lsl: number;
    target?: number;
  }): Promise<CapabilityResult> {
    const response = await this.client.get<CapabilityResult>('/analytics/capability', { params });
    return response.data;
  }

  async predictFailure(params: {
    equipment_id: string;
    horizon_hours?: number;
  }): Promise<FailurePrediction> {
    const response = await this.client.get<FailurePrediction>('/analytics/predict/failure', {
      params,
    });
    return response.data;
  }

  async predictQuality(params: {
    process_id: string;
    lot_id?: string;
  }): Promise<QualityPrediction> {
    const response = await this.client.get<QualityPrediction>('/analytics/predict/quality', {
      params,
    });
    return response.data;
  }

  // =========================================================
  // Model Training APIs
  // =========================================================

  async trainAnomalyModel(params: {
    equipment_id: string;
    model_type: 'isolation_forest' | 'autoencoder';
    feature_names: string[];
    lookback_days?: number;
  }): Promise<ApiResponse<{ model_id: string }>> {
    const response = await this.client.post('/analytics/models/train', params);
    return response.data;
  }
}

export const api = new ApiService();
