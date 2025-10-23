import axios, { AxiosInstance, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';

import { ApiResponse } from '@/types';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error);

        // Handle different error types
        if (error.response) {
          // Server responded with error status
          const { status, data } = error.response;
          const message = data?.message || data?.error || 'Server error occurred';

          switch (status) {
            case 401:
              toast.error('Authentication required');
              break;
            case 403:
              toast.error('Access denied');
              break;
            case 404:
              toast.error('Resource not found');
              break;
            case 500:
              toast.error('Server error occurred');
              break;
            default:
              toast.error(message);
          }
        } else if (error.request) {
          // Request was made but no response received
          toast.error('Network error - please check connection');
        } else {
          // Something else happened
          toast.error('An unexpected error occurred');
        }

        return Promise.reject(error);
      }
    );
  }

  // Generic GET request
  async get<T>(endpoint: string, params?: any): Promise<T> {
    const response = await this.client.get(endpoint, { params });
    return response.data;
  }

  // Generic POST request
  async post<T>(endpoint: string, data?: any): Promise<T> {
    const response = await this.client.post(endpoint, data);
    return response.data;
  }

  // Generic PUT request
  async put<T>(endpoint: string, data?: any): Promise<T> {
    const response = await this.client.put(endpoint, data);
    return response.data;
  }

  // Generic DELETE request
  async delete<T>(endpoint: string): Promise<T> {
    const response = await this.client.delete(endpoint);
    return response.data;
  }

  // Production System API
  async getProductionDashboard() {
    return this.get('/production/dashboard');
  }

  async getProductionSummary(params?: { startDate?: string; endDate?: string }) {
    return this.get('/production/summary', params);
  }

  async getProductionOrders(params?: { status?: string; limit?: number }) {
    return this.get('/production/orders', params);
  }

  async createProductionOrder(data: any) {
    return this.post('/production/orders', data);
  }

  async updateProductionOrder(orderId: string, data: any) {
    return this.put(`/production/orders/${orderId}`, data);
  }

  // Robot Management API
  async getRobotStatuses() {
    return this.get('/robots/status');
  }

  async getRobotDetails(robotId: string) {
    return this.get(`/robots/${robotId}`);
  }

  async updateRobotParameters(robotId: string, parameters: any) {
    return this.put(`/robots/${robotId}/parameters`, parameters);
  }

  async executeRobotCommand(robotId: string, command: any) {
    return this.post(`/robots/${robotId}/commands`, command);
  }

  // Metrics and Analytics API
  async getCurrentMetrics(params?: { robotIds?: string[] }) {
    return this.get('/metrics/current', params);
  }

  async getMetricsHistory(params: {
    kpiId?: string;
    robotId?: string;
    startDate?: string;
    endDate?: string;
    interval?: string;
  }) {
    return this.get('/metrics/history', params);
  }

  async getAnalyticsDashboard() {
    return this.get('/analytics/dashboard');
  }

  async getKpiTrends(kpiId: string, periodDays?: number) {
    return this.get(`/analytics/trends/${kpiId}`, { periodDays });
  }

  async getAnomalyDetection(params?: { robotId?: string; severity?: string }) {
    return this.get('/analytics/anomalies', params);
  }

  // Alerts API
  async getActiveAlerts(params?: { type?: string; acknowledged?: boolean }) {
    return this.get('/alerts/active', params);
  }

  async acknowledgeAlert(alertId: string) {
    return this.put(`/alerts/${alertId}/acknowledge`);
  }

  async createAlert(alert: any) {
    return this.post('/alerts', alert);
  }

  // AI Optimization API
  async getOptimizationRecommendations(params?: { robotId?: string; applied?: boolean }) {
    return this.get('/optimization/recommendations', params);
  }

  async applyOptimizationRecommendation(recommendationId: string) {
    return this.put(`/optimization/recommendations/${recommendationId}/apply`);
  }

  async predictQuality(robotId: string, parameters: any) {
    return this.post('/optimization/predict/quality', { robotId, parameters });
  }

  async predictMaintenance(robotId: string, operationalData: any) {
    return this.post('/optimization/predict/maintenance', { robotId, operationalData });
  }

  // Cloud Integration API
  async getCloudStatus() {
    return this.get('/cloud/status');
  }

  async syncToCloud(dataType?: string) {
    return this.post('/cloud/sync', { dataType });
  }

  async getCloudBackups(params?: { limit?: number }) {
    return this.get('/cloud/backups', params);
  }

  // Settings API
  async getSystemSettings() {
    return this.get('/settings/system');
  }

  async updateSystemSettings(settings: any) {
    return this.put('/settings/system', settings);
  }

  async getUserPreferences() {
    return this.get('/settings/user');
  }

  async updateUserPreferences(preferences: any) {
    return this.put('/settings/user', preferences);
  }

  // Health Check API
  async healthCheck() {
    return this.get('/health');
  }

  async getServiceStatus() {
    return this.get('/status/services');
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;

// Export types for API responses
export interface DashboardResponse {
  timestamp: string;
  systemHealth: {
    overall: 'healthy' | 'warning' | 'critical';
    components: any;
  };
  activeRobots: number;
  totalOrders: number;
  completedOrders: number;
  averageOEE: number;
  averageQuality: number;
  alerts: any[];
  kpis: any;
  recentActivity: any[];
}

export interface MetricsResponse {
  timestamp: string;
  metrics: Record<string, any>;
  robotStatus: Record<string, string>;
}

export interface AnalyticsResponse {
  timestamp: string;
  summary: any;
  kpis: Record<string, any>;
  trends: Record<string, any>;
  anomalies: any[];
  insights: any[];
}

export interface OptimizationResponse {
  timestamp: string;
  status: any;
  recentRecommendations: any[];
  optimizationHistory: any[];
  serviceRunning: boolean;
}