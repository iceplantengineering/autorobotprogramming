// Global type definitions for the Robot Production Monitoring System

export interface RobotStatus {
  robotId: string;
  status: 'idle' | 'running' | 'error' | 'maintenance';
  position: [number, number, number, number, number, number];
  lastUpdated: string;
  uptime: number;
  currentTask?: string;
}

export interface ProductionMetrics {
  timestamp: string;
  robotId: string;
  productionOrderId: string;
  workOrderId: string;
  cycleTime: number;
  throughput: number;
  qualityScore: number;
  oee: number;
  availability: number;
  performance: number;
  qualityRate: number;
  energyConsumption: number;
  maintenanceAlerts: number;
}

export interface KPIData {
  kpiId: string;
  timestamp: string;
  value: number;
  target: number;
  performancePercentage: number;
  trendDirection: 'improving' | 'declining' | 'stable' | 'unknown';
  confidenceInterval?: [number, number];
}

export interface ProductionAlert {
  id: string;
  type: 'warning' | 'error' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: string;
  robotId?: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
}

export interface ProductionOrder {
  orderId: string;
  productId: string;
  productName: string;
  quantity: number;
  priority: number;
  dueDate: string;
  status: 'planned' | 'running' | 'paused' | 'completed' | 'cancelled';
  progress: number;
  assignedResources: string[];
  estimatedDuration: number;
  actualStartTime?: string;
  actualEndTime?: string;
}

export interface AnalyticsData {
  timestamp: string;
  summary: {
    totalKpis: number;
    calculatedKpis: number;
    categories: Record<string, {
      count: number;
      meetingTarget: number;
      criticalIssues: number;
    }>;
    overallHealth: 'unknown' | 'excellent' | 'good' | 'poor' | 'critical';
  };
  kpis: Record<string, KPIData>;
  trends: Record<string, {
    kpiId: string;
    periodStart: string;
    periodEnd: string;
    trendDirection: string;
    trendStrength: number;
    slope: number;
    rSquared: number;
    predictionNextPeriod: number;
    predictionConfidence: number;
  }>;
  anomalies: Array<{
    detectionId: string;
    kpiId: string;
    timestamp: string;
    value: number;
    expectedRange: [number, number];
    anomalyScore: number;
    severity: string;
    description: string;
  }>;
  insights: Array<{
    insightId: string;
    category: string;
    title: string;
    description: string;
    impactLevel: string;
    actionable: boolean;
    recommendations: string[];
    dataEvidence: Record<string, any>;
    createdAt: string;
  }>;
}

export interface OptimizationRecommendation {
  recommendationId: string;
  optimizationType: string;
  targetRobot: string;
  currentParameters: Record<string, any>;
  recommendedParameters: Record<string, any>;
  expectedImprovement: number;
  confidence: number;
  implementationEffort: 'low' | 'medium' | 'high';
  riskLevel: 'low' | 'medium' | 'high';
  estimatedSavings: number;
  createdAt: string;
  applied: boolean;
  appliedAt?: string;
}

export interface SystemHealth {
  overall: 'healthy' | 'warning' | 'critical';
  components: {
    productionSystem: 'healthy' | 'warning' | 'critical';
    cloudConnector: 'healthy' | 'warning' | 'critical' | 'disconnected';
    aiOptimization: 'healthy' | 'warning' | 'critical' | 'disabled';
    monitoringService: 'healthy' | 'warning' | 'critical';
  };
  uptime: number;
  lastRestart: string;
}

export interface DashboardData {
  timestamp: string;
  systemHealth: SystemHealth;
  activeRobots: number;
  totalOrders: number;
  completedOrders: number;
  averageOEE: number;
  averageQuality: number;
  alerts: ProductionAlert[];
  kpis: Record<string, KPIData>;
  recentActivity: Array<{
    type: string;
    message: string;
    timestamp: string;
    robotId?: string;
  }>;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface FilterOptions {
  dateRange?: {
    start: string;
    end: string;
  };
  robotIds?: string[];
  status?: string[];
  alertTypes?: string[];
}

export interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string;
    borderWidth?: number;
    fill?: boolean;
  }>;
}

// Component Props Types
export interface MetricCardProps {
  title: string;
  value: number | string;
  unit?: string;
  target?: number;
  trend?: {
    direction: 'up' | 'down' | 'stable';
    percentage: number;
  };
  icon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  loading?: boolean;
}

export interface AlertListProps {
  alerts: ProductionAlert[];
  maxItems?: number;
  showAcknowledgeButton?: boolean;
  onAcknowledge?: (alertId: string) => void;
}

export interface KPITrendChartProps {
  kpiData: KPIData[];
  title: string;
  height?: number;
  showTarget?: boolean;
  targetValue?: number;
}

export interface RobotStatusCardProps {
  robot: RobotStatus;
  metrics?: ProductionMetrics;
  onViewDetails?: (robotId: string) => void;
}

export interface ProductionProgressBarProps {
  value: number;
  label?: string;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  size?: 'small' | 'medium' | 'large';
  showPercentage?: boolean;
}

// Navigation and Layout Types
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: React.ReactNode;
  children?: NavigationItem[];
  badge?: number;
}

export interface SidebarProps {
  open: boolean;
  onClose: () => void;
  navigationItems: NavigationItem[];
  currentPath: string;
}

// Form and Input Types
export interface FormField {
  name: string;
  label: string;
  type: 'text' | 'number' | 'select' | 'date' | 'time' | 'textarea';
  required?: boolean;
  options?: Array<{ value: string; label: string }>;
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
  };
}

export interface OptimizationFormProps {
  robotId: string;
  currentParameters: Record<string, any>;
  onSubmit: (parameters: Record<string, any>) => void;
  loading?: boolean;
}

// Theme and Style Types
export interface CustomTheme {
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  error: string;
  warning: string;
  success: string;
  info: string;
}