export interface SystemMetrics {
  // AI/ML Metrics
  accuracy: number;
  throughput: number;
  latency: number;
  gpuUtilization: number;
  memoryUsage: number;
  activeModels: number;
  
  // System Performance
  uptime: number;
  cpuUsage: number;
  diskUsage: number;
  networkTraffic: number;
  
  // Business Metrics
  activeUsers: number;
  globalReach: number;
  dataProcessed: number;
  totalProcessed: number;
  successRate: number;
  
  // Executive Metrics (CEO)
  revenue: number;
  growth: number;
  customers: number;
  marketShare: number;
  satisfaction: number;
  employees: number;
  
  // Financial Metrics (CFO)
  profit: number;
  cashFlow: number;
  expenses: number;
  roi: number;
  burnRate: number;
  
  // Marketing Metrics (CMO)
  cac: number; // Customer Acquisition Cost
  ltv: number; // Lifetime Value
  conversion: number;
  engagement: number;
  reach: number;
  brandAwareness: number;
  
  // Technology Metrics (CTO)
  deployments: number;
  codeQuality: number;
  security: number;
  performance: number;
  infrastructure: number;
  
  // Operations Metrics (COO)
  efficiency: number;
  quality: number;
  onTime: number;
  utilization: number;
  incidents: number;
}

export interface SystemComponent {
  name: string;
  status: 'online' | 'warning' | 'offline';
  uptime: string;
  load: number;
}

export interface PipelineStage {
  id: string;
  name: string;
  status: 'processing' | 'completed' | 'waiting' | 'error';
  progress: number;
  throughput: number;
}

export interface BudgetItem {
  category: string;
  allocated: number;
  spent: number;
  variance: number;
}

export interface Investment {
  investment: string;
  value: number;
  return: number;
}

export interface Campaign {
  name: string;
  budget: number;
  spent: number;
  performance: number;
  status: string;
}

export interface Channel {
  channel: string;
  traffic: number;
  conversion: number;
  roi: number;
}

export interface BrandMetric {
  metric: string;
  value: number;
  trend: 'up' | 'down';
}

export interface ProductionLine {
  line: string;
  efficiency: number;
  output: number;
  status: string;
}

export interface Supplier {
  supplier: string;
  delivery: number;
  quality: number;
  cost: string;
}

export interface PerformanceMetric {
  metric: string;
  value: number;
  target: number;
}

export interface TechService {
  service: string;
  status: string;
  load: number;
}

export interface SecurityCheck {
  check: string;
  status: string;
  lastRun: string;
}

export interface CodeStat {
  metric: string;
  value: string;
  change: string;
}

export interface TeamProductivity {
  metric: string;
  value: string;
  change: string;
}

export interface StrategicInitiative {
  name: string;
  progress: number;
  status: string;
}

export interface RegionalData {
  region: string;
  revenue: number;
  growth: number;
}

export interface KPI {
  metric: string;
  value: string;
  trend: 'up' | 'down';
}

class MetricsService {
  private baseUrl: string;
  private cache: Map<string, { data: any; timestamp: number }> = new Map();
  private cacheTimeout = 30000; // 30 seconds

  constructor(baseUrl: string = 'http://147.93.102.165:8000') {
    this.baseUrl = baseUrl;
  }

  private async fetchWithCache<T>(endpoint: string): Promise<T> {
    const cacheKey = endpoint;
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.cache.set(cacheKey, { data, timestamp: Date.now() });
      return data;
    } catch (error) {
      console.error(`Failed to fetch ${endpoint}:`, error);
      // Return cached data if available, otherwise throw
      if (cached) {
        return cached.data;
      }
      throw error;
    }
  }

  // Main System Metrics
  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.fetchWithCache<SystemMetrics>('/metrics/system');
  }

  // AI/ML Specific Metrics
  async getAIMetrics(): Promise<Partial<SystemMetrics>> {
    return this.fetchWithCache<Partial<SystemMetrics>>('/metrics/ai');
  }

  // System Components Status
  async getSystemComponents(): Promise<SystemComponent[]> {
    return this.fetchWithCache<SystemComponent[]>('/metrics/components');
  }

  // Processing Pipeline Status
  async getPipelineStatus(): Promise<PipelineStage[]> {
    return this.fetchWithCache<PipelineStage[]>('/metrics/pipeline');
  }

  // Executive Dashboard Metrics (CEO)
  async getExecutiveMetrics(): Promise<{
    metrics: Partial<SystemMetrics>;
    initiatives: StrategicInitiative[];
    regions: RegionalData[];
    kpis: KPI[];
  }> {
    return this.fetchWithCache('/metrics/executive');
  }

  // Financial Metrics (CFO)
  async getFinancialMetrics(): Promise<{
    metrics: Partial<SystemMetrics>;
    budget: BudgetItem[];
    investments: Investment[];
  }> {
    return this.fetchWithCache('/metrics/financial');
  }

  // Marketing Metrics (CMO)
  async getMarketingMetrics(): Promise<{
    metrics: Partial<SystemMetrics>;
    campaigns: Campaign[];
    channels: Channel[];
    brandMetrics: BrandMetric[];
  }> {
    return this.fetchWithCache('/metrics/marketing');
  }

  // Technology Metrics (CTO)
  async getTechnologyMetrics(): Promise<{
    metrics: Partial<SystemMetrics>;
    codeStats: CodeStat[];
    teamProductivity: TeamProductivity[];
    infrastructure: TechService[];
    security: SecurityCheck[];
  }> {
    return this.fetchWithCache('/metrics/technology');
  }

  // Operations Metrics (COO)
  async getOperationsMetrics(): Promise<{
    metrics: Partial<SystemMetrics>;
    productionLines: ProductionLine[];
    suppliers: Supplier[];
    performanceMetrics: PerformanceMetric[];
  }> {
    return this.fetchWithCache('/metrics/operations');
  }

  // Real-time Updates
  async subscribeToMetrics(callback: (metrics: SystemMetrics) => void): Promise<() => void> {
    const interval = setInterval(async () => {
      try {
        const metrics = await this.getSystemMetrics();
        callback(metrics);
      } catch (error) {
        console.error('Failed to fetch real-time metrics:', error);
      }
    }, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }

  // Health Check
  async healthCheck(): Promise<{ status: 'healthy' | 'unhealthy'; timestamp: number }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        timeout: 5000,
      } as RequestInit);
      
      return {
        status: response.ok ? 'healthy' : 'unhealthy',
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        timestamp: Date.now()
      };
    }
  }

  // Clear cache
  clearCache(): void {
    this.cache.clear();
  }

  // Update base URL
  updateBaseUrl(newUrl: string): void {
    this.baseUrl = newUrl;
    this.clearCache();
  }
}

// Create singleton instance
export const metricsService = new MetricsService();

// Export the class for custom instances
export default MetricsService;