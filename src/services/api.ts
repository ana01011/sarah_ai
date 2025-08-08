/**
 * API Service for communicating with Amesie AI Backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ChatRequest {
  prompt: string;
  max_length?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stream?: boolean;
}

export interface ChatResponse {
  text: string;
  prompt: string;
  inference_time: number;
  model_name: string;
  parameters: Record<string, any>;
}

export interface MetricsResponse {
  system: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    gpu_utilization: number;
    gpu_memory_usage: number;
  };
  ai: {
    accuracy: number;
    throughput: number;
    latency: number;
    gpu_utilization: number;
    memory_usage: number;
    active_models: number;
    requests_per_second: number;
    error_rate: number;
    avg_response_time: number;
  };
  performance: {
    total_requests: number;
    total_errors: number;
    total_inferences: number;
    error_rate: number;
    avg_response_time: number;
    requests_per_second: number;
  };
  health: {
    status: string;
    version: string;
    uptime: number;
    performance: Record<string, any>;
    memory_usage: Record<string, any>;
    active_connections: number;
  };
}

export interface NeuralNetworkMetrics {
  layers: Array<{
    name: string;
    neurons: number;
    activation: string;
  }>;
  total_parameters: string;
  model_size: string;
  quantization: string;
  memory_efficiency: string;
}

export interface ProcessingPipelineMetrics {
  stages: Array<{
    name: string;
    status: string;
    efficiency: number;
    throughput: number;
    latency: number;
  }>;
  overall_efficiency: number;
  total_throughput: number;
  total_latency: number;
}

// Agent chat API
export interface AgentChatRequest {
  role?: string;
  message: string;
}

export interface AgentChatResponse {
  role: string;
  response: string;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Chat API
  async generateCompletion(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/v1/chat/completion', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async generateConversation(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/v1/chat/conversation', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Metrics API
  async getPerformanceMetrics(): Promise<MetricsResponse> {
    return this.request<MetricsResponse>('/api/v1/metrics/performance');
  }

  async getSystemMetrics(): Promise<{
    system: Record<string, any>;
    ai: Record<string, any>;
  }> {
    return this.request('/api/v1/metrics/system');
  }

  async getUsageMetrics(): Promise<{
    total_requests: number;
    total_errors: number;
    error_rate: number;
    endpoints: Record<string, any>;
    time_period: string;
  }> {
    return this.request('/api/v1/metrics/usage');
  }

  async getNeuralNetworkMetrics(): Promise<NeuralNetworkMetrics> {
    return this.request<NeuralNetworkMetrics>('/api/v1/metrics/neural-network');
  }

  async getProcessingPipelineMetrics(): Promise<ProcessingPipelineMetrics> {
    return this.request<ProcessingPipelineMetrics>('/api/v1/metrics/processing-pipeline');
  }

  async getHealthStatus(): Promise<{
    status: string;
    version: string;
    uptime: number;
    performance: Record<string, any>;
    memory_usage: Record<string, any>;
    active_connections: number;
  }> {
    return this.request('/api/v1/metrics/health');
  }

  async getDashboardData(): Promise<{
    performance: MetricsResponse;
    neural_network: NeuralNetworkMetrics;
    processing_pipeline: ProcessingPipelineMetrics;
    timestamp: number;
  }> {
    return this.request('/api/v1/metrics/dashboard');
  }

  async getRealtimeMetrics(): Promise<{
    system: Record<string, any>;
    ai: Record<string, any>;
    activity: {
      requests_last_minute: number;
      errors_last_minute: number;
      requests_per_second: number;
    };
    timestamp: number;
  }> {
    return this.request('/api/v1/metrics/realtime');
  }

  // Model API
  async getModels(): Promise<{
    models: Array<{
      name: string;
      status: string;
      device: string;
      quantization: string;
      parameters: Record<string, any>;
    }>;
    active_model: string | null;
  }> {
    return this.request('/api/v1/chat/models');
  }

  async loadModel(): Promise<{
    message: string;
    model_name: string;
  }> {
    return this.request('/api/v1/chat/models/load', {
      method: 'POST',
    });
  }

  async unloadModel(): Promise<{
    message: string;
  }> {
    return this.request('/api/v1/chat/models/unload', {
      method: 'POST',
    });
  }

  // WebSocket connections
  createChatWebSocket(): WebSocket {
    return new WebSocket(`ws://localhost:8000/ws/chat`);
  }

  createMetricsWebSocket(): WebSocket {
    return new WebSocket(`ws://localhost:8000/ws/metrics`);
  }

  createSystemWebSocket(): WebSocket {
    return new WebSocket(`ws://localhost:8000/ws/system`);
  }

  // Health check
  async checkHealth(): Promise<{
    status: string;
    version: string;
    uptime: number;
    performance: Record<string, any>;
    memory_usage: Record<string, any>;
    active_connections: number;
  }> {
    return this.request('/health');
  }

  // API info
  async getApiInfo(): Promise<{
    name: string;
    version: string;
    model: {
      name: string;
      is_loaded: boolean;
      device: string;
      quantization: string;
    };
    endpoints: {
      chat: string;
      metrics: string;
      websockets: Record<string, string>;
    };
  }> {
    return this.request('/api/v1/info');
  }

  async sendAgentChat(request: AgentChatRequest): Promise<AgentChatResponse> {
    return this.request<AgentChatResponse>('/api/v1/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getAgentRoles(): Promise<string[]> {
    // In production, fetch from backend; for now, hardcode
    return ['CEO', 'CFO', 'CTO'];
  }
}

export const apiService = new ApiService();