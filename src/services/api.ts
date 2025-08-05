/**
 * API Service for Neural Network Backend Communication
 */

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

// Types
export interface ChatMessage {
  content: string;
  role: 'user' | 'assistant';
  timestamp?: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  max_length?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  stream?: boolean;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  request_id: string;
  processing_time: number;
  tokens_generated: number;
  model_info: any;
  timestamp: string;
}

export interface ConversationHistory {
  conversation_id: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export interface SystemHealth {
  status: string;
  timestamp: string;
  uptime: number;
  model_loaded: boolean;
  device: string;
  health_check_time?: number;
}

export interface PerformanceMetrics {
  request_count: number;
  avg_processing_time: number;
  requests_per_second: number;
  tokens_per_second: number;
  error_rate: number;
  uptime: number;
}

export interface SystemResources {
  cpu_usage_percent: number;
  memory_usage_percent: number;
  memory_available_gb: number;
  disk_usage_percent: number;
  gpu_memory_allocated_gb?: number;
  gpu_memory_reserved_gb?: number;
  gpu_utilization_percent?: number;
}

export interface CacheMetrics {
  cache_hits: number;
  cache_misses: number;
  hit_rate: number;
  total_requests: number;
}

export interface ModelInfo {
  model_name: string;
  vocab_size: number;
  d_model: number;
  num_layers: number;
  max_seq_len: number;
  total_parameters: number;
  device: string;
}

export interface ComprehensiveMetrics {
  performance: PerformanceMetrics;
  resources: SystemResources;
  cache: CacheMetrics;
  model_info?: ModelInfo;
  timestamp: string;
}

export interface Alert {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  metric_value: number;
  threshold: number;
  timestamp: string;
  resolved: boolean;
}

// API Client Class
class NeuralNetworkAPI {
  private baseURL: string;
  private headers: HeadersInit;

  constructor() {
    this.baseURL = `${API_BASE_URL}${API_VERSION}`;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const config: RequestInit = {
      headers: this.headers,
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      return await response.json();
    } catch (error) {
      console.error(`API Request failed: ${url}`, error);
      throw error;
    }
  }

  // Chat API Methods
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>('/chat/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getConversations(limit = 10, offset = 0): Promise<ConversationHistory[]> {
    return this.request<ConversationHistory[]>(
      `/chat/conversations?limit=${limit}&offset=${offset}`
    );
  }

  async getConversation(conversationId: string): Promise<ConversationHistory> {
    return this.request<ConversationHistory>(`/chat/conversations/${conversationId}`);
  }

  async deleteConversation(conversationId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/chat/conversations/${conversationId}`, {
      method: 'DELETE',
    });
  }

  async clearConversation(conversationId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/chat/conversations/${conversationId}/clear`, {
      method: 'POST',
    });
  }

  // WebSocket Chat Stream
  createChatStream(
    onMessage: (data: any) => void,
    onError: (error: Event) => void,
    onClose: () => void
  ): WebSocket {
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}${API_VERSION}/chat/chat/stream`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = onError;
    ws.onclose = onClose;

    return ws;
  }

  // Monitoring API Methods
  async getHealth(): Promise<SystemHealth> {
    return this.request<SystemHealth>('/monitoring/health');
  }

  async getMetrics(): Promise<ComprehensiveMetrics> {
    return this.request<ComprehensiveMetrics>('/monitoring/metrics');
  }

  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    return this.request<PerformanceMetrics>('/monitoring/metrics/performance');
  }

  async getResourceMetrics(): Promise<SystemResources> {
    return this.request<SystemResources>('/monitoring/metrics/resources');
  }

  async getCacheMetrics(): Promise<CacheMetrics> {
    return this.request<CacheMetrics>('/monitoring/metrics/cache');
  }

  async getModelInfo(): Promise<ModelInfo | null> {
    return this.request<ModelInfo | null>('/monitoring/model/info');
  }

  async getAlerts(activeOnly = true, severity?: string, limit = 50): Promise<Alert[]> {
    const params = new URLSearchParams({
      active_only: activeOnly.toString(),
      limit: limit.toString(),
    });
    
    if (severity) {
      params.append('severity', severity);
    }

    return this.request<Alert[]>(`/monitoring/alerts?${params.toString()}`);
  }

  async resolveAlert(alertId: string): Promise<{ message: string }> {
    return this.request<{ message: string }>(`/monitoring/alerts/${alertId}/resolve`, {
      method: 'POST',
    });
  }

  async clearCache(): Promise<{ message: string; keys_deleted: number }> {
    return this.request<{ message: string; keys_deleted: number }>('/monitoring/cache/clear', {
      method: 'POST',
    });
  }

  async getStatus(): Promise<{ status: string; timestamp: string }> {
    return this.request<{ status: string; timestamp: string }>('/monitoring/status');
  }

  // API Info
  async getApiInfo(): Promise<any> {
    return this.request<any>('/info');
  }
}

// Create and export API instance
export const neuralNetworkAPI = new NeuralNetworkAPI();

// Utility functions
export const formatProcessingTime = (time: number): string => {
  if (time < 1) {
    return `${Math.round(time * 1000)}ms`;
  }
  return `${time.toFixed(2)}s`;
};

export const formatTokensPerSecond = (tokens: number, time: number): string => {
  if (time === 0) return '0 tokens/s';
  const rate = tokens / time;
  return `${Math.round(rate)} tokens/s`;
};

export const formatMemoryUsage = (gb: number): string => {
  if (gb < 1) {
    return `${Math.round(gb * 1024)}MB`;
  }
  return `${gb.toFixed(2)}GB`;
};

export const getSeverityColor = (severity: string): string => {
  switch (severity) {
    case 'low':
      return '#10B981'; // green
    case 'medium':
      return '#F59E0B'; // yellow
    case 'high':
      return '#EF4444'; // red
    case 'critical':
      return '#DC2626'; // dark red
    default:
      return '#6B7280'; // gray
  }
};

export const formatUptime = (seconds: number): string => {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`;
  } else if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else {
    return `${minutes}m`;
  }
};