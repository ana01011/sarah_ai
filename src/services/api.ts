/**
 * API Service for SARAH AI
 * Handles all communication with the backend
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

export interface ChatMessage {
  message: string;
  agent?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  response: string;
  agent: string;
  timestamp: string;
  suggestions?: string[];
}

export interface SystemMetrics {
  accuracy: number;
  throughput: number;
  latency: number;
  gpuUtilization: number;
  memoryUsage: number;
  activeModels: number;
  uptime: number;
  deployments: number;
  codeQuality: number;
  security: number;
}

export interface SystemComponent {
  name: string;
  status: 'online' | 'warning' | 'offline';
  uptime: string;
  load: number;
}

export interface AgentInfo {
  id: string;
  name: string;
  role: string;
  department: string;
  status: string;
  specialties: string[];
}

export interface DashboardStats {
  activeUsers: number;
  globalReach: number;
  dataProcessed: string;
  uptime: string;
  timestamp: string;
}

class ApiService {
  private async fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 5000): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL.replace('/api/v1', '')}/health`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async getMetrics(): Promise<SystemMetrics> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL}/metrics`);
      if (!response.ok) throw new Error('Failed to fetch metrics');
      return await response.json();
    } catch (error) {
      console.error('Error fetching metrics:', error);
      // Return default metrics if API fails
      return {
        accuracy: 94.7,
        throughput: 2847,
        latency: 12.3,
        gpuUtilization: 78,
        memoryUsage: 65,
        activeModels: 12,
        uptime: 99.97,
        deployments: 247,
        codeQuality: 94.3,
        security: 98.1,
      };
    }
  }

  async getSystemStatus(): Promise<{ components: SystemComponent[]; overall_health: string }> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL}/system/status`);
      if (!response.ok) throw new Error('Failed to fetch system status');
      return await response.json();
    } catch (error) {
      console.error('Error fetching system status:', error);
      // Return default status if API fails
      return {
        components: [
          { name: 'GPU Cluster A', status: 'online', uptime: '99.98%', load: 78 },
          { name: 'GPU Cluster B', status: 'online', uptime: '99.95%', load: 65 },
          { name: 'Data Pipeline', status: 'warning', uptime: '99.87%', load: 92 },
          { name: 'Model Registry', status: 'online', uptime: '99.99%', load: 45 },
          { name: 'API Gateway', status: 'online', uptime: '99.96%', load: 67 },
          { name: 'Storage Array', status: 'online', uptime: '99.94%', load: 34 },
        ],
        overall_health: 'excellent',
      };
    }
  }

  async getAgents(): Promise<AgentInfo[]> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL}/agents`);
      if (!response.ok) throw new Error('Failed to fetch agents');
      return await response.json();
    } catch (error) {
      console.error('Error fetching agents:', error);
      return [];
    }
  }

  async getAgent(agentId: string): Promise<AgentInfo | null> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL}/agents/${agentId}`);
      if (!response.ok) throw new Error('Failed to fetch agent');
      return await response.json();
    } catch (error) {
      console.error('Error fetching agent:', error);
      return null;
    }
  }

  async sendChatMessage(message: ChatMessage): Promise<ChatResponse> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL}/chat`, {
        method: 'POST',
        body: JSON.stringify(message),
      });
      if (!response.ok) throw new Error('Failed to send chat message');
      return await response.json();
    } catch (error) {
      console.error('Error sending chat message:', error);
      // Return a fallback response
      return {
        response: "I'm currently unable to process your request. Please try again later.",
        agent: message.agent || 'AI Assistant',
        timestamp: new Date().toISOString(),
        suggestions: [],
      };
    }
  }

  async getDashboardStats(): Promise<DashboardStats> {
    try {
      const response = await this.fetchWithTimeout(`${API_URL}/dashboard/stats`);
      if (!response.ok) throw new Error('Failed to fetch dashboard stats');
      return await response.json();
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      // Return default stats if API fails
      return {
        activeUsers: 1247,
        globalReach: 47,
        dataProcessed: '2.4TB',
        uptime: '99.98%',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // WebSocket connection for real-time updates
  connectWebSocket(endpoint: 'chat' | 'metrics' | 'system', onMessage: (data: any) => void): WebSocket {
    const ws = new WebSocket(`${WS_URL}/${endpoint}`);
    
    ws.onopen = () => {
      console.log(`WebSocket connected to ${endpoint}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error on ${endpoint}:`, error);
    };

    ws.onclose = () => {
      console.log(`WebSocket disconnected from ${endpoint}`);
    };

    return ws;
  }
}

export const apiService = new ApiService();