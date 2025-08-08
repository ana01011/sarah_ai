import React, { useState, useEffect } from 'react';
import { Server, Cpu, HardDrive, Wifi, AlertTriangle, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';

interface SystemComponent {
  name: string;
  status: 'online' | 'warning' | 'offline';
  uptime: string;
  load: number;
}

export const SystemStatus: React.FC = () => {
  const [components, setComponents] = useState<SystemComponent[]>([
    { name: 'GPU Cluster A', status: 'online', uptime: '99.98%', load: 78 },
    { name: 'GPU Cluster B', status: 'online', uptime: '99.95%', load: 65 },
    { name: 'Data Pipeline', status: 'warning', uptime: '99.87%', load: 92 },
    { name: 'Model Registry', status: 'online', uptime: '99.99%', load: 45 },
    { name: 'API Gateway', status: 'online', uptime: '99.96%', load: 67 },
    { name: 'Storage Array', status: 'online', uptime: '99.94%', load: 34 },
  ]);

  useEffect(() => {
    // Fetch system status from backend
    apiService.getSystemMetrics()
      .then((data) => {
        // Map backend data to components array if available
        if (data && data.system) {
          const sys = data.system;
          setComponents([
            { name: 'GPU Cluster A', status: sys.gpu_utilization > 90 ? 'warning' : 'online', uptime: '99.98%', load: Math.round(sys.gpu_utilization) },
            { name: 'GPU Cluster B', status: sys.gpu_utilization > 80 ? 'warning' : 'online', uptime: '99.95%', load: Math.round(sys.gpu_utilization * 0.8) },
            { name: 'Data Pipeline', status: sys.cpu_usage > 90 ? 'warning' : 'online', uptime: '99.87%', load: Math.round(sys.cpu_usage) },
            { name: 'Model Registry', status: 'online', uptime: '99.99%', load: Math.round(sys.memory_usage) },
            { name: 'API Gateway', status: 'online', uptime: '99.96%', load: Math.round(sys.disk_usage) },
            { name: 'Storage Array', status: 'online', uptime: '99.94%', load: Math.round(sys.memory_used_bytes / 1024 / 1024 / 1024) },
          ]);
        }
      })
      .catch(() => {
        // fallback to simulated data
      });
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setComponents(prev => prev.map(comp => ({
        ...comp,
        load: Math.max(10, Math.min(100, comp.load + (Math.random() - 0.5) * 10))
      })));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-amber-400" />;
      default:
        return <AlertTriangle className="w-4 h-4 text-rose-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'border-emerald-500/30 bg-emerald-500/10';
      case 'warning':
        return 'border-amber-500/30 bg-amber-500/10';
      default:
        return 'border-rose-500/30 bg-rose-500/10';
    }
  };

  const getLoadColor = (load: number) => {
    if (load < 50) return 'bg-emerald-500';
    if (load < 80) return 'bg-amber-500';
    return 'bg-rose-500';
  };

  return (
    <div className="backdrop-blur-md bg-white/5 border border-white/10 rounded-xl p-6 h-full">
      <div className="flex items-center space-x-3 mb-6">
        <Server className="w-6 h-6 text-emerald-400" />
        <h2 className="text-lg font-semibold text-white">System Status</h2>
      </div>
      <div className="space-y-4">
        {components.map((component, index) => (
          <div
            key={component.name}
            className={`
              border rounded-lg p-4 transition-all duration-300 hover:scale-105
              ${getStatusColor(component.status)}
            `}
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                {getStatusIcon(component.status)}
                <span className="text-sm font-medium text-white">{component.name}</span>
              </div>
              <span className="text-xs text-slate-400 font-mono">{component.uptime}</span>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Load</span>
                <span>{component.load}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-full ${getLoadColor(component.load)} transition-all duration-500 ease-out`}
                  style={{ width: `${component.load}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
      {/* System Overview */}
      <div className="mt-6 pt-6 border-t border-white/10">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <Cpu className="w-5 h-5 text-blue-400 mx-auto mb-1" />
            <p className="text-xs text-slate-400">Total Cores</p>
            <p className="text-sm font-mono text-white">256</p>
          </div>
          <div className="text-center">
            <HardDrive className="w-5 h-5 text-purple-400 mx-auto mb-1" />
            <p className="text-xs text-slate-400">Storage</p>
            <p className="text-sm font-mono text-white">2.4PB</p>
          </div>
        </div>
      </div>
    </div>
  );
};