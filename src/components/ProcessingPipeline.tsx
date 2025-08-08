import React, { useState, useEffect } from 'react';
import { ArrowRight, Database, Cpu, Brain, CheckCircle, Clock, AlertTriangle } from 'lucide-react';
import { apiService, type ProcessingPipelineMetrics } from '../services/api';

interface PipelineStage {
  id: string;
  name: string;
  status: 'processing' | 'completed' | 'waiting' | 'error';
  progress: number;
  throughput: number;
  icon: React.ComponentType<any>;
}

export const ProcessingPipeline: React.FC = () => {
  const [stages, setStages] = useState<PipelineStage[]>([{
    id: 'ingest', name: 'Data Ingestion', status: 'completed', progress: 100, throughput: 1247, icon: Database },
    { id: 'preprocess', name: 'Preprocessing', status: 'processing', progress: 67, throughput: 892, icon: Cpu },
    { id: 'training', name: 'Model Training', status: 'processing', progress: 43, throughput: 156, icon: Brain },
    { id: 'validation', name: 'Validation', status: 'waiting', progress: 0, throughput: 0, icon: CheckCircle },
  ]);
  const [pipelineStats, setPipelineStats] = useState<{ totalProcessed: string; successRate: string }>({ totalProcessed: '847.2K', successRate: '99.7%' });

  useEffect(() => {
    // Fetch pipeline metrics from backend
    apiService.getProcessingPipelineMetrics()
      .then((metrics: ProcessingPipelineMetrics) => {
        // Map backend data to local stage format
        if (metrics && metrics.stages) {
          setStages(metrics.stages.map((stage, idx) => ({
            id: stage.name.toLowerCase().replace(/\s+/g, '-'),
            name: stage.name,
            status: stage.status === 'active' ? 'processing' : (stage.status as any),
            progress: Math.round(stage.efficiency),
            throughput: stage.throughput,
            icon: [Database, Cpu, Brain, CheckCircle][idx % 4],
          })));
          setPipelineStats({
            totalProcessed: metrics.total_throughput?.toLocaleString() || 'N/A',
            successRate: metrics.overall_efficiency ? metrics.overall_efficiency.toFixed(1) + '%' : 'N/A',
          });
        }
      })
      .catch(() => {
        // fallback to simulated data
      });
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setStages(prev => prev.map(stage => {
        if (stage.status === 'processing') {
          const newProgress = Math.min(100, stage.progress + Math.random() * 3);
          const newStatus = newProgress === 100 ? 'completed' : 'processing';
          return {
            ...stage,
            progress: newProgress,
            status: newStatus,
            throughput: stage.throughput + Math.floor((Math.random() - 0.5) * 50)
          };
        }
        if (stage.status === 'waiting' && prev.find(s => s.id === getNextStage(stage.id))?.status === 'completed') {
          return { ...stage, status: 'processing', progress: 1 };
        }
        return stage;
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const getNextStage = (currentId: string) => {
    const stageOrder = stages.map(s => s.id);
    const currentIndex = stageOrder.indexOf(currentId);
    return stageOrder[currentIndex - 1];
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
      case 'processing':
        return <Clock className="w-4 h-4 text-blue-400 animate-spin" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-rose-400" />;
      default:
        return <Clock className="w-4 h-4 text-slate-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'border-emerald-500/50 bg-emerald-500/10';
      case 'processing':
        return 'border-blue-500/50 bg-blue-500/10';
      case 'error':
        return 'border-rose-500/50 bg-rose-500/10';
      default:
        return 'border-slate-500/50 bg-slate-500/10';
    }
  };

  return (
    <div className="backdrop-blur-md bg-white/5 border border-white/10 rounded-xl p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Cpu className="w-6 h-6 text-blue-400" />
        <h2 className="text-lg font-semibold text-white">Processing Pipeline</h2>
      </div>

      <div className="space-y-4">
        {stages.map((stage, index) => {
          const Icon = stage.icon;
          return (
            <div key={stage.id} className="relative">
              <div className={`
                border rounded-lg p-4 transition-all duration-500
                ${getStatusColor(stage.status)}
                ${stage.status === 'processing' ? 'animate-pulse' : ''}
              `}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="relative">
                      <Icon className="w-5 h-5 text-white" />
                      <div className="absolute -top-1 -right-1">
                        {getStatusIcon(stage.status)}
                      </div>
                    </div>
                    <span className="text-sm font-medium text-white">{stage.name}</span>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-400">Throughput</p>
                    <p className="text-sm font-mono text-white">{stage.throughput.toLocaleString()}/s</p>
                  </div>
                </div>
                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>Progress</span>
                    <span>{stage.progress.toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ease-out ${
                        stage.status === 'completed' ? 'bg-emerald-500' :
                        stage.status === 'processing' ? 'bg-blue-500' :
                        stage.status === 'error' ? 'bg-rose-500' : 'bg-slate-500'
                      }`}
                      style={{ width: `${stage.progress}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              {/* Arrow connector */}
              {index < stages.length - 1 && (
                <div className="flex justify-center my-2">
                  <ArrowRight className="w-4 h-4 text-slate-400" />
                </div>
              )}
            </div>
          );
        })}
      </div>
      {/* Pipeline Stats */}
      <div className="mt-6 pt-6 border-t border-white/10">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <p className="text-xs text-slate-400">Total Processed</p>
            <p className="text-lg font-mono text-emerald-400">{pipelineStats.totalProcessed}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-slate-400">Success Rate</p>
            <p className="text-lg font-mono text-emerald-400">{pipelineStats.successRate}</p>
          </div>
        </div>
      </div>
    </div>
  );
};