import React, { useEffect, useRef, useState } from 'react';
import { Brain, Layers, Zap } from 'lucide-react';
import { apiService, type NeuralNetworkMetrics } from '../services/api';

export const NeuralNetworkViz: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const [stats, setStats] = useState<NeuralNetworkMetrics | null>(null);

  useEffect(() => {
    // Fetch neural network stats from backend
    apiService.getNeuralNetworkMetrics()
      .then(setStats)
      .catch(() => {
        // fallback to hardcoded values
        setStats({
          layers: [
            { name: 'Input Layer', neurons: 4, activation: 'Linear' },
            { name: 'Hidden Layer 1', neurons: 6, activation: 'ReLU' },
            { name: 'Hidden Layer 2', neurons: 6, activation: 'ReLU' },
            { name: 'Output Layer', neurons: 3, activation: 'Softmax' },
          ],
          total_parameters: '2.1M',
          model_size: '14.5GB',
          quantization: '4-bit',
          memory_efficiency: '85%'
        });
      });
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const nodes = [
      // Input layer
      ...Array.from({ length: 4 }, (_, i) => ({ x: 50, y: 50 + i * 40, layer: 0, active: Math.random() > 0.3 })),
      // Hidden layer 1
      ...Array.from({ length: 6 }, (_, i) => ({ x: 150, y: 30 + i * 30, layer: 1, active: Math.random() > 0.3 })),
      // Hidden layer 2
      ...Array.from({ length: 6 }, (_, i) => ({ x: 250, y: 30 + i * 30, layer: 2, active: Math.random() > 0.3 })),
      // Output layer
      ...Array.from({ length: 3 }, (_, i) => ({ x: 350, y: 70 + i * 40, layer: 3, active: Math.random() > 0.3 })),
    ];

    let time = 0;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
      time += 0.02;

      // Draw connections
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
      ctx.lineWidth = 1;
      
      for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < nodes.length; j++) {
          if (nodes[j].layer === nodes[i].layer + 1) {
            const strength = Math.sin(time + i * 0.5 + j * 0.3) * 0.5 + 0.5;
            ctx.strokeStyle = `rgba(59, 130, 246, ${strength * 0.6})`;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      nodes.forEach((node, i) => {
        const pulse = Math.sin(time * 2 + i * 0.3) * 0.3 + 0.7;
        const size = node.active ? 8 * pulse : 6;
        
        // Outer glow
        ctx.shadowColor = node.active ? '#3b82f6' : '#64748b';
        ctx.shadowBlur = node.active ? 15 : 5;
        
        ctx.fillStyle = node.active ? '#3b82f6' : '#64748b';
        ctx.beginPath();
        ctx.arc(node.x, node.y, size, 0, Math.PI * 2);
        ctx.fill();
        
        // Inner highlight
        ctx.shadowBlur = 0;
        ctx.fillStyle = node.active ? '#93c5fd' : '#94a3b8';
        ctx.beginPath();
        ctx.arc(node.x, node.y, size * 0.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Randomly toggle node activity
      if (Math.random() < 0.1) {
        const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
        randomNode.active = !randomNode.active;
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div className="backdrop-blur-md bg-white/5 border border-white/10 rounded-xl p-6 h-full">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Brain className="w-6 h-6 text-blue-400" />
          <h2 className="text-lg font-semibold text-white">Neural Network Topology</h2>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
            <span className="text-sm text-slate-300">Active</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-slate-500 rounded-full"></div>
            <span className="text-sm text-slate-300">Inactive</span>
          </div>
        </div>
      </div>

      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full h-64 rounded-lg"
          style={{ width: '100%', height: '256px' }}
        />
        {/* Layer labels */}
        <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4">
          <span className="text-xs text-slate-400">Input</span>
          <span className="text-xs text-slate-400">Hidden</span>
          <span className="text-xs text-slate-400">Hidden</span>
          <span className="text-xs text-slate-400">Output</span>
        </div>
      </div>

      {/* Model Stats */}
      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="text-center">
          <Layers className="w-5 h-5 text-emerald-400 mx-auto mb-1" />
          <p className="text-xs text-slate-400">Layers</p>
          <p className="text-sm font-mono text-white">{stats ? stats.layers.length : '...'}</p>
        </div>
        <div className="text-center">
          <Zap className="w-5 h-5 text-amber-400 mx-auto mb-1" />
          <p className="text-xs text-slate-400">Parameters</p>
          <p className="text-sm font-mono text-white">{stats ? stats.total_parameters : '...'}</p>
        </div>
        <div className="text-center">
          <Brain className="w-5 h-5 text-purple-400 mx-auto mb-1" />
          <p className="text-xs text-slate-400">Neurons</p>
          <p className="text-sm font-mono text-white">{stats ? stats.layers.reduce((sum, l) => sum + l.neurons, 0) : '...'}</p>
        </div>
      </div>
    </div>
  );
};