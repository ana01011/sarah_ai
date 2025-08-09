import React, { useState, useEffect } from 'react';
import { Activity, Brain, Cpu, Database, Zap, TrendingUp, AlertTriangle, CheckCircle, MessageCircle, Settings, Bell, Search, Filter, Download, Share, Maximize, BarChart3, Users, Globe } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { ThemeSelector } from './ThemeSelector';
import { MetricsCard } from './MetricsCard';
import { NeuralNetworkViz } from './NeuralNetworkViz';
import { PerformanceChart } from './PerformanceChart';
import { SystemStatus } from './SystemStatus';
import { ProcessingPipeline } from './ProcessingPipeline';
import { AIChat } from './AIChat';
import { useAgent } from '../contexts/AgentContext';
import { apiService, type SystemMetrics, type DashboardStats } from '../services/api';

interface DashboardProps {
  onBackToWelcome?: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onBackToWelcome }) => {
  const { currentTheme } = useTheme();
  const { setCurrentView } = useAgent();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [notifications, setNotifications] = useState(3);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [aiMetrics, setAiMetrics] = useState<SystemMetrics>({
    accuracy: 94.7,
    throughput: 2847,
    latency: 12.3,
    gpuUtilization: 78,
    memoryUsage: 65,
    activeModels: 12,
    uptime: 99.97,
    deployments: 247,
    codeQuality: 94.3,
    security: 98.1
  });
  const [dashboardStats, setDashboardStats] = useState<DashboardStats>({
    activeUsers: 1247,
    globalReach: 47,
    dataProcessed: '2.4TB',
    uptime: '99.98%',
    timestamp: new Date().toISOString()
  });
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    
    // Check backend health on mount
    const checkBackend = async () => {
      const isHealthy = await apiService.checkHealth();
      setIsBackendConnected(isHealthy);
    };
    checkBackend();
    
    // Fetch initial data
    const fetchInitialData = async () => {
      try {
        const [metrics, stats] = await Promise.all([
          apiService.getMetrics(),
          apiService.getDashboardStats()
        ]);
        setAiMetrics(metrics);
        setDashboardStats(stats);
      } catch (error) {
        console.error('Error fetching initial data:', error);
      }
    };
    fetchInitialData();
    
    // Set up periodic updates
    const metricsTimer = setInterval(async () => {
      try {
        const metrics = await apiService.getMetrics();
        setAiMetrics(metrics);
      } catch (error) {
        console.error('Error fetching metrics:', error);
        // Fall back to simulated updates if backend is down
        setAiMetrics(prev => ({
          ...prev,
          accuracy: prev.accuracy + (Math.random() - 0.5) * 0.2,
          throughput: prev.throughput + Math.floor((Math.random() - 0.5) * 100),
          latency: Math.max(8, prev.latency + (Math.random() - 0.5) * 2),
          gpuUtilization: Math.max(0, Math.min(100, prev.gpuUtilization + (Math.random() - 0.5) * 5)),
          memoryUsage: Math.max(0, Math.min(100, prev.memoryUsage + (Math.random() - 0.5) * 3)),
          activeModels: prev.activeModels + Math.floor((Math.random() - 0.5) * 2)
        }));
      }
    }, 2000);
    
    const statsTimer = setInterval(async () => {
      try {
        const stats = await apiService.getDashboardStats();
        setDashboardStats(stats);
      } catch (error) {
        console.error('Error fetching dashboard stats:', error);
      }
    }, 5000);

    return () => {
      clearInterval(timer);
      clearInterval(metricsTimer);
      clearInterval(statsTimer);
    };
  }, []);

  const handleNotificationClick = () => {
    setNotifications(0);
  };

  const handleExportData = () => {
    console.log('Exporting dashboard data...');
  };

  const handleShareDashboard = () => {
    console.log('Sharing dashboard...');
  };

  return (
    <div 
      className="min-h-screen overflow-hidden transition-all duration-500"
      style={{ 
        background: `linear-gradient(135deg, ${currentTheme.colors.background}, ${currentTheme.colors.surface})`,
        color: currentTheme.colors.text
      }}
    >
      {/* Animated Background */}
      <div className="fixed inset-0 opacity-10">
        <div 
          className="absolute top-0 left-0 w-[32rem] h-[32rem] rounded-full mix-blend-multiply filter blur-3xl animate-pulse"
          style={{ backgroundColor: currentTheme.colors.primary }}
        ></div>
        <div 
          className="absolute top-0 right-0 w-[28rem] h-[28rem] rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-1000"
          style={{ backgroundColor: currentTheme.colors.secondary }}
        ></div>
        <div 
          className="absolute bottom-0 left-1/2 w-[30rem] h-[30rem] rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-2000"
          style={{ backgroundColor: currentTheme.colors.accent }}
        ></div>
        <div 
          className="absolute -bottom-10 -right-10 w-[24rem] h-[24rem] rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-3000"
          style={{ backgroundColor: currentTheme.colors.primary + '40' }}
        ></div>
      </div>

      {/* Header */}
      <header 
        className="relative z-40 backdrop-blur-md border-b"
        style={{ 
          backgroundColor: currentTheme.colors.surface + '80',
          borderColor: currentTheme.colors.border
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 sm:space-x-4">
              <div className="relative">
                <div 
                  className="absolute -inset-2 rounded-full blur opacity-30 animate-pulse"
                  style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})` }}
                ></div>
                <Brain 
                  className="w-8 h-8 sm:w-10 sm:h-10 animate-pulse relative z-10" 
                  style={{ color: currentTheme.colors.primary }}
                />
                <div 
                  className="absolute -top-1 -right-1 w-3 h-3 sm:w-4 sm:h-4 rounded-full animate-ping"
                  style={{ backgroundColor: currentTheme.colors.secondary }}
                ></div>
                <div 
                  className="absolute -top-1 -right-1 w-3 h-3 sm:w-4 sm:h-4 rounded-full"
                  style={{ backgroundColor: currentTheme.colors.secondary }}
                ></div>
              </div>
              <div>
                <h1 
                  className="text-2xl sm:text-3xl font-bold bg-clip-text text-transparent"
                  style={{ 
                    backgroundImage: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})`
                  }}
                >
                  SARAH
                </h1>
                <p className="text-xs sm:text-sm flex items-center space-x-1 sm:space-x-2" style={{ color: currentTheme.colors.textSecondary }}>
                  <span className="hidden sm:inline">AI Operations Dashboard</span>
                  <span className="sm:hidden">AI Dashboard</span>
                  <span>•</span>
                  <span style={{ color: currentTheme.colors.secondary }}>v3.7.2</span>
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 sm:space-x-4">
              <ThemeSelector />
              
              <div className="relative hidden sm:block">
                <button
                  onClick={() => setIsSearchOpen(!isSearchOpen)}
                  className="p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95"
                >
                  <Search 
                    className="w-5 h-5 hover:text-white transition-colors" 
                    style={{ color: currentTheme.colors.textSecondary }}
                  />
                </button>
                {isSearchOpen && (
                  <div 
                    className="absolute top-full right-0 mt-2 w-80 backdrop-blur-md border rounded-xl p-4 shadow-2xl"
                    style={{ 
                      backgroundColor: currentTheme.colors.surface + 'f0',
                      borderColor: currentTheme.colors.border
                    }}
                  >
                    <input
                      type="text"
                      placeholder="Search metrics, models, or data..."
                      className="w-full border rounded-lg px-4 py-2 focus:outline-none transition-colors"
                      style={{ 
                        backgroundColor: currentTheme.colors.background + '80',
                        borderColor: currentTheme.colors.border,
                        color: currentTheme.colors.text
                      }}
                    />
                  </div>
                )}
              </div>

              <button
                onClick={handleNotificationClick}
                className="relative p-2 sm:p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95"
              >
                <Bell 
                  className="w-4 h-4 sm:w-5 sm:h-5 hover:text-white transition-colors" 
                  style={{ color: currentTheme.colors.textSecondary }}
                />
                {notifications > 0 && (
                  <div 
                    className="absolute -top-1 -right-1 w-4 h-4 sm:w-5 sm:h-5 text-white text-xs rounded-full flex items-center justify-center animate-pulse"
                    style={{ backgroundColor: currentTheme.colors.error }}
                  >
                    {notifications}
                  </div>
                )}
              </button>

              <button
                onClick={handleExportData}
                className="p-2 sm:p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                <Download 
                  className="w-4 h-4 sm:w-5 sm:h-5 hover:text-white transition-colors" 
                  style={{ color: currentTheme.colors.textSecondary }}
                />
              </button>

              <button
                onClick={handleShareDashboard}
                className="p-2 sm:p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                <Share 
                  className="w-4 h-4 sm:w-5 sm:h-5 hover:text-white transition-colors" 
                  style={{ color: currentTheme.colors.textSecondary }}
                />
              </button>

              <button
                onClick={() => setCurrentView('selector')}
                className="relative group border rounded-xl px-3 sm:px-6 py-2 sm:py-3 transition-all duration-300 
                         hover:scale-110 active:scale-95 hover:shadow-xl backdrop-blur-sm overflow-hidden"
                style={{
                  background: `linear-gradient(135deg, ${currentTheme.colors.secondary}20, ${currentTheme.colors.accent}20)`,
                  borderColor: currentTheme.colors.secondary + '50',
                  boxShadow: `0 10px 25px -5px ${currentTheme.shadows.secondary}`
                }}
              >
                <div 
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  style={{ background: 'linear-gradient(135deg, rgba(255,255,255,0.1), transparent)' }}
                ></div>
                <div className="flex items-center space-x-1 sm:space-x-2">
                  <Users 
                    className="w-4 h-4 sm:w-5 sm:h-5 transition-colors" 
                    style={{ color: currentTheme.colors.secondary }}
                  />
                  <span 
                    className="text-xs sm:text-sm font-semibold transition-colors relative z-10"
                    style={{ color: currentTheme.colors.text }}
                  >
                    <span className="hidden sm:inline">AI Agents</span>
                    <span className="sm:hidden">Agents</span>
                  </span>
                </div>
              </button>

              <button className="p-2 sm:p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block">
                <Settings 
                  className="w-4 h-4 sm:w-5 sm:h-5 hover:text-white transition-colors" 
                  style={{ color: currentTheme.colors.textSecondary }}
                />
              </button>

              <div className="text-right hidden lg:block">
                <p className="text-sm font-mono" style={{ color: currentTheme.colors.secondary }}>
                  {currentTime.toLocaleTimeString()}
                </p>
                <p className="text-xs" style={{ color: currentTheme.colors.textSecondary }}>
                  {currentTime.toLocaleDateString()}
                </p>
              </div>
              
              <div className="flex items-center space-x-2 hidden md:flex">
                {isBackendConnected ? (
                  <>
                    <CheckCircle className="w-4 h-4" style={{ color: currentTheme.colors.success }} />
                    <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.success }}>Backend Connected</span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4" style={{ color: currentTheme.colors.warning }} />
                    <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.warning }}>Backend Offline (Demo Mode)</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-4 sm:py-8">
        {/* Integrated Chat Section */}
        <div 
          className="mb-4 sm:mb-8 backdrop-blur-xl border rounded-2xl overflow-hidden"
          style={{
            background: `linear-gradient(135deg, ${currentTheme.colors.surface}80, ${currentTheme.colors.surface}40)`,
            borderColor: currentTheme.colors.border,
            height: '400px'
          }}
        >
          <AIChat 
            isOpen={true} 
            onClose={() => {}} 
            isIntegrated={true}
          />
        </div>

        {/* Quick Stats Bar */}
        <div 
          className="mb-4 sm:mb-8 backdrop-blur-md border rounded-2xl p-4 sm:p-6"
          style={{ 
            background: `linear-gradient(135deg, ${currentTheme.colors.surface}80, ${currentTheme.colors.surface}40)`,
            borderColor: currentTheme.colors.border
          }}
        >
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-6">
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <Users className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.primary }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Active Users</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>{dashboardStats.activeUsers.toLocaleString()}</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>+12% today</p>
            </div>
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <Globe className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.secondary }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Global Reach</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>{dashboardStats.globalReach}</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>countries</p>
            </div>
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.accent }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Data Processed</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>{dashboardStats.dataProcessed}</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>today</p>
            </div>
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <Zap className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.info }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Uptime</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>{dashboardStats.uptime}</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>30 days</p>
            </div>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-6 mb-4 sm:mb-8">
          <MetricsCard
            title="Model Accuracy"
            value={`${aiMetrics.accuracy.toFixed(1)}%`}
            change="+2.3%"
            icon={TrendingUp}
            color="success"
          />
          <MetricsCard
            title="Throughput"
            value={`${aiMetrics.throughput.toLocaleString()}`}
            change="+15.2%"
            icon={Zap}
            color="primary"
            suffix=" req/s"
          />
          <MetricsCard
            title="Avg Latency"
            value={`${aiMetrics.latency.toFixed(1)}`}
            change="-8.7%"
            icon={Activity}
            color="warning"
            suffix="ms"
          />
          <MetricsCard
            title="GPU Usage"
            value={`${aiMetrics.gpuUtilization.toFixed(0)}%`}
            change="+5.1%"
            icon={Cpu}
            color="info"
          />
          <MetricsCard
            title="Memory"
            value={`${aiMetrics.memoryUsage.toFixed(0)}%`}
            change="-2.4%"
            icon={Database}
            color="error"
          />
          <MetricsCard
            title="Active Models"
            value={aiMetrics.activeModels.toString()}
            change="+3"
            icon={Brain}
            color="secondary"
          />
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-8 mb-4 sm:mb-8">
          <div className="lg:col-span-2 order-2 lg:order-1">
            <NeuralNetworkViz />
          </div>
          <div className="order-1 lg:order-2">
            <SystemStatus />
          </div>
        </div>

        {/* Performance Charts and Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-8">
          <PerformanceChart metrics={aiMetrics} />
          <ProcessingPipeline />
        </div>
      </div>
    </div>
  );
};