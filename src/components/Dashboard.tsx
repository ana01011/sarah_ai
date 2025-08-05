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
import { neuralNetworkAPI, ComprehensiveMetrics, Alert, formatUptime, formatMemoryUsage } from '../services/api';

interface DashboardProps {
  onBackToWelcome?: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onBackToWelcome }) => {
  const { currentTheme } = useTheme();
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [notifications, setNotifications] = useState(3);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<ComprehensiveMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isConnected, setIsConnected] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch real-time metrics from backend
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [metricsData, alertsData] = await Promise.all([
          neuralNetworkAPI.getMetrics(),
          neuralNetworkAPI.getAlerts(true) // active alerts only
        ]);
        
        setSystemMetrics(metricsData);
        setAlerts(alertsData);
        setIsConnected(true);
        setLastUpdated(new Date());
        
        // Update notifications count based on alerts
        setNotifications(alertsData.filter(alert => alert.severity === 'high' || alert.severity === 'critical').length);
        
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
        setIsConnected(false);
      }
    };

    // Initial fetch
    fetchMetrics();

    // Set up real-time updates every 5 seconds
    const metricsTimer = setInterval(fetchMetrics, 5000);

    return () => clearInterval(metricsTimer);
  }, []);

  const handleNotificationClick = () => {
    setNotifications(0);
    // Show notifications panel
  };

  const handleExportData = () => {
    // Export dashboard data
    console.log('Exporting dashboard data...');
  };

  const handleShareDashboard = () => {
    // Share dashboard
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
      <div className="fixed inset-0 opacity-20">
        <div 
          className="absolute top-0 left-0 w-96 h-96 rounded-full mix-blend-multiply filter blur-xl animate-pulse"
          style={{ backgroundColor: currentTheme.colors.primary }}
        ></div>
        <div 
          className="absolute top-0 right-0 w-96 h-96 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-1000"
          style={{ backgroundColor: currentTheme.colors.secondary }}
        ></div>
        <div 
          className="absolute bottom-0 left-1/2 w-96 h-96 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-2000"
          style={{ backgroundColor: currentTheme.colors.accent }}
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
              {/* Theme Selector */}
              <ThemeSelector />
              
              {/* Search */}
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

              {/* Notifications */}
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

              {/* Export */}
              <button
                onClick={handleExportData}
                className="p-2 sm:p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                <Download 
                  className="w-4 h-4 sm:w-5 sm:h-5 hover:text-white transition-colors" 
                  style={{ color: currentTheme.colors.textSecondary }}
                />
              </button>

              {/* Share */}
              <button
                onClick={handleShareDashboard}
                className="p-2 sm:p-3 hover:bg-white/10 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                <Share 
                  className="w-4 h-4 sm:w-5 sm:h-5 hover:text-white transition-colors" 
                  style={{ color: currentTheme.colors.textSecondary }}
                />
              </button>

              {/* Settings */}
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
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </p>
              </div>
              
              <div className="flex items-center space-x-2 hidden md:flex">
                {isConnected ? (
                  <>
                    <CheckCircle className="w-4 h-4" style={{ color: currentTheme.colors.success }} />
                    <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.success }}>
                      Neural Network Online
                    </span>
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-4 h-4" style={{ color: currentTheme.colors.error }} />
                    <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.error }}>
                      Connection Lost
                    </span>
                  </>
                )}
              </div>
              
              {/* AI Chat Button */}
              <button
                onClick={() => setIsChatOpen(true)}
                className="relative group bg-gradient-to-r from-blue-500/20 to-emerald-500/20 
                         hover:from-blue-500/30 hover:to-emerald-500/30 border border-blue-500/30 
                         hover:border-emerald-500/50 rounded-xl px-3 sm:px-6 py-2 sm:py-3 transition-all duration-300 
                         hover:scale-110 active:scale-95 hover:shadow-xl hover:shadow-blue-500/30
                         backdrop-blur-sm overflow-hidden"
                style={{
                  background: `linear-gradient(135deg, ${currentTheme.colors.primary}20, ${currentTheme.colors.secondary}20)`,
                  borderColor: currentTheme.colors.primary + '50',
                  boxShadow: `0 10px 25px -5px ${currentTheme.shadows.primary}`
                }}
              >
                <div 
                  className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  style={{ background: 'linear-gradient(135deg, rgba(255,255,255,0.1), transparent)' }}
                ></div>
                <div className="flex items-center space-x-1 sm:space-x-2">
                  <div className="relative">
                    <MessageCircle 
                      className="w-4 h-4 sm:w-5 sm:h-5 transition-colors" 
                      style={{ color: currentTheme.colors.primary }}
                    />
                    <div 
                      className="absolute -top-1 -right-1 w-2 h-2 sm:w-3 sm:h-3 rounded-full animate-ping"
                      style={{ backgroundColor: currentTheme.colors.secondary }}
                    ></div>
                    <div 
                      className="absolute -top-1 -right-1 w-2 h-2 sm:w-3 sm:h-3 rounded-full"
                      style={{ backgroundColor: currentTheme.colors.secondary }}
                    ></div>
                  </div>
                  <span 
                    className="text-xs sm:text-sm font-semibold transition-colors relative z-10"
                    style={{ color: currentTheme.colors.text }}
                  >
                    <span className="hidden sm:inline">Ask Sarah</span>
                    <span className="sm:hidden">Chat</span>
                  </span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 py-4 sm:py-8">
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
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>1,247</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>+12% today</p>
            </div>
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <Globe className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.secondary }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Global Reach</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>47</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>countries</p>
            </div>
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.accent }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Data Processed</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>2.4TB</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>today</p>
            </div>
            <div className="text-center group cursor-pointer hover:scale-105 transition-transform duration-200">
              <div className="flex items-center justify-center space-x-1 sm:space-x-2 mb-1 sm:mb-2">
                <Zap className="w-4 h-4 sm:w-5 sm:h-5" style={{ color: currentTheme.colors.info }} />
                <span className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>Uptime</span>
              </div>
              <p className="text-lg sm:text-2xl font-bold font-mono" style={{ color: currentTheme.colors.text }}>99.98%</p>
              <p className="text-xs" style={{ color: currentTheme.colors.success }}>30 days</p>
            </div>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-6 mb-4 sm:mb-8">
          <MetricsCard
            title="Requests/Sec"
            value={systemMetrics?.performance.requests_per_second.toFixed(1) || "0"}
            change={isConnected ? "+15.2%" : "N/A"}
            icon={Zap}
            color={isConnected ? "primary" : "error"}
          />
          <MetricsCard
            title="Avg Response"
            value={systemMetrics?.performance.avg_processing_time.toFixed(2) || "0"}
            change={isConnected ? "-8.7%" : "N/A"}
            icon={Activity}
            color={isConnected ? "success" : "error"}
            suffix="s"
          />
          <MetricsCard
            title="CPU Usage"
            value={`${systemMetrics?.resources.cpu_usage_percent.toFixed(0) || "0"}%`}
            change={isConnected ? "+5.1%" : "N/A"}
            icon={Cpu}
            color={isConnected ? "info" : "error"}
          />
          <MetricsCard
            title="Memory"
            value={`${systemMetrics?.resources.memory_usage_percent.toFixed(0) || "0"}%`}
            change={isConnected ? "-2.4%" : "N/A"}
            icon={Database}
            color={isConnected ? "warning" : "error"}
          />
          <MetricsCard
            title="Cache Hit Rate"
            value={`${((systemMetrics?.cache.hit_rate || 0) * 100).toFixed(1)}%`}
            change={isConnected ? "+12.3%" : "N/A"}
            icon={TrendingUp}
            color={isConnected ? "success" : "error"}
          />
          <MetricsCard
            title="Total Requests"
            value={systemMetrics?.performance.request_count.toLocaleString() || "0"}
            change={isConnected ? "+847" : "N/A"}
            icon={Brain}
            color={isConnected ? "secondary" : "error"}
          />
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-8 mb-4 sm:mb-8">
          {/* Neural Network Visualization */}
          <div className="lg:col-span-2 order-2 lg:order-1">
            <NeuralNetworkViz />
          </div>

          {/* System Status */}
          <div className="order-1 lg:order-2">
            <SystemStatus />
          </div>
        </div>

        {/* Performance Charts and Pipeline */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-8">
          <PerformanceChart metrics={aiMetrics} />
          <ProcessingPipeline />
        </div>
      </main>
      
      {/* AI Chat */}
      <AIChat isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} />
    </div>
  );
};