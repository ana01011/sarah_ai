import React, { useState, useEffect } from 'react';
import { Brain, ArrowRight, Sparkles, Zap, Shield, Cpu } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface WelcomeScreenProps {
  onEnter: () => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onEnter }) => {
  const { currentTheme } = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [currentSystem, setCurrentSystem] = useState(0);

  const systems = [
    'Neural Networks',
    'Data Pipeline',
    'GPU Clusters',
    'Model Registry',
    'Security Layer',
    'API Gateway'
  ];

  useEffect(() => {
    const loadingTimer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          setIsLoading(false);
          clearInterval(loadingTimer);
          return 100;
        }
        return prev + 2;
      });
    }, 50);

    const systemTimer = setInterval(() => {
      setCurrentSystem(prev => (prev + 1) % systems.length);
    }, 800);

    return () => {
      clearInterval(loadingTimer);
      clearInterval(systemTimer);
    };
  }, []);

  return (
    <div 
      className="min-h-screen overflow-hidden relative transition-all duration-500"
      style={{ 
        background: `linear-gradient(135deg, ${currentTheme.colors.background}, ${currentTheme.colors.surface})`,
        color: currentTheme.colors.text
      }}
    >
      {/* Animated Background */}
      <div className="fixed inset-0">
        <div 
          className="absolute top-0 left-0 w-96 h-96 rounded-full mix-blend-multiply filter blur-xl animate-pulse opacity-20"
          style={{ backgroundColor: currentTheme.colors.primary }}
        ></div>
        <div 
          className="absolute top-0 right-0 w-96 h-96 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-1000 opacity-20"
          style={{ backgroundColor: currentTheme.colors.secondary }}
        ></div>
        <div 
          className="absolute bottom-0 left-1/2 w-96 h-96 rounded-full mix-blend-multiply filter blur-xl animate-pulse delay-2000 opacity-20"
          style={{ backgroundColor: currentTheme.colors.accent }}
        ></div>
        
        {/* Floating particles */}
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 rounded-full opacity-30 animate-ping"
            style={{
              backgroundColor: currentTheme.colors.primary,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 flex items-center justify-center min-h-screen">
        <div className="text-center max-w-4xl mx-auto px-4 sm:px-6">
          {/* Logo and Branding */}
          <div className="mb-8 sm:mb-12">
            <div className="relative inline-block">
              <div 
                className="absolute -inset-4 rounded-full blur-lg opacity-30 animate-pulse"
                style={{ background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})` }}
              ></div>
              <Brain 
                className="w-16 h-16 sm:w-24 sm:h-24 mx-auto relative animate-pulse" 
                style={{ color: currentTheme.colors.primary }}
              />
            </div>
            
            <h1 
              className="text-5xl sm:text-8xl font-bold bg-clip-text text-transparent mt-6 sm:mt-8 mb-3 sm:mb-4 animate-pulse"
              style={{ 
                backgroundImage: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary}, ${currentTheme.colors.accent})`
              }}
            >
              SARAH
            </h1>
            
            <p className="text-lg sm:text-2xl mb-2 font-light" style={{ color: currentTheme.colors.textSecondary }}>
              Synthetic Autonomous Reasoning & Analysis Hub
            </p>
            
            <div className="flex items-center justify-center space-x-1 sm:space-x-2" style={{ color: currentTheme.colors.secondary }}>
              <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" style={{ color: currentTheme.colors.secondary }} />
              <span className="text-sm sm:text-lg font-mono">AI Operations Platform v3.7.2</span>
              <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" style={{ color: currentTheme.colors.secondary }} />
            </div>
          </div>

          {/* Loading Section */}
          {isLoading ? (
            <div className="space-y-6 sm:space-y-8">
              <div 
                className="backdrop-blur-md border rounded-2xl p-6 sm:p-8 max-w-md mx-auto"
                style={{ 
                  backgroundColor: currentTheme.colors.surface + '80',
                  borderColor: currentTheme.colors.border
                }}
              >
                <div className="space-y-4 sm:space-y-6">
                  <div className="flex items-center justify-center space-x-3">
                    <Cpu className="w-5 h-5 sm:w-6 sm:h-6 animate-spin" style={{ color: currentTheme.colors.primary }} />
                    <span className="text-base sm:text-lg font-medium">Initializing Systems</span>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>
                      <span>Loading {systems[currentSystem]}...</span>
                      <span className="font-mono" style={{ color: currentTheme.colors.secondary }}>{progress}%</span>
                    </div>
                    
                    <div 
                      className="w-full rounded-full h-2 sm:h-3 overflow-hidden"
                      style={{ backgroundColor: currentTheme.colors.background }}
                    >
                      <div
                        className="h-full transition-all duration-300 ease-out"
                        style={{ 
                          background: `linear-gradient(90deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})`,
                          width: `${progress}%`
                        }}
                      />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-xs">
                    <div className="text-center">
                      <Shield className="w-3 h-3 sm:w-4 sm:h-4 mx-auto mb-1" style={{ color: currentTheme.colors.secondary }} />
                      <span style={{ color: currentTheme.colors.textSecondary }}>Secure</span>
                    </div>
                    <div className="text-center">
                      <Zap className="w-3 h-3 sm:w-4 sm:h-4 mx-auto mb-1" style={{ color: currentTheme.colors.accent }} />
                      <span style={{ color: currentTheme.colors.textSecondary }}>Fast</span>
                    </div>
                    <div className="text-center">
                      <Brain className="w-3 h-3 sm:w-4 sm:h-4 mx-auto mb-1" style={{ color: currentTheme.colors.primary }} />
                      <span style={{ color: currentTheme.colors.textSecondary }}>Smart</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6 sm:space-y-8 animate-fade-in">
              <div 
                className="backdrop-blur-md border rounded-2xl p-6 sm:p-8 max-w-2xl mx-auto"
                style={{ 
                  backgroundColor: currentTheme.colors.surface + '80',
                  borderColor: currentTheme.colors.border
                }}
              >
                <div className="space-y-4 sm:space-y-6">
                  <div className="flex items-center justify-center space-x-2" style={{ color: currentTheme.colors.secondary }}>
                    <div 
                      className="w-3 h-3 rounded-full animate-pulse"
                      style={{ backgroundColor: currentTheme.colors.secondary }}
                    ></div>
                    <span className="text-base sm:text-lg font-medium">All Systems Online</span>
                  </div>
                  
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6 text-xs sm:text-sm">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span style={{ color: currentTheme.colors.textSecondary }}>GPU Clusters</span>
                        <span style={{ color: currentTheme.colors.secondary }}>✓ Active</span>
                      </div>
                      <div className="flex justify-between">
                        <span style={{ color: currentTheme.colors.textSecondary }}>Neural Networks</span>
                        <span style={{ color: currentTheme.colors.secondary }}>✓ Ready</span>
                      </div>
                      <div className="flex justify-between">
                        <span style={{ color: currentTheme.colors.textSecondary }}>Data Pipeline</span>
                        <span style={{ color: currentTheme.colors.secondary }}>✓ Streaming</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span style={{ color: currentTheme.colors.textSecondary }}>Model Registry</span>
                        <span style={{ color: currentTheme.colors.secondary }}>✓ Synced</span>
                      </div>
                      <div className="flex justify-between">
                        <span style={{ color: currentTheme.colors.textSecondary }}>Security Layer</span>
                        <span style={{ color: currentTheme.colors.secondary }}>✓ Protected</span>
                      </div>
                      <div className="flex justify-between">
                        <span style={{ color: currentTheme.colors.textSecondary }}>API Gateway</span>
                        <span style={{ color: currentTheme.colors.secondary }}>✓ Online</span>
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={onEnter}
                    className="w-full font-semibold py-4 sm:py-5 px-6 sm:px-8 rounded-xl transition-all duration-300 
                             hover:scale-110 active:scale-95 hover:shadow-2xl hover:shadow-blue-500/30 
                             flex items-center justify-center space-x-3 group relative overflow-hidden
                             backdrop-blur-sm border"
                    style={{
                      background: `linear-gradient(135deg, ${currentTheme.colors.primary}, ${currentTheme.colors.secondary})`,
                      color: currentTheme.id === 'light' ? '#ffffff' : currentTheme.colors.text,
                      borderColor: currentTheme.colors.border,
                      boxShadow: `0 10px 25px -5px ${currentTheme.shadows.primary}`
                    }}
                  >
                    <div 
                      className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                      style={{ background: 'linear-gradient(135deg, rgba(255,255,255,0.2), transparent)' }}
                    ></div>
                    <span className="text-base sm:text-lg">Enter Dashboard</span>
                    <ArrowRight className="w-5 h-5 sm:w-6 sm:h-6 group-hover:translate-x-2 transition-transform relative z-10" />
                  </button>
                </div>
              </div>
              
              <p className="text-xs sm:text-sm" style={{ color: currentTheme.colors.textSecondary }}>
                Advanced AI Operations • Real-time Analytics • Neural Network Management
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};