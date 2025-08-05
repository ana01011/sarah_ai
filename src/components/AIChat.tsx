import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageCircle, 
  Send, 
  X, 
  Brain, 
  Zap, 
  Mic, 
  MicOff, 
  Paperclip, 
  MoreVertical,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Sparkles,
  Settings,
  Download,
  Share,
  Maximize2,
  Minimize2,
  Volume2,
  VolumeX,
  Image,
  FileText,
  Code,
  BarChart3,
  Lightbulb,
  Rocket,
  Shield,
  Star
} from 'lucide-react';
import { apiService } from '../services/api';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  typing?: boolean;
  suggestions?: string[];
  attachments?: string[];
  reactions?: { type: string; count: number }[];
}

interface AIChatProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AIChat: React.FC<AIChatProps> = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Hello! I'm Sarah, your advanced AI assistant. I can help you with system monitoring, data analysis, model optimization, code generation, and much more. What would you like to explore today?",
      sender: 'ai',
      timestamp: new Date(),
      suggestions: [
        "🚀 Show me system performance",
        "📊 Analyze model accuracy trends", 
        "⚡ Check GPU utilization",
        "🔧 Optimize training pipeline",
        "💡 Generate code snippets",
        "📈 Create performance reports"
      ]
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [selectedModel, setSelectedModel] = useState('GPT-4 Turbo');
  const [chatTheme, setChatTheme] = useState('dark');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const playSound = (type: 'send' | 'receive' | 'notification') => {
    if (!soundEnabled) return;
    // Sound implementation would go here
    console.log(`Playing ${type} sound`);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    playSound('send');
    
    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      // Send request to backend
      const response = await apiService.generateConversation({
        prompt: inputValue,
        max_length: 2048,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50
      });

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.text,
        sender: 'ai',
        timestamp: new Date(),
        suggestions: [
          "📊 Show detailed metrics",
          "⚡ Apply optimizations", 
          "📋 Export report",
          "🔧 Schedule maintenance"
        ],
        reactions: [
          { type: '👍', count: 0 },
          { type: '❤️', count: 0 },
          { type: '🚀', count: 0 }
        ]
      };

      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
      playSound('receive');
    } catch (error) {
      console.error('Failed to generate response:', error);
      
      // Fallback response
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I apologize, but I'm having trouble connecting to the AI model right now. Please try again in a moment, or check if the backend service is running.",
        sender: 'ai',
        timestamp: new Date(),
        suggestions: ["🔄 Try again", "📊 Check system status", "🔧 Restart service"],
        reactions: [
          { type: '👍', count: 0 },
          { type: '❤️', count: 0 },
          { type: '🚀', count: 0 }
        ]
      };

      setMessages(prev => [...prev, fallbackMessage]);
      setIsTyping(false);
      playSound('receive');
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    // Remove emoji and set as input
    const cleanSuggestion = suggestion.replace(/^[^\w\s]+\s*/, '');
    setInputValue(cleanSuggestion);
    inputRef.current?.focus();
  };

  const toggleVoice = () => {
    setIsListening(!isListening);
    playSound('notification');
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    playSound('notification');
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const exportChat = () => {
    const chatData = JSON.stringify(messages, null, 2);
    const blob = new Blob([chatData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sarah-chat-export.json';
    a.click();
    playSound('notification');
  };

  const shareChat = () => {
    if (navigator.share) {
      navigator.share({
        title: 'Sarah AI Chat',
        text: 'Check out my conversation with Sarah AI!',
        url: window.location.href
      });
    }
    playSound('notification');
  };

  const addReaction = (messageId: string, reactionType: string) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id === messageId && msg.reactions) {
        return {
          ...msg,
          reactions: msg.reactions.map(reaction => 
            reaction.type === reactionType 
              ? { ...reaction, count: reaction.count + 1 }
              : reaction
          )
        };
      }
      return msg;
    }));
    playSound('notification');
  };

  if (!isOpen) return null;

  return (
    <>
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        accept="image/*,.pdf,.txt,.doc,.docx"
        onChange={(e) => {
          if (e.target.files) {
            playSound('notification');
            // Handle file upload
          }
        }}
      />
      
      <div className={`fixed inset-0 z-[9999] flex items-center justify-center p-2 sm:p-4 bg-black/60 backdrop-blur-sm transition-all duration-300 ${isMaximized ? 'p-0' : ''}`}>
        <div className={`w-full bg-gradient-to-br from-slate-900/98 to-slate-800/98 backdrop-blur-xl border border-white/20 shadow-2xl flex flex-col overflow-hidden transition-all duration-500 hover:shadow-blue-500/20 ${
          isMaximized 
            ? 'h-full rounded-none max-w-none' 
            : 'max-w-6xl h-[90vh] sm:h-[85vh] rounded-xl sm:rounded-2xl hover:scale-[1.01]'
        }`}>
          {/* Enhanced Header */}
          <div className="flex items-center justify-between p-3 sm:p-6 border-b border-white/10 bg-gradient-to-r from-white/5 to-white/10 backdrop-blur-md">
            <div className="flex items-center space-x-2 sm:space-x-4">
              <div className="relative group">
                <div className="absolute -inset-2 bg-gradient-to-r from-blue-500 to-emerald-500 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity"></div>
                <Brain className="w-8 h-8 sm:w-10 sm:h-10 text-blue-400 animate-pulse relative z-10" />
                <div className="absolute -top-1 -right-1 w-3 h-3 sm:w-4 sm:h-4 bg-emerald-400 rounded-full animate-ping"></div>
                <div className="absolute -top-1 -right-1 w-3 h-3 sm:w-4 sm:h-4 bg-emerald-400 rounded-full"></div>
              </div>
              <div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                  Sarah AI Assistant
                </h2>
                <div className="flex items-center space-x-2 sm:space-x-3">
                  <div className="flex items-center space-x-1 text-emerald-400">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                    <span className="text-xs sm:text-sm font-medium">Online • {selectedModel}</span>
                  </div>
                  <div className="text-xs text-slate-400 bg-white/10 px-2 py-1 rounded-full hidden sm:block">
                    Advanced Mode
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-1 sm:space-x-2">
              <button
                onClick={() => setSoundEnabled(!soundEnabled)}
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                {soundEnabled ? 
                  <Volume2 className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-white" /> : 
                  <VolumeX className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-white" />
                }
              </button>
              
              <button
                onClick={exportChat}
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                <Download className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-white" />
              </button>
              
              <button
                onClick={shareChat}
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
              >
                <Share className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-white" />
              </button>
              
              <button
                onClick={() => setIsMaximized(!isMaximized)}
                className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95"
              >
                {isMaximized ? 
                  <Minimize2 className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-white" /> : 
                  <Maximize2 className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-white" />
                }
              </button>
              
              <button
                onClick={onClose}
                className="p-1.5 sm:p-2 hover:bg-rose-500/20 hover:border-rose-500/30 border border-transparent rounded-lg transition-all duration-200 hover:scale-110 active:scale-95"
              >
                <X className="w-4 h-4 sm:w-5 sm:h-5 text-slate-400 hover:text-rose-400" />
              </button>
            </div>
          </div>

          {/* Enhanced Messages */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-6 space-y-4 sm:space-y-6 custom-scrollbar">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} group`}
              >
                <div className={`max-w-[90%] sm:max-w-[85%] ${message.sender === 'user' ? 'order-2' : 'order-1'}`}>
                  {message.sender === 'ai' && (
                    <div className="flex items-center space-x-2 mb-2 sm:mb-3">
                      <div className="relative">
                        <Brain className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400" />
                        <div className="absolute -top-1 -right-1 w-2 h-2 bg-emerald-400 rounded-full"></div>
                      </div>
                      <span className="text-xs sm:text-sm text-slate-400 font-medium">Sarah AI</span>
                      <span className="text-xs text-slate-500">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                      <div className="flex items-center space-x-1 hidden sm:flex">
                        <Star className="w-3 h-3 text-amber-400" />
                        <span className="text-xs text-amber-400">Premium</span>
                      </div>
                    </div>
                  )}
                  
                  <div
                    className={`
                      relative p-3 sm:p-5 rounded-xl sm:rounded-2xl backdrop-blur-md border transition-all duration-300 hover:scale-[1.02] hover:shadow-lg group-hover:shadow-xl
                      ${message.sender === 'user'
                        ? 'bg-gradient-to-r from-blue-500/20 to-emerald-500/20 border-blue-500/40 text-white ml-auto hover:from-blue-500/30 hover:to-emerald-500/30'
                        : 'bg-white/5 border-white/20 text-white hover:bg-white/10'
                      }
                    `}
                  >
                    {/* Message glow effect */}
                    <div className={`absolute -inset-0.5 bg-gradient-to-r ${
                      message.sender === 'user' 
                        ? 'from-blue-500/20 to-emerald-500/20' 
                        : 'from-white/10 to-white/5'
                    } rounded-2xl opacity-0 group-hover:opacity-100 blur transition-opacity duration-300`}></div>
                    
                    <div className="relative z-10">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                      
                      {message.sender === 'ai' && (
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-white/10 space-y-2 sm:space-y-0">
                          <div className="flex items-center space-x-2 sm:space-x-3">
                            <button
                              onClick={() => copyMessage(message.content)}
                              className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 group/btn"
                            >
                              <Copy className="w-3 h-3 sm:w-4 sm:h-4 text-slate-400 group-hover/btn:text-white" />
                            </button>
                            <button
                              onClick={() => addReaction(message.id, '👍')}
                              className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 group/btn"
                            >
                              <ThumbsUp className="w-3 h-3 sm:w-4 sm:h-4 text-slate-400 group-hover/btn:text-emerald-400" />
                            </button>
                            <button
                              onClick={() => addReaction(message.id, '❤️')}
                              className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 group/btn"
                            >
                              <ThumbsDown className="w-3 h-3 sm:w-4 sm:h-4 text-slate-400 group-hover/btn:text-rose-400" />
                            </button>
                            <button className="p-1.5 sm:p-2 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 group/btn">
                              <RefreshCw className="w-3 h-3 sm:w-4 sm:h-4 text-slate-400 group-hover/btn:text-blue-400" />
                            </button>
                          </div>
                          
                          <div className="flex items-center space-x-2 sm:space-x-3">
                            {message.reactions && (
                              <div className="flex items-center space-x-1">
                                {message.reactions.map((reaction, idx) => (
                                  <button
                                    key={idx}
                                    onClick={() => addReaction(message.id, reaction.type)}
                                    className="flex items-center space-x-1 px-1.5 sm:px-2 py-1 bg-white/5 hover:bg-white/10 rounded-full transition-all duration-200 hover:scale-110"
                                  >
                                    <span className="text-xs sm:text-sm">{reaction.type}</span>
                                    {reaction.count > 0 && (
                                      <span className="text-xs text-slate-400">{reaction.count}</span>
                                    )}
                                  </button>
                                ))}
                              </div>
                            )}
                            
                            <div className="flex items-center space-x-1">
                              <Sparkles className="w-3 h-3 text-amber-400 animate-pulse" />
                              <span className="text-xs text-amber-400 font-medium">AI Generated</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {message.suggestions && (
                    <div className="mt-3 sm:mt-4 flex flex-wrap gap-2">
                      {message.suggestions.map((suggestion, index) => (
                        <button
                          key={index}
                          onClick={() => handleSuggestionClick(suggestion)}
                          className="px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm bg-gradient-to-r from-white/5 to-white/10 hover:from-white/10 hover:to-white/15 
                                   border border-white/20 hover:border-white/30 rounded-full text-slate-300 hover:text-white 
                                   transition-all duration-200 hover:scale-105 active:scale-95 backdrop-blur-sm
                                   hover:shadow-lg hover:shadow-blue-500/20"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-white/5 border border-white/20 rounded-xl sm:rounded-2xl p-3 sm:p-5 backdrop-blur-md hover:bg-white/10 transition-all duration-300">
                  <div className="flex items-center space-x-3">
                    <Brain className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400 animate-pulse" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce delay-100"></div>
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce delay-200"></div>
                    </div>
                    <span className="text-xs sm:text-sm text-slate-400">Sarah is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Enhanced Input */}
          <div className="p-3 sm:p-6 border-t border-white/10 bg-gradient-to-r from-white/5 to-white/10 backdrop-blur-md">
            <div className="flex items-center space-x-2 sm:space-x-4 mb-3 sm:mb-4">
              <div className="flex-1 relative group">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask Sarah anything..."
                  className="w-full bg-white/5 border border-white/20 rounded-xl px-4 sm:px-6 py-3 sm:py-4 pr-12 sm:pr-16 
                           text-white placeholder-slate-400 focus:outline-none focus:border-blue-500/50 
                           focus:bg-white/10 transition-all duration-200 hover:bg-white/10 hover:border-white/30
                           focus:shadow-lg focus:shadow-blue-500/20"
                />
                <div className="absolute right-2 sm:right-4 top-1/2 transform -translate-y-1/2 flex items-center space-x-1 sm:space-x-2">
                  <button
                    onClick={handleFileUpload}
                    className="p-1 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
                  >
                    <Paperclip className="w-3 h-3 sm:w-4 sm:h-4 text-slate-400 hover:text-white" />
                  </button>
                  <button
                    onClick={() => {}}
                    className="p-1 hover:bg-white/10 rounded-lg transition-all duration-200 hover:scale-110 active:scale-95 hidden sm:block"
                  >
                    <Image className="w-3 h-3 sm:w-4 sm:h-4 text-slate-400 hover:text-white" />
                  </button>
                </div>
              </div>
              
              <button
                onClick={toggleVoice}
                className={`p-3 sm:p-4 rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 ${
                  isListening 
                    ? 'bg-rose-500/20 border-rose-500/40 text-rose-400 shadow-lg shadow-rose-500/20' 
                    : 'bg-white/5 border-white/20 text-slate-400 hover:bg-white/10 hover:border-white/30'
                } border backdrop-blur-sm`}
              >
                {isListening ? <MicOff className="w-4 h-4 sm:w-5 sm:h-5" /> : <Mic className="w-4 h-4 sm:w-5 sm:h-5" />}
              </button>
              
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim()}
                className="p-3 sm:p-4 bg-gradient-to-r from-blue-500 to-emerald-500 hover:from-blue-600 hover:to-emerald-600 
                         disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed
                         text-white rounded-xl transition-all duration-200 hover:scale-110 active:scale-95 disabled:scale-100
                         hover:shadow-lg hover:shadow-blue-500/30 disabled:shadow-none backdrop-blur-sm
                         relative overflow-hidden group"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <Send className="w-4 h-4 sm:w-5 sm:h-5 relative z-10" />
              </button>
            </div>
            
            {/* Quick Actions */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs space-y-2 sm:space-y-0">
              <div className="flex items-center space-x-3 sm:space-x-6 text-slate-400">
                <div className="flex items-center space-x-2">
                  <kbd className="px-2 py-1 bg-white/10 rounded text-xs">Enter</kbd>
                  <span>to send</span>
                </div>
                <div className="flex items-center space-x-2 hidden sm:flex">
                  <kbd className="px-2 py-1 bg-white/10 rounded text-xs">Shift</kbd>
                  <span>+</span>
                  <kbd className="px-2 py-1 bg-white/10 rounded text-xs">Enter</kbd>
                  <span>for new line</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-2 sm:space-x-4">
                <div className="flex items-center space-x-2">
                  <Code className="w-3 h-3 text-blue-400" />
                  <span className="text-slate-400 hidden sm:inline">Code generation ready</span>
                  <span className="text-slate-400 sm:hidden">Code ready</span>
                </div>
                <div className="flex items-center space-x-2 hidden sm:flex">
                  <BarChart3 className="w-3 h-3 text-emerald-400" />
                  <span className="text-slate-400">Analytics enabled</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Zap className="w-3 h-3 text-amber-400 animate-pulse" />
                  <span className="text-amber-400 font-medium text-xs">Sarah AI v3.7.2</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};