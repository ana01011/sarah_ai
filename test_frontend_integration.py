#!/usr/bin/env python3
"""
Frontend Integration Test Script
Tests the frontend API service and integration components
"""
import os
import json
import sys
from pathlib import Path

def test_frontend_structure():
    """Test frontend file structure"""
    print("🎨 Testing frontend structure...")
    
    required_files = [
        "src/components/AIChat.tsx",
        "src/components/Dashboard.tsx", 
        "src/services/api.ts",
        "package.json",
        "tsconfig.json",
        "tailwind.config.js",
        "vite.config.ts"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def test_package_json():
    """Test package.json configuration"""
    print("\n📦 Testing package.json...")
    
    try:
        with open("package.json", "r") as f:
            package_data = json.load(f)
        
        # Check required dependencies
        required_deps = ["react", "react-dom", "lucide-react"]
        missing_deps = []
        
        dependencies = package_data.get("dependencies", {})
        for dep in required_deps:
            if dep in dependencies:
                print(f"✅ {dep}: {dependencies[dep]}")
            else:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing dependencies: {missing_deps}")
            return False
        
        # Check scripts
        scripts = package_data.get("scripts", {})
        required_scripts = ["dev", "build"]
        for script in required_scripts:
            if script in scripts:
                print(f"✅ Script '{script}': {scripts[script]}")
            else:
                print(f"❌ Missing script: {script}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error reading package.json: {e}")
        return False

def test_api_service():
    """Test the API service TypeScript file"""
    print("\n🌐 Testing API service...")
    
    try:
        with open("src/services/api.ts", "r") as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            "class NeuralNetworkAPI",
            "sendMessage",
            "getMetrics",
            "getHealth",
            "createChatStream",
            "export const neuralNetworkAPI"
        ]
        
        missing_components = []
        for component in required_components:
            if component in content:
                print(f"✅ {component} found")
            else:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        
        # Check for API endpoints
        endpoints = [
            "/chat/chat",
            "/monitoring/health", 
            "/monitoring/metrics",
            "/chat/conversations"
        ]
        
        for endpoint in endpoints:
            if endpoint in content:
                print(f"✅ Endpoint {endpoint} configured")
            else:
                print(f"⚠️ Endpoint {endpoint} not found")
        
        # Check for WebSocket support
        if "WebSocket" in content and "createChatStream" in content:
            print("✅ WebSocket streaming support configured")
        else:
            print("⚠️ WebSocket streaming support missing")
        
        return True
    except Exception as e:
        print(f"❌ Error reading API service: {e}")
        return False

def test_chat_component():
    """Test the AIChat component"""
    print("\n💬 Testing AIChat component...")
    
    try:
        with open("src/components/AIChat.tsx", "r") as f:
            content = f.read()
        
        # Check for key features
        required_features = [
            "neuralNetworkAPI.sendMessage",
            "handleSendMessage",
            "conversationId",
            "isConnected",
            "formatProcessingTime"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature in content:
                print(f"✅ {feature} implemented")
            else:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
        
        # Check for error handling
        error_handling = [
            "try {",
            "catch (error)",
            "Connection Lost",
            "error generating response"
        ]
        
        for handler in error_handling:
            if handler in content:
                print(f"✅ Error handling: {handler}")
            else:
                print(f"⚠️ Error handling missing: {handler}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading AIChat component: {e}")
        return False

def test_dashboard_integration():
    """Test Dashboard integration with backend"""
    print("\n📊 Testing Dashboard integration...")
    
    try:
        with open("src/components/Dashboard.tsx", "r") as f:
            content = f.read()
        
        # Check for API integration
        api_integration = [
            "neuralNetworkAPI.getMetrics",
            "neuralNetworkAPI.getAlerts",
            "ComprehensiveMetrics",
            "systemMetrics",
            "isConnected",
            "lastUpdated"
        ]
        
        missing_integration = []
        for integration in api_integration:
            if integration in content:
                print(f"✅ {integration} integrated")
            else:
                missing_integration.append(integration)
        
        if missing_integration:
            print(f"❌ Missing integration: {missing_integration}")
            return False
        
        # Check for real-time updates
        realtime_features = [
            "fetchMetrics",
            "setInterval",
            "5000",  # 5 second intervals
            "performance.requests_per_second",
            "resources.cpu_usage_percent"
        ]
        
        for feature in realtime_features:
            if feature in content:
                print(f"✅ Real-time feature: {feature}")
            else:
                print(f"⚠️ Real-time feature missing: {feature}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading Dashboard component: {e}")
        return False

def test_build_configuration():
    """Test build configuration"""
    print("\n🔧 Testing build configuration...")
    
    try:
        # Check Vite config
        with open("vite.config.ts", "r") as f:
            vite_content = f.read()
        
        if "@vitejs/plugin-react" in vite_content:
            print("✅ React plugin configured")
        else:
            print("❌ React plugin missing")
            return False
        
        # Check TypeScript config
        with open("tsconfig.json", "r") as f:
            ts_config = json.load(f)
        
        if "compilerOptions" in ts_config or "references" in ts_config:
            print("✅ TypeScript configuration found")
            # Also check tsconfig.app.json if references exist
            if "references" in ts_config:
                try:
                    with open("tsconfig.app.json", "r") as f:
                        app_config = json.load(f)
                    if "compilerOptions" in app_config:
                        print("✅ TypeScript app configuration found")
                except:
                    pass
        else:
            print("❌ TypeScript configuration missing")
            return False
        
        # Check Tailwind config
        if Path("tailwind.config.js").exists():
            print("✅ Tailwind CSS configured")
        else:
            print("❌ Tailwind CSS configuration missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error checking build configuration: {e}")
        return False

def test_docker_configuration():
    """Test Docker configuration"""
    print("\n🐳 Testing Docker configuration...")
    
    docker_files = [
        "docker-compose.yml",
        "Dockerfile.frontend", 
        "backend/Dockerfile",
        "nginx-frontend.conf"
    ]
    
    missing_files = []
    for file_path in docker_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing Docker files: {missing_files}")
        return False
    
    # Check docker-compose structure
    try:
        with open("docker-compose.yml", "r") as f:
            content = f.read()
        
        services = [
            "neural-backend",
            "neural-frontend", 
            "postgres",
            "redis",
            "nginx"
        ]
        
        for service in services:
            if service in content:
                print(f"✅ Service {service} configured")
            else:
                print(f"⚠️ Service {service} missing")
        
        return True
    except Exception as e:
        print(f"❌ Error reading docker-compose.yml: {e}")
        return False

def main():
    """Run all frontend integration tests"""
    print("🚀 Frontend Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Frontend Structure", test_frontend_structure),
        ("Package Configuration", test_package_json),
        ("API Service", test_api_service),
        ("Chat Component", test_chat_component),
        ("Dashboard Integration", test_dashboard_integration),
        ("Build Configuration", test_build_configuration),
        ("Docker Configuration", test_docker_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All frontend integration tests passed!")
        print("\n🚀 Ready to test with:")
        print("   npm run dev  (development)")
        print("   npm run build && npm run preview  (production)")
        print("   ./start.sh  (full stack with Docker)")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)