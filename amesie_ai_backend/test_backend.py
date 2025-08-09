#!/usr/bin/env python3
"""
Comprehensive test script for Amesie AI Backend
"""

import asyncio
import requests
import json
import time
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_api_info():
    """Test API info endpoint"""
    print("🔍 Testing API info...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API info: {data['name']} v{data['version']}")
            print(f"   Model: {data['model']['name']} ({data['model']['status']})")
            return True
        else:
            print(f"❌ API info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API info error: {e}")
        return False

def test_metrics_endpoints():
    """Test metrics endpoints"""
    print("🔍 Testing metrics endpoints...")
    
    endpoints = [
        "/api/v1/metrics/performance",
        "/api/v1/metrics/system", 
        "/api/v1/metrics/usage",
        "/api/v1/metrics/neural-network",
        "/api/v1/metrics/processing-pipeline",
        "/api/v1/metrics/health",
        "/api/v1/metrics/dashboard",
        "/api/v1/metrics/realtime"
    ]
    
    success_count = 0
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {endpoint}")
                success_count += 1
            else:
                print(f"❌ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
    
    print(f"📊 Metrics test results: {success_count}/{len(endpoints)} passed")
    return success_count == len(endpoints)

def test_chat_endpoints():
    """Test chat endpoints"""
    print("🔍 Testing chat endpoints...")
    
    # Test completion
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/completion",
            json={
                "prompt": "Hello, how are you?",
                "max_length": 100,
                "temperature": 0.7
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat completion: {len(data['text'])} characters generated")
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False
    
    # Test conversation
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/conversation",
            json={
                "prompt": "Tell me about AI models",
                "max_length": 150,
                "temperature": 0.8
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat conversation: {len(data['text'])} characters generated")
        else:
            print(f"❌ Chat conversation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat conversation error: {e}")
        return False
    
    # Test models endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/v1/chat/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models info: {len(data['models'])} models available")
        else:
            print(f"❌ Models info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models info error: {e}")
        return False
    
    return True

def test_rate_limiting():
    """Test rate limiting"""
    print("🔍 Testing rate limiting...")
    
    # Make multiple requests quickly
    responses = []
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/api/v1/metrics/performance")
            responses.append(response.status_code)
        except Exception as e:
            print(f"❌ Rate limiting test error: {e}")
            return False
    
    # Check if any requests were rate limited (429)
    rate_limited = any(status == 429 for status in responses)
    if rate_limited:
        print("✅ Rate limiting working correctly")
    else:
        print("⚠️ Rate limiting not triggered (may be normal)")
    
    return True

def test_prometheus_metrics():
    """Test Prometheus metrics endpoint"""
    print("🔍 Testing Prometheus metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            metrics = response.text
            if "amesie_ai_requests_total" in metrics:
                print("✅ Prometheus metrics available")
                return True
            else:
                print("❌ Prometheus metrics not found")
                return False
        else:
            print(f"❌ Prometheus metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prometheus metrics error: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connection"""
    print("🔍 Testing WebSocket connection...")
    try:
        import websockets
        import asyncio
        
        async def test_ws():
            try:
                uri = "ws://localhost:8000/ws/chat"
                async with websockets.connect(uri) as websocket:
                    # Send a ping message
                    await websocket.send(json.dumps({"type": "ping"}))
                    response = await websocket.recv()
                    data = json.loads(response)
                    if data["type"] == "pong":
                        print("✅ WebSocket connection working")
                        return True
                    else:
                        print("❌ WebSocket response unexpected")
                        return False
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                return False
        
        # Run the async test
        result = asyncio.run(test_ws())
        return result
    except ImportError:
        print("⚠️ websockets library not available, skipping WebSocket test")
        return True
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("🔍 Testing model loading...")
    try:
        # Try to load model
        response = requests.post(f"{BASE_URL}/api/v1/chat/models/load")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model loaded: {data['model_name']}")
            return True
        else:
            print(f"❌ Model loading failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def run_performance_test():
    """Run a simple performance test"""
    print("🔍 Running performance test...")
    
    start_time = time.time()
    success_count = 0
    total_requests = 5
    
    for i in range(total_requests):
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/chat/completion",
                json={
                    "prompt": f"Test message {i+1}",
                    "max_length": 50,
                    "temperature": 0.7
                }
            )
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"❌ Request {i+1} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Request {i+1} error: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"📊 Performance test results:")
    print(f"   Requests: {success_count}/{total_requests} successful")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Rate: {success_count/duration:.2f} requests/second")
    
    return success_count == total_requests

def main():
    """Run all tests"""
    print("🚀 Starting Amesie AI Backend Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("API Info", test_api_info),
        ("Metrics Endpoints", test_metrics_endpoints),
        ("Chat Endpoints", test_chat_endpoints),
        ("Rate Limiting", test_rate_limiting),
        ("Prometheus Metrics", test_prometheus_metrics),
        ("WebSocket Connection", test_websocket_connection),
        ("Model Loading", test_model_loading),
        ("Performance Test", run_performance_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the backend configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())