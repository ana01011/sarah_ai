#!/usr/bin/env python3
"""
Test script for Amesie AI Backend API
"""

import requests
import json
import time
from datetime import datetime

# Base URL for the API
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health Status: {data['status']}")
        print(f"   Timestamp: {data['timestamp']}")
        print(f"   Services:")
        for service, status in data['services'].items():
            print(f"     - {service}: {status}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

def test_roles():
    """Test roles endpoint"""
    print_section("Testing Available Roles")
    response = requests.get(f"{BASE_URL}/api/v1/chat/roles")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data['roles'])} roles:")
        for role in data['roles']:
            domain = data['configurations'][role]['domain']
            print(f"   - {role}: {domain}")
    else:
        print(f"❌ Roles fetch failed: {response.status_code}")

def test_chat_completion():
    """Test chat completion with different roles"""
    print_section("Testing Chat Completion")
    
    test_cases = [
        {
            "role": "CTO",
            "prompt": "What's the best programming language for microservices?"
        },
        {
            "role": "CFO", 
            "prompt": "How can we reduce operational costs?"
        },
        {
            "role": "CEO",
            "prompt": "What should be our company's growth strategy?"
        }
    ]
    
    for test in test_cases:
        print(f"\n📝 Testing {test['role']} with: '{test['prompt'][:50]}...'")
        
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/completion",
            json={
                "prompt": test['prompt'],
                "role": test['role'],
                "max_length": 500,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response received:")
            print(f"   Model: {data['model']}")
            print(f"   Inference Time: {data['inference_time']:.2f}s")
            print(f"   Response: {data['text'][:200]}...")
        else:
            print(f"❌ Chat completion failed: {response.status_code}")

def test_metrics():
    """Test metrics endpoints"""
    print_section("Testing Metrics Endpoints")
    
    endpoints = [
        "/api/v1/metrics/performance",
        "/api/v1/metrics/system",
        "/api/v1/metrics/dashboard"
    ]
    
    for endpoint in endpoints:
        response = requests.get(f"{BASE_URL}{endpoint}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {endpoint}:")
            # Print first few key-value pairs
            for i, (key, value) in enumerate(data.items()):
                if i >= 3:  # Limit output
                    print(f"     ... and {len(data) - 3} more fields")
                    break
                if isinstance(value, dict):
                    print(f"   - {key}: {len(value)} sub-fields")
                else:
                    print(f"   - {key}: {value}")
        else:
            print(f"❌ {endpoint} failed: {response.status_code}")

def test_performance():
    """Test API performance with multiple requests"""
    print_section("Testing Performance")
    
    print("Sending 5 rapid requests...")
    start_time = time.time()
    
    for i in range(5):
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/completion",
            json={
                "prompt": f"Test request {i+1}",
                "role": "AI Assistant",
                "max_length": 100
            }
        )
        print(f"   Request {i+1}: {'✅' if response.status_code == 200 else '❌'}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Completed 5 requests in {elapsed:.2f} seconds")
    print(f"   Average: {elapsed/5:.2f} seconds per request")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" AMESIE AI BACKEND API TEST SUITE")
    print("="*60)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Testing: {BASE_URL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        if response.status_code == 200:
            print(f" ✅ Server is running")
        else:
            print(f" ⚠️  Server responded with status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f" ❌ Cannot connect to server: {e}")
        print("\n Please ensure the server is running:")
        print("   cd /workspace/amesie_ai_backend")
        print("   source venv/bin/activate")
        print("   python main_simple.py")
        return
    
    # Run all tests
    test_health()
    test_roles()
    test_chat_completion()
    test_metrics()
    test_performance()
    
    print_section("Test Suite Complete")
    print("✅ All tests completed successfully!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()