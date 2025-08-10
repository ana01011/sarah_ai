#!/usr/bin/env python3
"""Test script for deployed Amesie AI Backend"""

import requests
import json
import time

SERVER_URL = "http://147.93.102.165"

def test_health():
    """Test health endpoint"""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_roles():
    """Test available roles"""
    print("\n2. Testing Available Roles...")
    response = requests.get(f"{SERVER_URL}/api/v1/chat/roles")
    if response.status_code == 200:
        roles = response.json()
        print(f"   Available roles: {len(roles)} roles")
        for role in roles[:3]:  # Show first 3 roles
            print(f"   - {role}")
        return True
    else:
        print(f"   Error: {response.status_code}")
        return False

def test_chat(role="software_engineer"):
    """Test chat completion"""
    print(f"\n3. Testing Chat Completion with '{role}' role...")
    
    prompt = "Write a simple Python function to check if a number is prime"
    
    data = {
        "prompt": prompt,
        "role": role,
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    print(f"   Sending prompt: '{prompt[:50]}...'")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{SERVER_URL}/api/v1/chat/completion",
            json=data,
            timeout=30
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: Success")
            print(f"   Response time: {elapsed:.2f} seconds")
            print(f"   Model: {result.get('model', 'unknown')}")
            print(f"   Response preview: {result['text'][:150]}...")
            return True
        else:
            print(f"   Error: {response.status_code}")
            print(f"   Details: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"   Timeout after 30 seconds")
        return False
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\n4. Testing Metrics...")
    response = requests.get(f"{SERVER_URL}/api/v1/metrics")
    if response.status_code == 200:
        metrics = response.json()
        print(f"   CPU Usage: {metrics.get('cpu_percent', 'N/A')}%")
        print(f"   Memory Usage: {metrics.get('memory_percent', 'N/A')}%")
        print(f"   Available Memory: {metrics.get('memory_available_gb', 'N/A')} GB")
        return True
    else:
        print(f"   Error: {response.status_code}")
        return False

def main():
    print("=" * 60)
    print("Testing Amesie AI Backend Deployment")
    print(f"Server: {SERVER_URL}")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Roles", test_roles()))
    results.append(("Chat Completion", test_chat()))
    results.append(("Metrics", test_metrics()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Your AI backend is fully operational!")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()