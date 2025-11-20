#!/usr/bin/env python3
"""
Test the API fallback system
"""

import requests
import json
import time

API_BASE_URL = 'http://localhost:5001'

def test_api_fallback():
    """Test the API fallback functionality"""
    print("🎵 Testing API Fallback System")
    print("=" * 60)
    
    # Test health endpoint
    print("\n1. Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✓ API is healthy")
            print(f"    Model loaded: {health.get('model_loaded', False)}")
            print(f"    Device: {health.get('device', 'unknown')}")
        else:
            print(f"  ❌ API health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Cannot connect to API: {e}")
        print("  Make sure the API server is running: python3 api.py")
        return
    
    # Test emotions endpoint
    print("\n2. Testing emotions endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/emotions", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Emotions endpoint working")
            print(f"    Model trained: {data.get('model_trained', False)}")
            print(f"    Fallback enabled: {data.get('fallback_enabled', False)}")
            
            emotions = data.get('emotions', [])
            for emotion in emotions:
                fallback_status = "✓" if emotion.get('fallback_available', False) else "✗"
                print(f"    {emotion['name']}: {fallback_status}")
        else:
            print(f"  ❌ Emotions endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Emotions endpoint error: {e}")
    
    # Test text generation with fallback
    print("\n3. Testing text generation (fallback mode)...")
    test_inputs = [
        "I want happy music for 2 minutes",
        "Play me sad music for 1 minute", 
        "Give me calm music for 30 seconds"
    ]
    
    for text_input in test_inputs:
        print(f"\n  Testing: '{text_input}'")
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/generate",
                json={
                    "text": text_input,
                    "use_demo": True  # Force fallback mode
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"    ✓ Generated successfully")
                print(f"      ID: {result.get('generation_id', 'unknown')[:8]}...")
                print(f"      Emotion: {result.get('emotion', 'unknown')}")
                print(f"      Duration: {result.get('duration', 0)}m")
                print(f"      Demo mode: {result.get('demo_mode', False)}")
                print(f"      MIDI: {result.get('midi_file', 'none')}")
                if result.get('audio_file'):
                    print(f"      Audio: {result.get('audio_file')}")
            else:
                print(f"    ❌ Generation failed: {response.status_code}")
                print(f"      Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Request error: {e}")
    
    # Test emotion-based generation
    print("\n4. Testing emotion-based generation...")
    emotions_to_test = ['joy', 'sadness', 'calm']
    
    for emotion in emotions_to_test:
        print(f"\n  Testing emotion: {emotion}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/generate-emotion",
                json={
                    "emotion": emotion,
                    "duration": 1.0,
                    "use_demo": True  # Force fallback mode
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"    ✓ Generated successfully")
                print(f"      ID: {result.get('generation_id', 'unknown')[:8]}...")
                print(f"      Demo mode: {result.get('demo_mode', False)}")
            else:
                print(f"    ❌ Generation failed: {response.status_code}")
                print(f"      Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Request error: {e}")
    
    print("\n" + "=" * 60)
    print("✓ API fallback test complete")

if __name__ == "__main__":
    test_api_fallback()