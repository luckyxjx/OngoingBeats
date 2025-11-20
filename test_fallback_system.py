#!/usr/bin/env python3
"""
Test script for the fallback system
"""

from src.generation.fallback_manager import FallbackManager
from src.generation.text_parser import parse_text_input

def test_fallback_system():
    """Test the complete fallback system"""
    print("🎵 Testing Fallback System")
    print("=" * 60)
    
    # Test text parsing
    print("\n1. Testing text parsing:")
    test_inputs = [
        "I want happy music for 2 minutes",
        "Play me some sad music for 3 minutes", 
        "Give me angry intense music for 1 minute",
        "I need calm peaceful music for 4 minutes",
        "Create scary mysterious music for 2.5 minutes",
        "Make surprising weird music for 1.5 minutes"
    ]
    
    for text in test_inputs:
        parsed = parse_text_input(text)
        print(f"  '{text}'")
        print(f"    -> Emotion: {parsed['emotion']}, Duration: {parsed['duration_minutes']}m")
    
    # Test fallback manager
    print("\n2. Testing fallback manager:")
    try:
        manager = FallbackManager()
        
        # Test emotion normalization
        print("\n  Emotion synonym handling:")
        synonyms = ['happy', 'sad', 'angry', 'peaceful', 'scared', 'shocked']
        for synonym in synonyms:
            normalized = manager._normalize_emotion(synonym)
            print(f"    '{synonym}' -> '{normalized}'")
        
        # Test file availability
        print("\n  Available fallback files:")
        available = manager.list_available_files()
        for emotion, durations in available.items():
            if durations:
                print(f"    {emotion}: {len(durations)} files ({durations}s)")
            else:
                print(f"    {emotion}: No files available")
        
        # Test actual fallback retrieval
        print("\n  Testing fallback retrieval:")
        test_cases = [
            ("happy", 2.0),  # Should map to joy
            ("sad", 1.5),    # Should map to sadness  
            ("calm", 3.0),   # Direct match
            ("unknown", 2.0) # Should default to calm
        ]
        
        for emotion, duration in test_cases:
            info = manager.get_fallback_info(emotion, duration)
            print(f"    {emotion} ({duration}m) -> {info['filename']} ({info['fallback_duration']:.1f}m)")
            print(f"      File exists: {info['exists']}")
            
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        print("  Run the following to create fallback files:")
        print("  python3 scripts/utilities/create_fallback_midi.py")
    
    # Test complete workflow
    print("\n3. Testing complete workflow:")
    user_inputs = [
        "I'm feeling happy, play me upbeat music for 2 minutes",
        "I'm sad, give me melancholic music for 3 minutes",
        "I'm angry, I want intense music for 1 minute"
    ]
    
    for user_input in user_inputs:
        print(f"\n  User: '{user_input}'")
        parsed = parse_text_input(user_input)
        print(f"    Parsed -> {parsed['emotion']} music for {parsed['duration_minutes']}m")
        
        if 'manager' in locals():
            info = manager.get_fallback_info(parsed['emotion'], parsed['duration_minutes'])
            if info['exists']:
                print(f"    ✓ Would use fallback: {info['filename']}")
            else:
                print(f"    ⚠️  No fallback available for {parsed['emotion']}")
    
    print("\n" + "=" * 60)
    print("✓ Fallback system test complete")

if __name__ == "__main__":
    test_fallback_system()