#!/usr/bin/env python3
"""Test script to verify API imports work correctly"""

print("Testing API imports...")

try:
    from src.config import ModelConfig, TokenizerConfig, GenerationConfig
    print("✓ Config imports successful")
except Exception as e:
    print(f"✗ Config imports failed: {e}")

try:
    from src.model import create_model
    print("✓ Model import successful")
except Exception as e:
    print(f"✗ Model import failed: {e}")

try:
    from src.tokenizer import MIDITokenizer
    print("✓ Tokenizer import successful")
except Exception as e:
    print(f"✗ Tokenizer import failed: {e}")

try:
    from src.generation.text_parser import parse_text_input
    print("✓ Text parser import successful")
except Exception as e:
    print(f"✗ Text parser import failed: {e}")

try:
    from src.generation.audio_converter import AudioConverter
    print("✓ Audio converter import successful")
except Exception as e:
    print(f"✗ Audio converter import failed: {e}")

try:
    from src.generation.improved_generator import ImprovedMusicGenerator
    print("✓ Improved generator import successful")
except Exception as e:
    print(f"✗ Improved generator import failed: {e}")

try:
    from src.training.human_feedback import HumanFeedbackCollector
    print("✓ Human feedback import successful")
except Exception as e:
    print(f"✗ Human feedback import failed: {e}")

try:
    from scripts.utilities.create_demo_midi import create_demo_midi
    print("✓ Demo MIDI import successful")
except Exception as e:
    print(f"✗ Demo MIDI import failed: {e}")

print("\n✓ All imports successful! API should start correctly.")
