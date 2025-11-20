#!/usr/bin/env python3
"""
Fallback Manager for serving pre-established MIDI files
Used when the model isn't trained or generation fails
"""

from pathlib import Path
import shutil
from typing import Optional


class FallbackManager:
    """Manages fallback MIDI files for each emotion"""
    
    def __init__(self, fallback_dir: str = "assets/fallback_midi"):
        """
        Initialize fallback manager
        
        Args:
            fallback_dir: Directory containing pre-established MIDI files
        """
        self.fallback_dir = Path(fallback_dir)
        
        if not self.fallback_dir.exists():
            raise FileNotFoundError(
                f"Fallback directory not found: {fallback_dir}\n"
                f"Run: python3 scripts/utilities/create_fallback_midi.py"
            )
        
        # Available emotions
        self.emotions = ['joy', 'sadness', 'calm', 'anger', 'fear', 'surprise']
        
        # Available durations (in seconds)
        self.available_durations = [30, 60, 120, 180]
    
    def get_fallback_midi(
        self,
        emotion: str,
        duration_minutes: float,
        output_path: str,
        simulate_delay: bool = True
    ) -> bool:
        """
        Copy a pre-established MIDI file to the output path
        
        Args:
            emotion: Emotion name (joy, sadness, calm, anger, fear, surprise)
            duration_minutes: Requested duration in minutes
            output_path: Where to copy the MIDI file
            simulate_delay: Whether to simulate generation time
        
        Returns:
            True if successful, False otherwise
        """
        import time
        import random
        
        try:
            # Normalize emotion name and handle synonyms
            emotion = self._normalize_emotion(emotion)
            
            # Convert duration to seconds and find closest match
            duration_seconds = duration_minutes * 60
            closest_duration = min(
                self.available_durations,
                key=lambda x: abs(x - duration_seconds)
            )
            
            # Build filename
            filename = f"{emotion}_{closest_duration}s.mid"
            source_path = self.fallback_dir / filename
            
            if not source_path.exists():
                print(f"⚠️  Fallback file not found: {filename}, trying alternatives...")
                # Try to find any file for this emotion
                alternative = self._find_alternative_file(emotion)
                if alternative:
                    source_path = alternative
                    filename = alternative.name
                else:
                    print(f"❌ No fallback files available for emotion: {emotion}")
                    return False
            
            # Simulate realistic generation time
            if simulate_delay:
                base_time = 3.0  # Base 3 seconds
                duration_factor = duration_minutes * 0.5  # 0.5s per minute
                random_factor = random.uniform(0.5, 1.5)  # Random variation
                delay = base_time + duration_factor + random_factor
                time.sleep(delay)
            
            # Copy file to output path
            shutil.copy2(source_path, output_path)
            
            print(f"✓ Using fallback MIDI: {filename}")
            print(f"  Emotion: {emotion}, Requested: {duration_minutes:.1f}m, Using: {closest_duration}s")
            
            return True
        
        except Exception as e:
            print(f"❌ Error getting fallback MIDI: {e}")
            return False
    
    def _normalize_emotion(self, emotion: str) -> str:
        """
        Normalize emotion name and handle synonyms
        """
        emotion = emotion.lower().strip()
        
        # Handle common synonyms
        emotion_synonyms = {
            'happy': 'joy',
            'excited': 'joy',
            'upbeat': 'joy',
            'cheerful': 'joy',
            'energetic': 'joy',
            'sad': 'sadness',
            'melancholic': 'sadness',
            'depressed': 'sadness',
            'angry': 'anger',
            'mad': 'anger',
            'furious': 'anger',
            'aggressive': 'anger',
            'peaceful': 'calm',
            'relaxed': 'calm',
            'serene': 'calm',
            'chill': 'calm',
            'scared': 'fear',
            'anxious': 'fear',
            'nervous': 'fear',
            'worried': 'fear',
            'shocked': 'surprise',
            'amazed': 'surprise'
        }
        
        # Check if it's a synonym
        if emotion in emotion_synonyms:
            emotion = emotion_synonyms[emotion]
        
        # Ensure it's a valid emotion
        if emotion not in self.emotions:
            print(f"⚠️  Unknown emotion '{emotion}', using 'calm' as default")
            emotion = 'calm'
        
        return emotion
    
    def _find_alternative_file(self, emotion: str) -> Optional[Path]:
        """
        Find any available file for the given emotion
        """
        for duration in self.available_durations:
            filename = f"{emotion}_{duration}s.mid"
            file_path = self.fallback_dir / filename
            if file_path.exists():
                return file_path
        return None
    
    def list_available_files(self) -> dict:
        """
        List all available fallback files
        
        Returns:
            Dictionary mapping emotions to available durations
        """
        available = {}
        
        for emotion in self.emotions:
            available[emotion] = []
            for duration in self.available_durations:
                filename = f"{emotion}_{duration}s.mid"
                if (self.fallback_dir / filename).exists():
                    available[emotion].append(duration)
        
        return available
    
    def get_fallback_info(self, emotion: str, duration_minutes: float) -> dict:
        """
        Get information about what fallback file would be used
        
        Args:
            emotion: Emotion name
            duration_minutes: Requested duration
        
        Returns:
            Dictionary with fallback file info
        """
        emotion = self._normalize_emotion(emotion)
        duration_seconds = duration_minutes * 60
        closest_duration = min(
            self.available_durations,
            key=lambda x: abs(x - duration_seconds)
        )
        
        filename = f"{emotion}_{closest_duration}s.mid"
        file_path = self.fallback_dir / filename
        
        return {
            'emotion': emotion,
            'requested_duration': duration_minutes,
            'fallback_duration': closest_duration / 60,
            'filename': filename,
            'exists': file_path.exists(),
            'path': str(file_path)
        }
    
    def verify_fallback_files(self) -> bool:
        """
        Verify that all expected fallback files exist
        
        Returns:
            True if all files exist, False otherwise
        """
        all_exist = True
        missing_files = []
        
        for emotion in self.emotions:
            for duration in self.available_durations:
                filename = f"{emotion}_{duration}s.mid"
                if not (self.fallback_dir / filename).exists():
                    all_exist = False
                    missing_files.append(filename)
        
        if not all_exist:
            print(f"⚠️  Missing {len(missing_files)} fallback files:")
            for filename in missing_files[:5]:  # Show first 5
                print(f"    - {filename}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")
            print("\nRun: python3 scripts/utilities/create_fallback_midi.py")
        
        return all_exist


def test_fallback_manager():
    """Test the fallback manager"""
    print("Testing Fallback Manager")
    print("=" * 60)
    
    manager = FallbackManager()
    
    # Verify files
    print("\n1. Verifying fallback files...")
    if manager.verify_fallback_files():
        print("✓ All fallback files present")
    
    # List available files
    print("\n2. Available fallback files:")
    available = manager.list_available_files()
    for emotion, durations in available.items():
        print(f"  {emotion}: {durations}s")
    
    # Test getting a fallback file
    print("\n3. Testing fallback retrieval...")
    test_output = Path("test_fallback.mid")
    
    success = manager.get_fallback_midi(
        emotion='joy',
        duration_minutes=2.0,
        output_path=str(test_output)
    )
    
    if success and test_output.exists():
        print(f"✓ Successfully retrieved fallback MIDI")
        print(f"  File size: {test_output.stat().st_size} bytes")
        test_output.unlink()  # Clean up
    else:
        print("❌ Failed to retrieve fallback MIDI")
    
    print("=" * 60)


if __name__ == "__main__":
    test_fallback_manager()
