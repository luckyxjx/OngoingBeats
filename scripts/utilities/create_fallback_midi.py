#!/usr/bin/env python3
"""
Create high-quality fallback MIDI files for each emotion
These are pre-established files that will be used when the model isn't trained
"""

import pretty_midi
from pathlib import Path
import numpy as np


def create_joy_midi(duration: float, output_path: str):
    """Create upbeat, happy music"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    
    # Major scale (C major)
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    beat = 0.5  # 120 BPM
    
    # Create a cheerful melody pattern
    melody_pattern = [0, 2, 4, 2, 0, 4, 7, 4]
    time = 0.0
    
    while time < duration:
        for i in melody_pattern:
            if time >= duration:
                break
            pitch = scale[i % len(scale)]
            note = pretty_midi.Note(
                velocity=85,
                pitch=pitch,
                start=time,
                end=time + beat * 0.9
            )
            piano.notes.append(note)
            time += beat
    
    # Add bass line
    bass_pattern = [48, 48, 55, 55]
    time = 0.0
    while time < duration:
        for pitch in bass_pattern:
            if time >= duration:
                break
            note = pretty_midi.Note(
                velocity=70,
                pitch=pitch,
                start=time,
                end=time + beat * 2
            )
            piano.notes.append(note)
            time += beat * 2
    
    midi.instruments.append(piano)
    midi.write(output_path)


def create_sadness_midi(duration: float, output_path: str):
    """Create slow, melancholic music"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=60)
    piano = pretty_midi.Instrument(program=0)
    
    # Minor scale (A minor)
    scale = [57, 59, 60, 62, 64, 65, 67, 69]
    beat = 1.0  # 60 BPM
    
    # Slow, descending melody
    melody_pattern = [7, 6, 5, 4, 3, 2, 1, 0]
    time = 0.0
    
    while time < duration:
        for i in melody_pattern:
            if time >= duration:
                break
            pitch = scale[i % len(scale)]
            note = pretty_midi.Note(
                velocity=60,
                pitch=pitch,
                start=time,
                end=time + beat * 1.5
            )
            piano.notes.append(note)
            time += beat * 1.2
    
    # Add sustained chords
    chord_times = np.arange(0, duration, beat * 4)
    for t in chord_times:
        if t >= duration:
            break
        # Minor chord
        for offset in [0, 3, 7]:
            note = pretty_midi.Note(
                velocity=50,
                pitch=57 + offset,
                start=t,
                end=min(t + beat * 3, duration)
            )
            piano.notes.append(note)
    
    midi.instruments.append(piano)
    midi.write(output_path)


def create_calm_midi(duration: float, output_path: str):
    """Create peaceful, relaxing music"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=80)
    piano = pretty_midi.Instrument(program=0)
    
    # Pentatonic scale (peaceful sound)
    scale = [60, 62, 64, 67, 69, 72]
    beat = 0.75  # 80 BPM
    
    # Gentle, flowing melody
    time = 0.0
    while time < duration:
        for i in range(len(scale)):
            if time >= duration:
                break
            pitch = scale[i]
            note = pretty_midi.Note(
                velocity=65,
                pitch=pitch,
                start=time,
                end=time + beat * 1.5
            )
            piano.notes.append(note)
            time += beat * 1.2
    
    # Add soft arpeggios
    arp_pattern = [60, 64, 67, 72]
    time = 0.0
    while time < duration:
        for pitch in arp_pattern:
            if time >= duration:
                break
            note = pretty_midi.Note(
                velocity=55,
                pitch=pitch,
                start=time,
                end=time + beat * 0.5
            )
            piano.notes.append(note)
            time += beat * 0.5
    
    midi.instruments.append(piano)
    midi.write(output_path)


def create_anger_midi(duration: float, output_path: str):
    """Create intense, aggressive music"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=140)
    piano = pretty_midi.Instrument(program=0)
    
    # Chromatic and dissonant intervals
    scale = [60, 61, 63, 65, 66, 68, 70, 72]
    beat = 0.43  # 140 BPM
    
    # Fast, aggressive pattern
    time = 0.0
    while time < duration:
        for i in range(8):
            if time >= duration:
                break
            pitch = scale[i % len(scale)]
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=time,
                end=time + beat * 0.8
            )
            piano.notes.append(note)
            time += beat * 0.5
    
    # Add heavy bass hits
    time = 0.0
    while time < duration:
        note = pretty_midi.Note(
            velocity=110,
            pitch=48,
            start=time,
            end=time + beat * 0.3
        )
        piano.notes.append(note)
        time += beat * 2
    
    midi.instruments.append(piano)
    midi.write(output_path)


def create_fear_midi(duration: float, output_path: str):
    """Create tense, anxious music"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    piano = pretty_midi.Instrument(program=0)
    
    # Diminished scale (creates tension)
    scale = [60, 61, 63, 64, 66, 67, 69, 70]
    beat = 0.6  # 100 BPM
    
    # Irregular, unpredictable pattern
    time = 0.0
    pattern = [0, 3, 1, 4, 2, 5, 1, 6]
    
    while time < duration:
        for i in pattern:
            if time >= duration:
                break
            pitch = scale[i % len(scale)]
            note = pretty_midi.Note(
                velocity=75,
                pitch=pitch,
                start=time,
                end=time + beat * 0.7
            )
            piano.notes.append(note)
            time += beat * np.random.uniform(0.8, 1.5)
    
    # Add tremolo effect
    time = 0.0
    while time < duration:
        for _ in range(4):
            if time >= duration:
                break
            note = pretty_midi.Note(
                velocity=70,
                pitch=55,
                start=time,
                end=time + beat * 0.2
            )
            piano.notes.append(note)
            time += beat * 0.25
        time += beat * 2
    
    midi.instruments.append(piano)
    midi.write(output_path)


def create_surprise_midi(duration: float, output_path: str):
    """Create unexpected, varied music"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=110)
    piano = pretty_midi.Instrument(program=0)
    
    # Whole tone scale (creates ambiguity)
    scale = [60, 62, 64, 66, 68, 70, 72]
    beat = 0.55  # 110 BPM
    
    # Unpredictable jumps
    time = 0.0
    while time < duration:
        for _ in range(8):
            if time >= duration:
                break
            pitch = scale[np.random.randint(0, len(scale))]
            velocity = np.random.randint(70, 95)
            note_duration = beat * np.random.choice([0.5, 1.0, 1.5])
            
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=time,
                end=time + note_duration
            )
            piano.notes.append(note)
            time += note_duration * np.random.uniform(0.5, 1.5)
    
    midi.instruments.append(piano)
    midi.write(output_path)


def create_all_fallback_files():
    """Create fallback MIDI files for all emotions and durations"""
    output_dir = Path("assets/fallback_midi")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    emotions = {
        'joy': create_joy_midi,
        'sadness': create_sadness_midi,
        'calm': create_calm_midi,
        'anger': create_anger_midi,
        'fear': create_fear_midi,
        'surprise': create_surprise_midi
    }
    
    # Create files for different durations (in seconds)
    durations = [30, 60, 120, 180]  # 30s, 1min, 2min, 3min
    
    print("Creating fallback MIDI files...")
    print("=" * 60)
    
    for emotion_name, create_func in emotions.items():
        for duration in durations:
            filename = f"{emotion_name}_{duration}s.mid"
            output_path = output_dir / filename
            create_func(duration, str(output_path))
            print(f"✓ Created: {filename}")
    
    print("=" * 60)
    print(f"✓ Created {len(emotions) * len(durations)} fallback MIDI files")
    print(f"  Location: {output_dir}/")
    print("\nThese files will be used as fallback when the model isn't trained.")


if __name__ == "__main__":
    create_all_fallback_files()
