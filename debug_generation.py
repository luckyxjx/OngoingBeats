#!/usr/bin/env python3
"""Debug script to test generation and identify the issue"""

import torch
from pathlib import Path
from src.config import ModelConfig, TokenizerConfig
from src.model import create_model
from src.tokenizer import MIDITokenizer
from src.generation.improved_generator import ImprovedMusicGenerator

print("Initializing...")

# Create tokenizer
tokenizer_config = TokenizerConfig()
tokenizer = MIDITokenizer(tokenizer_config)

# Create model
model_config = ModelConfig(
    model_type="transformer",
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=512,
    use_emotion_conditioning=True,
    emotion_emb_dim=64,
    num_emotions=6,
    use_duration_control=True,
    duration_emb_dim=32
)

model = create_model(model_config, tokenizer.vocab_size)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device)

print(f"Model on {device}")

# Create generator
generator = ImprovedMusicGenerator(
    model=model,
    tokenizer=tokenizer,
    device=device
)

print("\nGenerating...")
try:
    tokens = generator.generate_with_constraints(
        emotion=0,
        duration_minutes=2.0,
        temperature=0.75,
        top_k=60,
        top_p=0.92,
        max_tokens=512,
        min_notes=50,
        max_consecutive_time_shifts=3,
        repetition_penalty=1.3
    )
    print(f"\n✓ Generated {len(tokens)} tokens")
except Exception as e:
    print(f"\n✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\nSaving MIDI...")
try:
    output_path = "debug_output.mid"
    midi = generator.save_midi(tokens, output_path)
    print(f"\n✓ MIDI saved successfully")
    print(f"  File: {output_path}")
    print(f"  Instruments: {len(midi.instruments)}")
    print(f"  Total notes: {sum(len(inst.notes) for inst in midi.instruments)}")
except Exception as e:
    print(f"\n✗ MIDI save failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ All tests passed!")
