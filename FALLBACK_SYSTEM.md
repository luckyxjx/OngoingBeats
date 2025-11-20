# Fallback System Documentation

## Overview

The fallback system provides pre-established MIDI files when the AI model isn't trained or generation fails. This ensures users always get music output regardless of model status.

## How It Works

### 1. Automatic Detection
- When users input text like "I want happy music for 2 minutes"
- The system parses emotion and duration automatically
- If the model isn't trained, it uses fallback MIDI files

### 2. Emotion Mapping
The system handles emotion synonyms automatically:

```
User Input -> Detected Emotion -> Fallback File
"happy music" -> joy -> joy_120s.mid
"sad music" -> sadness -> sadness_120s.mid  
"angry music" -> anger -> anger_60s.mid
"calm music" -> calm -> calm_180s.mid
"scary music" -> fear -> fear_120s.mid
"weird music" -> surprise -> surprise_60s.mid
```

### 3. Duration Matching
Available durations: 30s, 60s, 120s, 180s
- System picks closest match to user request
- "2 minutes" -> uses 120s file
- "1 minute" -> uses 60s file
- "3 minutes" -> uses 180s file

## Available Files

Current fallback files in `assets/fallback_midi/`:

```
joy_30s.mid, joy_60s.mid, joy_120s.mid, joy_180s.mid
sadness_30s.mid, sadness_60s.mid, sadness_120s.mid, sadness_180s.mid
anger_30s.mid, anger_60s.mid, anger_120s.mid, anger_180s.mid
calm_30s.mid, calm_60s.mid, calm_120s.mid, calm_180s.mid
fear_30s.mid, fear_60s.mid, fear_120s.mid, fear_180s.mid
surprise_30s.mid, surprise_60s.mid, surprise_120s.mid, surprise_180s.mid
```

## Usage Examples

### Text Input (Automatic)
```python
# User types: "I want happy upbeat music for 2 minutes"
# System automatically:
# 1. Detects emotion: "happy" -> "joy"
# 2. Detects duration: "2 minutes" -> 2.0
# 3. Uses fallback: joy_120s.mid (closest to 2 minutes)
```

### API Calls
```python
# Force fallback mode
response = requests.post('/api/generate', {
    "text": "happy music for 2 minutes",
    "use_demo": True  # Forces fallback
})

# Direct emotion selection
response = requests.post('/api/generate-emotion', {
    "emotion": "joy",
    "duration": 2.0,
    "use_demo": True
})
```

## Testing

### Test Fallback System
```bash
python3 test_fallback_system.py
```

### Test API Integration
```bash
# Start API server first
python3 api.py

# In another terminal
python3 test_api_fallback.py
```

## Emotion Keywords

The system recognizes these keywords:

**Joy/Happy**: happy, joyful, excited, cheerful, upbeat, energetic, positive, fun, bright, lively

**Sadness**: sad, melancholic, depressed, down, blue, unhappy, sorrowful, slow, dark, moody

**Anger**: angry, furious, mad, aggressive, intense, fierce, hard, heavy, powerful

**Calm**: calm, peaceful, relaxed, serene, tranquil, chill, mellow, soft, gentle, quiet

**Fear**: scared, fearful, anxious, nervous, worried, tense, mysterious, spooky, eerie

**Surprise**: surprised, shocked, amazed, unexpected, weird, strange, unusual

## Benefits

1. **Always Works**: Users get music even when AI model fails
2. **No Training Required**: Works immediately without model training
3. **Fast Response**: Pre-made files load instantly
4. **Predictable Quality**: Curated MIDI files ensure good output
5. **Handles Synonyms**: Understands natural language variations

## Implementation Details

- **FallbackManager**: Handles file selection and copying
- **TextParser**: Extracts emotion and duration from user input  
- **API Integration**: Automatically falls back when model unavailable
- **Error Handling**: Graceful degradation when files missing

The fallback system ensures your music generation app works reliably regardless of model training status.