# Requirements Document

## Introduction

This document specifies the requirements for an Emotion-Based MIDI Fallback System. The system provides a reliable backup mechanilable or untrained. The fallback integrates directly into the current api.py structure.

## Glossary

- **Fallback System**: A backup mechanism that copies pre-recorded MIDI files when the AI model cannot generate music
- **Fallback Library**: A directory containing pre-established MIDI files organized by emotion
- **API**: The existing Flask server in api.py that handles music generation requests

## Requirements

### Requirement 1

**User Story:** As a user, I want to receive playable music for any emotion I select, so that I can experience the application even when the AI model is not trained.

#### Acceptance Criteria

1. WHEN the API receives a generation request THEN the system SHALL check if the model is trained
2. WHEN the model is not trained THEN the system SHALL use the fallback library instead of attempting generation
3. WHEN using fallback THEN the system SHALL copy a pre-established MIDI file matching the requested emotion
4. WHEN a fallback file is used THEN the system SHALL convert it to MP3 using the existing audio converter
5. WHEN returning a response THEN the system SHALL set the demo_mode flag to indicate fallback was used

### Requirement 2

**User Story:** As a developer, I want a simple directory structure for fallback files, so that I can easily add pre-established MIDI files.

#### Acceptance Criteria

1. WHEN the system initializes THEN the system SHALL create a fallback_library directory with emotion subdirectories
2. WHEN storing fallback files THEN the system SHALL use subdirectories named: joy, sadness, anger, calm, surprise, fear
3. WHEN selecting a fallback file THEN the system SHALL randomly pick one file from the emotion subdirectory
4. WHEN no fallback file exists THEN the system SHALL use the existing demo generation as final fallback
5. WHEN the API starts THEN the system SHALL log the number of available fallback files per emotion

### Requirement 3

**User Story:** As a developer, I want to generate initial fallback MIDI files, so that the system has a working library without manual creation.

#### Acceptance Criteria

1. WHEN running the setup script THEN the system SHALL generate 2 MIDI files per emotion
2. WHEN generating fallback files THEN the system SHALL create files with 60-second and 120-second durations
3. WHEN creating files THEN the system SHALL use simple musical patterns appropriate for each emotion
4. WHEN files are generated THEN the system SHALL save them in the correct emotion subdirectories
5. WHEN the script completes THEN the system SHALL print a summary of created files
