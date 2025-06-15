# Smart Subtitle Generator

An AI-powered tool for transcribing audio files to SRT subtitles with speaker detection and translation capabilities.

## Setup

1. Clone this repository
2. [Install uv package manager on your machine](https://docs.astral.sh/uv/getting-started/installation/)
3. Copy `.env.example` to `.env` and add your Google API key

## Usage

```bash
# uv run makes sure that all the dependencies are installed correctly

# Basic usage (transcribes and translates to English by default)
uv run transcribe.py path/to/audio_file.mp3

# Specify a different translation language
uv run transcribe.py path/to/audio_file.mp3 --language Spanish

# Disable translation
uv run transcribe.py path/to/audio_file.mp3 --no-translate
```

## Arguments

- `file`: Path to the audio file to transcribe (required)
- `--language`, `-l`: Target language for translation (default: English)
- `--no-translate`, `-n`: Disable translation (translation is enabled by default)

## Output

The script generates two files:
- `filename.srt`: Original transcription with speaker detection
- `filename_language.srt`: Translated version (if translation is enabled)

## Requirements

- Python 3.10+
- Google API key with access to Gemini models
- Audio file in [a supported format](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/audio-understanding) (MP3, WAV, etc.) 
