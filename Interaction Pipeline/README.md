# Knightro Interaction Pipeline Baseline

This folder contains the first clean baseline for Knightro interactions.
It is fully laptop-testable and intentionally uses placeholders for behavior.

## Pipeline Flow

```
User speech/input
  -> speech-to-text layer (offline Whisper)
  -> safety/profanity pre-check
  -> intent detection
  -> interaction routing
  -> offline handler OR online cloud LLM handler
  -> output safety check
  -> print response/action summary
```

## Supported Baseline Intents

- greeting
- known_person
- ucf_trivia
- directions
- knightro_info
- unknown

## How to Run

1. Open a terminal in this folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run:

```bash
python3 main.py
```

4. Speak into your microphone.
5. Say `exit` to quit.

### Offline Whisper Notes

- STT is fully local using Whisper (no cloud transcription).
- To force typed fallback during setup/troubleshooting:

```bash
export KNIGHTRO_STT_MODE=typed
python3 main.py
```

- Optional local model path (pre-downloaded model files):

```bash
export KNIGHTRO_WHISPER_MODEL_PATH="/absolute/path/to/whisper-model"
```

- Optional model choice:

```bash
export KNIGHTRO_WHISPER_MODEL=base
```

## Optional Cloud LLM Setup

The online unknown-intent handler supports OpenAI-compatible APIs via env vars.

Set these before running if you want cloud calls when online:

```bash
export LLM_API_KEY="your_key"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="llama-3.1-8b-instant"
```

If these are not set, the router will automatically fall back to offline unknown behavior.

## Notes

- No hardware-specific robot behavior is implemented in this baseline.
- Offline handlers only print what they would do.
- Safety checks are placeholder keyword filters and are easy to expand.
- Set `KNIGHTRO_FORCE_OFFLINE=true` to force offline mode for testing.
