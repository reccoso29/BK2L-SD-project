"""speech_to_text.py — Speech recognition for Knightro

Two modes:
1. GROQ WHISPER (default when internet available) — FAST! Under 1 second
2. LOCAL WHISPER (fallback when offline) — Slow on Pi, 5-15 seconds
3. TYPED MODE — reads keyboard input for testing without microphone

The system automatically picks the best option based on internet availability.
"""

from __future__ import annotations

import io
import os
import sys
import wave
import tempfile

# Load API key from .env file
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(_env_path)
except ImportError:
    pass

_MODEL_CACHE = {}


# ── Typed input fallback ──────────────────────────────────────────────────────

def get_user_utterance(prompt: str = "You: ") -> dict:
    """Typed-input fallback — reads from keyboard instead of microphone."""
    try:
        raw = input(prompt)
    except EOFError:
        return {"text": "", "error": True, "reason": "eof", "source": "typed"}
    except KeyboardInterrupt:
        return {"text": "", "error": True, "reason": "interrupt", "source": "typed"}

    normalized = raw.strip()
    if not normalized:
        return {"text": "", "error": True, "reason": "empty", "source": "typed"}

    return {"text": normalized.lower(), "error": False, "reason": "ok", "source": "typed"}


# ── Audio recording ───────────────────────────────────────────────────────────

def _record_audio(timeout: float = 7.0, phrase_time_limit: float = 10.0):
    """
    Records audio from microphone and returns it as a speech_recognition Audio object.
    Returns (audio, None) on success or (None, error_reason) on failure.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "missing_speech_recognition_dependency"

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("[speech_to_text] Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
        return audio, None

    except sr.WaitTimeoutError:
        return None, "timeout"
    except OSError as exc:
        print(f"[speech_to_text] Microphone unavailable: {exc}")
        return None, "microphone_unavailable"


# ── Groq Whisper (fast cloud transcription) ───────────────────────────────────

def _transcribe_with_groq(audio) -> dict:
    """
    Sends recorded audio to Groq Whisper API — under 1 second response!
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"text": "", "error": True, "reason": "no_api_key", "source": "groq_whisper"}

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            print("[speech_to_text] Sending to Groq Whisper (fast!)...")
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file,
                    response_format="text",
                    language="en",
                )

            text = transcription.strip().lower() if transcription else ""

            if not text:
                return {"text": "", "error": True, "reason": "unintelligible", "source": "groq_whisper"}

            print(f"[speech_to_text] Groq heard: '{text}'")
            return {"text": text, "error": False, "reason": "ok", "source": "groq_whisper"}

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    except ImportError:
        return {"text": "", "error": True, "reason": "groq_not_installed", "source": "groq_whisper"}
    except Exception as exc:
        print(f"[speech_to_text] Groq Whisper failed: {exc}")
        return {"text": "", "error": True, "reason": f"groq_failed: {exc}", "source": "groq_whisper"}


# ── Local Whisper (offline fallback) ─────────────────────────────────────────

def _load_whisper_model():
    """Load and cache a local Whisper model — tiny for speed on Pi."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("missing_whisper_dependency") from exc

    model_name = os.getenv("KNIGHTRO_WHISPER_MODEL", "tiny").strip() or "tiny"

    if model_name not in _MODEL_CACHE:
        print(f"[speech_to_text] Loading local Whisper model: {model_name}")
        _MODEL_CACHE[model_name] = whisper.load_model(model_name)
        print("[speech_to_text] Local model loaded!")

    return _MODEL_CACHE[model_name]


def _transcribe_locally(audio) -> dict:
    """Transcribes audio using local Whisper — slower but works offline."""
    try:
        import numpy as np

        model     = _load_whisper_model()
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels     = wav_file.getnchannels()
            frames       = wav_file.readframes(wav_file.getnframes())

        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels).mean(axis=1)

        print("[speech_to_text] Transcribing locally (may take a moment)...")
        result = model.transcribe(audio_array, fp16=False, task="transcribe")
        text   = (result.get("text") or "").strip().lower()

        if not text:
            return {"text": "", "error": True, "reason": "unintelligible", "source": "whisper_offline"}

        print(f"[speech_to_text] Local Whisper heard: '{text}'")
        return {"text": text, "error": False, "reason": "ok", "source": "whisper_offline"}

    except Exception as exc:
        print(f"[speech_to_text] Local transcription failed: {exc}")
        return {"text": "", "error": True, "reason": "transcription_failed", "source": "whisper_offline"}


# ── Main public function ──────────────────────────────────────────────────────

def speech_to_text(timeout: float = 7.0, phrase_time_limit: float = 10.0) -> dict:
    """
    Records microphone audio and transcribes it.
    Automatically uses Groq Whisper if internet available, local Whisper otherwise.
    """
    # Check for typed mode
    mode = os.getenv("KNIGHTRO_STT_MODE", "whisper").strip().lower()
    if mode == "typed":
        return get_user_utterance()

    # Record audio from microphone
    audio, error = _record_audio(timeout=timeout, phrase_time_limit=phrase_time_limit)

    if audio is None:
        return {
            "text": "",
            "error": True,
            "reason": error or "recording_failed",
            "source": "microphone"
        }

    # Try Groq first, fall back to local
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        result = _transcribe_with_groq(audio)
        if not result["error"]:
            return result
        print(f"[speech_to_text] Groq failed ({result['reason']}), falling back to local...")

    return _transcribe_locally(audio)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing speech to text — say something into the microphone!")
    result = speech_to_text()
    print(f"Result: {result}")