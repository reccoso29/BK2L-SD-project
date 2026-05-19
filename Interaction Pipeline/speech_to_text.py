"""Offline speech-to-text using local Whisper models.

Default behavior:
- Captures microphone audio with `speech_recognition`
- Transcribes locally with `openai-whisper`
- Returns structured payload used by the interaction pipeline

Helpful env vars:
- KNIGHTRO_STT_MODE=whisper|typed   (default: whisper)
- KNIGHTRO_WHISPER_MODEL=tiny|base|small|medium|large (default: base)
- KNIGHTRO_WHISPER_MODEL_PATH=/absolute/path/to/local/model (optional)

The optional typed mode is kept for laptop demos when audio dependencies are
not installed yet.
"""

from __future__ import annotations

import io
import os
import wave


_MODEL_CACHE = {}


def get_user_utterance(prompt: str = "You: ") -> dict:
    """Typed-input fallback with the same return contract as Whisper STT."""
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


def _load_whisper_model():
    """Load and cache a local Whisper model instance."""
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("missing_whisper_dependency") from exc

    model_path = os.getenv("KNIGHTRO_WHISPER_MODEL_PATH", "").strip()
    model_name = os.getenv("KNIGHTRO_WHISPER_MODEL", "base").strip() or "base"
    cache_key = model_path or model_name

    if cache_key not in _MODEL_CACHE:
        # When model_path is provided, Whisper loads from local files only.
        _MODEL_CACHE[cache_key] = whisper.load_model(model_path or model_name)

    return _MODEL_CACHE[cache_key]


def _wav_bytes_to_float32_mono(wav_bytes: bytes):
    """Convert WAV bytes to mono float32 samples in range [-1.0, 1.0]."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("missing_numpy_dependency") from exc

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError("unsupported_sample_width")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


def speech_to_text(timeout: float = 7.0, phrase_time_limit: float = 10.0) -> dict:
    """Capture microphone input and transcribe with local Whisper.

    Returns:
        {
            "text": str,
            "error": bool,
            "reason": str,
            "source": str
        }
    """
    mode = os.getenv("KNIGHTRO_STT_MODE", "whisper").strip().lower() or "whisper"
    if mode == "typed":
        return get_user_utterance()

    try:
        import speech_recognition as sr
    except ImportError:
        print("[speech_to_text] Missing speech_recognition dependency.")
        return {
            "text": "",
            "error": True,
            "reason": "missing_speech_recognition_dependency",
            "source": "whisper_offline",
        }

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("[speech_to_text] Listening (offline Whisper)...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except sr.WaitTimeoutError:
        return {"text": "", "error": True, "reason": "timeout", "source": "whisper_offline"}
    except OSError as exc:
        print(f"[speech_to_text] Microphone unavailable: {exc}")
        return {"text": "", "error": True, "reason": "microphone_unavailable", "source": "whisper_offline"}

    try:
        model = _load_whisper_model()
        wav_bytes = audio.get_wav_data(convert_rate=16000, convert_width=2)
        audio_array = _wav_bytes_to_float32_mono(wav_bytes)

        result = model.transcribe(audio_array, fp16=False, task="transcribe")
        text = (result.get("text") or "").strip().lower()

        if not text:
            return {"text": "", "error": True, "reason": "unintelligible", "source": "whisper_offline"}

        print(f"[speech_to_text] Heard: '{text}'")
        return {"text": text, "error": False, "reason": "ok", "source": "whisper_offline"}

    except RuntimeError as exc:
        return {"text": "", "error": True, "reason": str(exc), "source": "whisper_offline"}
    except Exception as exc:
        print(f"[speech_to_text] Whisper transcription failed: {exc}")
        return {"text": "", "error": True, "reason": "transcription_failed", "source": "whisper_offline"}


if __name__ == "__main__":
    print(speech_to_text())
