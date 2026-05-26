"""Offline text-to-speech module for Knightro using Piper TTS.

Piper is a fast, local neural TTS engine — no internet needed.
It runs on macOS, Linux, and Raspberry Pi.

First-time setup:
    pip install piper-tts sounddevice soundfile
    python3 -m piper.download_voices en_US-lessac-medium

Usage:
    from tts import speak
    speak("Go Knights! Charge On!")
"""

import os
import tempfile
import threading
import wave

VOICE_MODEL = "en_US-ryan-high"

# This finds the model relative to wherever tts.py is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "en_US-ryan-high.onnx")
LENGTH_SCALE = float(os.getenv("KNIGHTRO_TTS_SPEED", "0.9"))

_TEMP_DIR = tempfile.gettempdir()
_AUDIO_PATH = os.path.join(_TEMP_DIR, "knightro_speech.wav")
_voice_cache = {}


def _get_voice():
    """Load and cache the Piper voice model."""
    cache_key = VOICE_MODEL_PATH or VOICE_MODEL

    if cache_key not in _voice_cache:
        try:
            from piper.voice import PiperVoice
        except ImportError:
            print("[tts] ERROR: piper-tts not installed!")
            print("[tts] Run: pip install piper-tts")
            return None

        try:
            if VOICE_MODEL_PATH and os.path.exists(VOICE_MODEL_PATH):
                print(f"[tts] Loading voice from: {VOICE_MODEL_PATH}")
                voice = PiperVoice.load(VOICE_MODEL_PATH)
            else:
                model_path = _find_model_path(VOICE_MODEL)
                if model_path:
                    print(f"[tts] Loading voice: {VOICE_MODEL}")
                    voice = PiperVoice.load(model_path)
                else:
                    print(f"[tts] Voice model not found: {VOICE_MODEL}")
                    print(f"[tts] Download it with: python3 -m piper.download_voices {VOICE_MODEL}")
                    return None

            _voice_cache[cache_key] = voice
            print("[tts] Voice loaded successfully!")

        except Exception as e:
            print(f"[tts] Failed to load voice model: {e}")
            return None

    return _voice_cache[cache_key]


def _find_model_path(model_name: str) -> str | None:
    """Search common locations for the Piper voice model .onnx file."""
    search_dirs = [
        os.path.expanduser("~/.local/share/piper_voices"),
        os.path.expanduser("~/piper_voices"),
        os.path.join(os.getcwd(), "models", "tts"),
        os.path.join(os.getcwd(), "piper_voices"),
        os.getcwd(),
    ]

    onnx_filename = f"{model_name}.onnx"

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        direct = os.path.join(search_dir, onnx_filename)
        if os.path.exists(direct):
            return direct

        sub = os.path.join(search_dir, model_name, onnx_filename)
        if os.path.exists(sub):
            return sub

        for root, _, files in os.walk(search_dir):
            if onnx_filename in files:
                return os.path.join(root, onnx_filename)

    return None


def _play_wav(filepath: str) -> None:
    """Play a WAV file. Tries sounddevice first, then falls back to other methods."""
    # Method 1: sounddevice + soundfile (cleanest install on macOS)
    try:
        import sounddevice as sd
        import soundfile as sf

        data, samplerate = sf.read(filepath)
        sd.play(data, samplerate)
        sd.wait()
        return
    except ImportError:
        pass

    # Method 2: pygame (if they installed it)
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        return
    except ImportError:
        pass

    # Method 3: macOS built-in afplay command (no pip install needed)
    import subprocess
    import sys
    if sys.platform == "darwin":
        try:
            subprocess.run(["afplay", filepath], check=True)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Method 4: aplay on Linux / Raspberry Pi
    if sys.platform == "linux":
        try:
            subprocess.run(["aplay", filepath], check=True)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    print("[tts] No audio player available!")
    print("[tts] Install one: pip install sounddevice soundfile")


def speak(text: str) -> None:
    """Convert text to speech and play it. Fully offline.

    Args:
        text: What Knightro should say.
    """
    if not text or not text.strip():
        return

    print(f'[tts] Speaking: "{text}"')

    voice = _get_voice()
    if voice is None:
        print(f"[tts] (no voice available) Would say: {text}")
        return

    try:
        with wave.open(_AUDIO_PATH, "wb") as wav_file:
            # Set WAV parameters: 1 channel (mono), 2 bytes per sample (16-bit),
            # sample rate from the voice model (usually 22050 Hz)
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)

            # Synthesize audio — Piper returns chunks, write each one
            for chunk in voice.synthesize(text):
                wav_file.writeframes(chunk.audio_int16_bytes)

        _play_wav(_AUDIO_PATH)

        try:
            os.remove(_AUDIO_PATH)
        except OSError:
            pass

    except Exception as e:
        print(f"[tts] Speech generation failed: {e}")
        print(f"[tts] Would have said: {text}")


def speak_async(text: str) -> threading.Thread:
    """Speak in a background thread (non-blocking).

    Usage:
        t = speak_async("Go Knights!")
        # ... play animation at the same time ...
        t.join()  # wait for speech to finish
    """
    thread = threading.Thread(target=speak, args=(text,), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    print("=" * 50)
    print("  Knightro TTS Test (Piper — Offline)")
    print("=" * 50)
    print()

    speak("Hey there! Welcome to UCF! I'm Knightro, your favorite knight!")
    print()
    speak("Go Knights! Charge On!")
    print()
    speak("The John C. Hitt Library is just down the main walkway. You can't miss it!")
    print()
    print("Done! If you heard audio, Piper is working.")
    print(f"Current voice: {VOICE_MODEL}")
    print(f"Speed scale: {LENGTH_SCALE} (lower = faster)")