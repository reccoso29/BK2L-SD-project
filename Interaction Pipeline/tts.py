"""
Knightro Text-to-Speech (TTS) module

What this file does:
  1. Converts text to audio using Piper TTS (millhouse.onnx voice)
  2. Plays the audio through the speaker
  3. Plays a short bell sound so the user knows Knightro is done talking

To test this file directly:
    python3 tts.py
"""

import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave


# ── Settings ──────────────────────────────────────────────────────────────────
VOICE_MODEL_FILENAME = "en_US-norman-medium.onnx"
PLAY_BELL            = True
BELL_FREQUENCY_HZ    = 880
BELL_DURATION_SEC    = 0.4
BELL_VOLUME          = 0.5

# ── File paths ────────────────────────────────────────────────────────────────
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL_PATH = os.path.join(_SCRIPT_DIR, VOICE_MODEL_FILENAME)

_TEMP_DIR   = tempfile.gettempdir()
_AUDIO_PATH = os.path.join(_TEMP_DIR, "knightro_speech.wav")
_BELL_PATH  = os.path.join(_TEMP_DIR, "knightro_bell.wav")

_voice_cache = {}


# ── Bell sound ────────────────────────────────────────────────────────────────

def _generate_bell_wav(filepath: str) -> None:
    sample_rate = 22050
    num_samples = int(sample_rate * BELL_DURATION_SEC)

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = []
        for i in range(num_samples):
            t            = i / sample_rate
            sine_value   = math.sin(2 * math.pi * BELL_FREQUENCY_HZ * t)
            fade         = 1.0 - (t / BELL_DURATION_SEC)
            sample_value = sine_value * fade * BELL_VOLUME
            int_value    = int(sample_value * 32767)
            int_value    = max(-32767, min(32767, int_value))
            frames.append(struct.pack("<h", int_value))

        wav_file.writeframes(b"".join(frames))


def _play_bell() -> None:
    if not PLAY_BELL:
        return
    try:
        _generate_bell_wav(_BELL_PATH)
        _play_wav(_BELL_PATH)
        print("[tts] *ding* — your turn to talk!")
    except Exception as e:
        print(f"[tts] Bell sound failed: {e}")
    finally:
        try:
            os.remove(_BELL_PATH)
        except OSError:
            pass


# ── Voice model ───────────────────────────────────────────────────────────────

def _get_voice():
    """Loads and caches the Piper voice model."""
    if VOICE_MODEL_PATH not in _voice_cache:
        try:
            from piper.voice import PiperVoice
        except ImportError:
            print("[tts] ERROR: piper-tts not installed! Run: pip install piper-tts")
            return None

        if not os.path.exists(VOICE_MODEL_PATH):
            print(f"[tts] ERROR: Voice model not found: {VOICE_MODEL_PATH}")
            print(f"[tts] Make sure '{VOICE_MODEL_FILENAME}' is in the Interaction Pipeline folder!")
            return None

        try:
            print(f"[tts] Loading voice: {VOICE_MODEL_FILENAME}")
            voice = PiperVoice.load(VOICE_MODEL_PATH)
            _voice_cache[VOICE_MODEL_PATH] = voice
            print("[tts] Voice loaded!")
        except Exception as e:
            print(f"[tts] Failed to load voice: {e}")
            return None

    return _voice_cache[VOICE_MODEL_PATH]


# ── Audio playback ────────────────────────────────────────────────────────────

def _play_wav(filepath: str) -> None:
    """Plays a WAV file using aplay."""
    if sys.platform == "linux":
        try:
            subprocess.run(["aplay", filepath], check=True)
            return
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"[tts] aplay failed: {e}")

    if sys.platform == "darwin":
        try:
            subprocess.run(["afplay", filepath], check=True)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    print("[tts] No audio player found!")

# ── Main public functions ─────────────────────────────────────────────────────

def speak(text: str, bell: bool = True) -> None:
    """
    Makes Knightro say something out loud!

    Args:
        text: What Knightro should say
        bell: Whether to ring the bell after speaking (default True)
              Use bell=False when about to scan a face or wait for yes/no
    """
    if not text or not text.strip():
        return

    print(f'[tts] Knightro says: "{text}"')

    voice = _get_voice()
    if voice is None:
        print(f"[tts] No voice available — would have said: {text}")
        return

    try:
        # Collect ALL audio chunks first then write at once
        # This prevents glitchy sound from writing pieces one at a time
        all_audio = b"".join(
            chunk.audio_int16_bytes for chunk in voice.synthesize(text)
        )

        with wave.open(_AUDIO_PATH, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            wav_file.writeframes(all_audio)

        _play_wav(_AUDIO_PATH)

    except Exception as e:
        print(f"[tts] Speech failed: {e}")
        print(f"[tts] Would have said: {text}")
    finally:
        try:
            os.remove(_AUDIO_PATH)
        except OSError:
            pass

    if bell:
        time.sleep(0.8)
        _play_bell()


def speak_async(text: str, bell: bool = True) -> threading.Thread:
    """Speaks in a background thread so other things can happen simultaneously."""
    thread = threading.Thread(
        target=speak,
        args=(text,),
        kwargs={"bell": bell},
        daemon=True
    )
    thread.start()
    return thread


def preload_voice() -> None:
    """
    Pre-loads the voice model into memory at startup.
    Call this once when the program starts so the first
    speak() call is instant instead of slow!
    Think of it like warming up a car before driving!
    """
    print("[tts] Pre-loading voice model...")
    voice = _get_voice()
    if voice:
        print("[tts] Voice pre-loaded and ready!")
    else:
        print("[tts] Voice pre-load failed — will try again when speaking")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Knightro TTS Test")
    print(f"  Voice: {VOICE_MODEL_FILENAME}")
    print(f"  Bell:  {'ON' if PLAY_BELL else 'OFF'}")
    print("=" * 55)
    print()

    preload_voice()
    print()

    speak("Hey there! Welcome to UCF! I am Knightro!")
    print()
    speak("U! C! F! Knights! Charge On!")
    print()
    print("Done!")