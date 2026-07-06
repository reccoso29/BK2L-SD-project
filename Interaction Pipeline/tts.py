"""
Knightro Text-to-Speech (TTS) module

What this file does in simple terms:
  1. Takes text like "Go Knights!"
  2. Converts it to audio using the Piper voice model (millhouse.onnx)
  3. Plays the audio through the speaker
  4. Plays a short bell sound so the user knows Knightro is done talking
     and they can now respond (like a walkie-talkie beep!)

To test this file directly, run:
  python tts.py
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


# =============================================================================
# SETTINGS — you can change these!
# =============================================================================

# Which voice file to use (must be in the same folder as this file)
VOICE_MODEL_FILENAME = "millhouse.onnx"

# Play a bell sound after Knightro finishes talking?
# True = yes play the bell, False = no bell
PLAY_BELL = True

# Bell sound settings
BELL_FREQUENCY_HZ = 880   # how high-pitched the bell is (880 = musical note A5)
BELL_DURATION_SEC = 0.8   # how long the bell plays in seconds
BELL_VOLUME = 1.0         # how loud (0.0 = silent, 1.0 = full volume)

# =============================================================================
# INTERNAL STUFF — you don't need to change anything below this line!
# =============================================================================

# Figure out the full path to the voice model file
# This assumes the .onnx file is in the same folder as tts.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL_PATH = os.path.join(_SCRIPT_DIR, VOICE_MODEL_FILENAME)

# Temporary files for audio (they get deleted after playing)
_TEMP_DIR = tempfile.gettempdir()
_AUDIO_PATH = os.path.join(_TEMP_DIR, "knightro_speech.wav")
_BELL_PATH  = os.path.join(_TEMP_DIR, "knightro_bell.wav")

# We store the loaded voice here so we don't have to reload it every time
# (loading takes a couple seconds, so we only do it once)
_voice_cache = {}


def _generate_bell_wav(filepath):
    """
    Creates a simple 'ding' bell sound and saves it as a WAV file.
    No extra libraries needed — we make the sound using pure math!

    How sound works in simple terms:
    - Sound is just air vibrating really fast
    - We create a mathematical wave (sine wave) that matches how fast we want it to vibrate
    - The volume fades out so it sounds like a real bell, not an annoying beep
    """
    sample_rate = 22050  # how many audio samples per second (standard quality)
    num_samples = int(sample_rate * BELL_DURATION_SEC)

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)       # mono sound (1 channel)
        wav_file.setsampwidth(2)       # 16-bit audio quality
        wav_file.setframerate(sample_rate)

        frames = []
        for i in range(num_samples):
            t = i / sample_rate  # current time in seconds

            # Create the actual sound wave
            sine_value = math.sin(2 * math.pi * BELL_FREQUENCY_HZ * t)

            # Make it fade out (so it sounds like a bell, not a beep)
            fade = 1.0 - (t / BELL_DURATION_SEC)

            # Combine everything
            sample_value = sine_value * fade * BELL_VOLUME

            # Convert to the format WAV files need (16-bit integer)
            int_value = int(sample_value * 32767)
            int_value = max(-32767, min(32767, int_value))
            frames.append(struct.pack("<h", int_value))

        wav_file.writeframes(b"".join(frames))


def _play_bell():
    """Generates and plays the bell sound after Knightro finishes talking."""
    if not PLAY_BELL:
        return
    try:
        _generate_bell_wav(_BELL_PATH)
        time.sleep(0.8)
        _play_wav(_BELL_PATH)
        print("[tts] *ding* — your turn to talk!")
    except Exception as e:
        print(f"[tts] Bell sound failed (not a big deal): {e}")
    finally:
        try:
            os.remove(_BELL_PATH)
        except OSError:
            pass


def _get_voice():
    """
    Loads the Piper voice model from the .onnx file.
    We only load it once and save it in memory for reuse.
    """
    if VOICE_MODEL_PATH not in _voice_cache:
        # Check if piper is installed
        try:
            from piper.voice import PiperVoice
        except ImportError:
            print("[tts] ERROR: piper-tts is not installed!")
            print("[tts] Run this to fix it: pip install piper-tts")
            return None

        # Check if the voice file actually exists
        if not os.path.exists(VOICE_MODEL_PATH):
            print(f"[tts] ERROR: Can't find voice file at: {VOICE_MODEL_PATH}")
            print(f"[tts] Make sure '{VOICE_MODEL_FILENAME}' is in the same folder as tts.py")
            return None

        # Load the voice model
        try:
            print(f"[tts] Loading voice: {VOICE_MODEL_FILENAME}")
            voice = PiperVoice.load(VOICE_MODEL_PATH)
            _voice_cache[VOICE_MODEL_PATH] = voice
            print("[tts] Voice loaded successfully!")
        except Exception as e:
            print(f"[tts] Failed to load voice: {e}")
            return None

    return _voice_cache[VOICE_MODEL_PATH]


def _play_wav(filepath):
    """
    Plays a WAV audio file through the speakers.
    Tries a few different methods in case one doesn't work.
    """
    # Try sounddevice first (works great on Mac, Linux, and Pi)
    try:
        import sounddevice as sd
        import soundfile as sf
        data, samplerate = sf.read(filepath)
        sd.play(data, samplerate)
        sd.wait()
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"[tts] sounddevice failed: {e}, trying backup method...")

    # Try afplay (built into every Mac, no install needed)
    if sys.platform == "darwin":
        try:
            subprocess.run(["afplay", filepath], check=True)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # Try aplay (built into Linux and Raspberry Pi)
    if sys.platform == "linux":
        try:
            subprocess.run(["aplay", filepath], check=True)
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    print("[tts] Could not play audio! Try installing: pip install sounddevice soundfile")


def speak(text, bell=True):
    """
    Makes Knightro say something out loud!

    How it works:
    1. Load the voice model (millhouse.onnx)
    2. Convert the text into audio
    3. Save the audio as a temporary file
    4. Play the audio file through the speakers
    5. Play the bell sound so user knows they can talk
    6. Delete the temporary file

    Example:
        speak("Go Knights! Charge On!")
    """
    if not text or not text.strip():
        return

    print(f'[tts] Knightro says: "{text}"')

    voice = _get_voice()
    if voice is None:
        print(f"[tts] No voice available — would have said: {text}")
        return

    try:
        # Collect ALL audio chunks first, then write them all at once
        # This prevents glitchy sound from writing pieces one at a time
        all_audio = b"".join(
            chunk.audio_int16_bytes for chunk in voice.synthesize(text)
        )

        with wave.open(_AUDIO_PATH, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(voice.config.sample_rate)
            wav_file.writeframes(all_audio)

        # Play the speech audio
        _play_wav(_AUDIO_PATH)

    except Exception as e:
        print(f"[tts] Speech failed: {e}")
        print(f"[tts] Would have said: {text}")
    finally:
        # Always clean up the temp file
        try:
            os.remove(_AUDIO_PATH)
        except OSError:
            pass

    # Only play bell if we want it!
    if bell:
        time.sleep(0.8)
        _play_bell()


def speak_async(text):
    """
    Same as speak() but runs in the background.
    This lets Knightro talk AND do animations at the same time!

    Example:
        t = speak_async("Go Knights!")
        # do animation here while Knightro is talking
        t.join()  # wait for speech to finish
    """
    thread = threading.Thread(target=speak, args=(text,), daemon=True)
    thread.start()
    return thread


# =============================================================================
# TEST — runs when you do "python tts.py" directly
# =============================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Knightro TTS Test")
    print(f"  Voice: {VOICE_MODEL_FILENAME}")
    print(f"  Bell:  {'ON' if PLAY_BELL else 'OFF'}")
    print("=" * 55)
    print()

    print("Test 1 — Greeting:")
    speak("Hey there! Welcome to UCF! Let me see who I am talking to!")
    print()

    print("Test 2 — Chant:")
    speak("UCF Knights! Charge On!")
    print()

    print("Test 3 — Directions:")
    speak("The library is just down the main walkway. You cannot miss it!")
    print()

    print("All done! Did you hear Knightro speak followed by a bell each time?")