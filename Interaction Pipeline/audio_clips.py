"""
audio_clips.py — Knightro Pre-recorded Sound Clips

Only two clips are used now:
    "startup" → ucf_spell.wav   (plays when Knightro first turns on)
    "chant"   → CHANT_T1.mp3   (plays for chant intent — no robot voice)

Everything else uses the robot voice (en_US-bryce-medium)
"""

import os
import threading
import time

try:
    import pygame
    pygame.mixer.init()
    _PYGAME_AVAILABLE = True
    print("[audio_clips] pygame loaded — audio clips ACTIVE")
except ImportError:
    _PYGAME_AVAILABLE = False
    print("[audio_clips] WARNING: pygame not installed — audio clips DISABLED")
    print("[audio_clips] Fix: pip install pygame")


_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_AUDIO_DIR    = os.path.join(_PROJECT_ROOT, "audio")

# Only two clips kept!
CLIP_MAP = {
    "startup": ["ucf_spell.wav"],
    "chant":   ["CHANT_T1.mp3"],
}


def _get_clip_path(clip_name):
    if clip_name not in CLIP_MAP:
        print(f"[audio_clips] Unknown clip: '{clip_name}'")
        print(f"[audio_clips] Available clips: {list(CLIP_MAP.keys())}")
        return None

    filename  = CLIP_MAP[clip_name][0]
    full_path = os.path.join(_AUDIO_DIR, filename)

    if not os.path.exists(full_path):
        print(f"[audio_clips] File not found: {full_path}")
        return None

    return full_path


def play(clip_name):
    """Plays a pre-recorded audio clip and waits for it to finish."""
    if not _PYGAME_AVAILABLE:
        print(f"[audio_clips] Would play: {clip_name} (pygame not available)")
        return

    clip_path = _get_clip_path(clip_name)
    if clip_path is None:
        return

    try:
        print(f"[audio_clips] Playing: {clip_name} ({os.path.basename(clip_path)})")
        pygame.mixer.music.load(clip_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

        print(f"[audio_clips] Done playing: {clip_name}")

    except Exception as e:
        print(f"[audio_clips] Error playing '{clip_name}': {e}")


def play_async(clip_name):
    """Plays a clip in the background without blocking."""
    thread = threading.Thread(target=play, args=(clip_name,), daemon=True)
    thread.start()
    return thread


def stop():
    """Stops whatever clip is currently playing."""
    if not _PYGAME_AVAILABLE:
        return
    try:
        pygame.mixer.music.stop()
        print("[audio_clips] Stopped audio")
    except Exception as e:
        print(f"[audio_clips] Error stopping audio: {e}")


if __name__ == "__main__":
    print("=" * 55)
    print("  Knightro Audio Clips Test")
    print("  Only startup and chant clips are active!")
    print("=" * 55)
    print()

    print("Testing: startup")
    play("startup")
    time.sleep(0.5)

    print("Testing: chant")
    play("chant")

    print("\nDone!")