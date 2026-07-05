"""
audio_clips.py — Knightro Pre-recorded Sound Clips

Think of this file like a jukebox!
Instead of the robot voice saying things, we play real recorded audio clips
for special moments. This makes Knightro sound way more natural and fun!

How to use it from other files:
    import audio_clips
    audio_clips.play("chant")        # plays CHANT_T1.mp3
    audio_clips.play("farewell")     # plays Peace_Out.mp3 or Until_the_next_time.mp3
    audio_clips.play("startup")      # plays ucf_spell.wav

Available clip names:
    "startup"       → ucf_spell.wav         (plays when Knightro first turns on)
    "cant_hear"     → HELMET_T1_new.mp3     (plays when Knightro can't hear you)
    "happy_to_help" → Happy_to_Help.mp3     (plays after answering a question)
    "goknights"     → Go_knights_Charge_On.mp3  (plays for Go Knights intent)
    "touchdown"     → Touch_Down_UCF_.mp3   (plays for UCF football questions)
    "charge_on"     → Charge_on.mp3         (plays when user says Go Knights)
    "farewell"      → randomly picks Peace_Out.mp3 OR Until_the_next_time.mp3
    "nice_to_meet"  → Nice_to_meet_you.mp3  (plays for unknown visitors)
    "chant"         → CHANT_T1.mp3          (plays for chant intent)
"""

import os
import random
import threading
import time

# Try to import pygame — this is our mp3/wav player
# If it's not installed, we just print a message instead of crashing
try:
    import pygame
    pygame.mixer.init()
    _PYGAME_AVAILABLE = True
    print("[audio_clips] pygame loaded — audio clips ACTIVE")
except ImportError:
    _PYGAME_AVAILABLE = False
    print("[audio_clips] WARNING: pygame not installed — audio clips DISABLED")
    print("[audio_clips] Fix: pip install pygame")


# ── Where are the audio files? ────────────────────────────────────────────────
# This finds the "audio" folder in the root of our project
# Our file structure looks like this:
#   BK2L-SD-project/
#     audio/          ← audio clips go here
#     Interaction Pipeline/
#       audio_clips.py  ← this file is here
#
# So we go UP one folder from here, then INTO the audio folder
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_AUDIO_DIR    = os.path.join(_PROJECT_ROOT, "audio")


# ── The jukebox! Which clip name maps to which file? ─────────────────────────
# This is like a menu at the jukebox — you pick a name, it plays the right song!
# "farewell" randomly picks between two files to keep things interesting!
CLIP_MAP = {
    "startup":       ["ucf_spell.wav"],
    "cant_hear":     ["HELMET_T1_new.mp3"],
    "happy_to_help": ["Happy_to_Help.mp3"],
    "goknights":     ["Go_knights_Charge_On.mp3"],
    "touchdown":     ["Touch_Down_UCF_.mp3"],
    "charge_on":     ["Charge_on.mp3"],
    "farewell":      ["Peace_Out.mp3", "Until_the_next_time.mp3"],  # picks randomly!
    "nice_to_meet":  ["Nice_to_meet_you.mp3"],
    "chant":         ["CHANT_T1.mp3"],
}


def _get_clip_path(clip_name):
    """
    Finds the full file path for a clip name.
    If the clip has multiple files (like farewell), picks one randomly!

    Think of it like a vending machine — you press B3 and it gives you
    whatever snack is assigned to B3!
    """
    if clip_name not in CLIP_MAP:
        print(f"[audio_clips] Unknown clip: '{clip_name}'")
        print(f"[audio_clips] Available clips: {list(CLIP_MAP.keys())}")
        return None

    # Pick a random file from the list (most have just one, farewell has two)
    filename = random.choice(CLIP_MAP[clip_name])
    full_path = os.path.join(_AUDIO_DIR, filename)

    # Check if the file actually exists before trying to play it
    if not os.path.exists(full_path):
        print(f"[audio_clips] File not found: {full_path}")
        print(f"[audio_clips] Make sure '{filename}' is in your audio/ folder")
        return None

    return full_path


def play(clip_name):
    """
    Plays a pre-recorded audio clip and WAITS for it to finish.

    This is like pressing play on a song and waiting until it's done
    before doing anything else.

    Args:
        clip_name: The name of the clip to play (see the list at the top!)

    Example:
        audio_clips.play("chant")       # plays the chant sound
        audio_clips.play("farewell")    # plays a farewell sound
    """
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

        # Wait for the clip to finish playing before moving on
        # This is like waiting for a microwave to beep before opening it!
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)  # check every 50ms if it's still playing

        print(f"[audio_clips] Done playing: {clip_name}")

    except Exception as e:
        print(f"[audio_clips] Error playing '{clip_name}': {e}")


def play_async(clip_name):
    """
    Plays a pre-recorded audio clip in the BACKGROUND.

    This is like putting on background music — the clip plays while
    other things can still happen at the same time!

    Use this when you don't want to wait for the clip to finish.

    Args:
        clip_name: The name of the clip to play

    Returns:
        The background thread (you can call .join() to wait for it)

    Example:
        audio_clips.play_async("startup")   # plays in background
        # ... other code runs here while audio plays ...
    """
    thread = threading.Thread(target=play, args=(clip_name,), daemon=True)
    thread.start()
    return thread


def stop():
    """
    Stops whatever audio clip is currently playing.

    Like hitting the stop button on a CD player!
    """
    if not _PYGAME_AVAILABLE:
        return
    try:
        pygame.mixer.music.stop()
        print("[audio_clips] Stopped audio")
    except Exception as e:
        print(f"[audio_clips] Error stopping audio: {e}")


def list_clips():
    """
    Prints all available clip names and their files.
    Helpful for debugging!
    """
    print("\n[audio_clips] Available clips:")
    for name, files in CLIP_MAP.items():
        for f in files:
            full_path = os.path.join(_AUDIO_DIR, f)
            exists = "✓" if os.path.exists(full_path) else "✗ NOT FOUND"
            print(f"  '{name}' → {f} {exists}")
    print()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Knightro Audio Clips Test")
    print("=" * 55)
    print()

    # Show all available clips and check if files exist
    list_clips()

    # Test each clip one by one
    test_clips = ["startup", "nice_to_meet", "chant", "goknights", "farewell"]

    for clip in test_clips:
        print(f"Testing: {clip}")
        play(clip)
        time.sleep(0.5)  # small pause between clips

    print("\nAll done! Did you hear all the clips?")
