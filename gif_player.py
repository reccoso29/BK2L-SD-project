"""gif_player.py — LED matrix eye display for Knightro

Drives the RGB LED matrix on Knightro's face, displaying looping GIFs
that match the current interaction state.

Hardware: 60x120 RGB LED matrix connected to the Pi via GPIO
Library:  rpi-rgb-led-matrix (must be installed on the Pi)
          https://github.com/hzeller/rpi-rgb-led-matrix

Install on Pi:
    sudo apt-get install python3-dev python3-pillow
    git clone https://github.com/hzeller/rpi-rgb-led-matrix.git
    cd rpi-rgb-led-matrix
    make build-python PYTHON=$(which python3)
    sudo make install-python PYTHON=$(which python3)

Usage (from integrated_demo.py):
    import gif_player
    gif_player.show("default")     # starts/switches the display
    gif_player.stop()              # blacks out the screen

The display runs in a background thread — it never blocks the main loop.

GIF → intent mapping (must match files in gifs/ folder):
    default       → idle / waiting for wake word
    thinking      → processing / LLM call in progress
    hello         → greeting / activation
    goodbye       → farewell
    chant         → chant intent
    dance         → dance intent
    goknights     → go knights intent
    error         → error / unknown
    wave          → directions, trivia, info
    cheerful_eyes → recognized known person
"""

import os
import sys
import threading
import time

# ── Path to gifs folder ───────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
GIFS_DIR      = os.path.join(_SCRIPT_DIR, "gifs")

# ── Matrix configuration ──────────────────────────────────────────────────────
# These values match the old BK2L3 hardware exactly — do not change unless
# the LED panel or wiring has been physically changed.
MATRIX_ROWS      = 60
MATRIX_COLS      = 120
HW_MAPPING       = "regular"
GPIO_SLOWDOWN    = 4
RGB_SEQUENCE     = "BGR"
PIXEL_MAPPER     = "Mirror:V"
BRIGHTNESS       = 100

# ── Try to import the RGB matrix library ─────────────────────────────────────
# On a laptop/dev machine this won't be installed, so we fall back to a stub
# that just prints what it would show — same pattern as animate.py.
try:
    from rgbmatrix import RGBMatrix, RGBMatrixOptions
    from PIL import Image
    _MATRIX_AVAILABLE = True
except ImportError:
    _MATRIX_AVAILABLE = False


# ── Internal state ────────────────────────────────────────────────────────────
_current_gif  = None       # name of gif currently showing
_stop_event   = threading.Event()
_thread       = None
_lock         = threading.Lock()


# ── Intent → GIF mapping ──────────────────────────────────────────────────────
INTENT_GIF_MAP = {
    "greeting":      "hello",
    "farewell":      "goodbye",
    "chant":         "chant",
    "dance":         "dance",
    "goknights":     "goknights",
    "known_person":  "cheerful_eyes",
    "identity":      "hello",
    "directions":    "wave",
    "ucf_trivia":    "wave",
    "knightro_info": "wave",
    "unknown":       "thinking",
    "error":         "error",
    "idle":          "default",
    "thinking":      "thinking",
    "recognized":    "cheerful_eyes",
}


def gif_for_intent(intent: str) -> str:
    """Returns the GIF name for a given intent. Falls back to 'default'."""
    return INTENT_GIF_MAP.get(intent, "default")


# ── Core display logic ────────────────────────────────────────────────────────

def _run_matrix(gif_name: str, stop_event: threading.Event):
    """
    Internal: runs in a background thread.
    Loads the GIF and loops it on the matrix until stop_event is set.
    """
    gif_path = os.path.join(GIFS_DIR, f"{gif_name}.gif")

    # black.png is used to blank the screen — handle separately
    if gif_name == "black":
        png_path = os.path.join(GIFS_DIR, "black.png")
        if not os.path.exists(png_path):
            print(f"[gif_player] black.png not found at {png_path}")
            return
        _show_static(png_path, stop_event)
        return

    if not os.path.exists(gif_path):
        print(f"[gif_player] WARNING: '{gif_name}.gif' not found in {GIFS_DIR}")
        print(f"[gif_player] Showing default instead")
        gif_path = os.path.join(GIFS_DIR, "default.gif")
        if not os.path.exists(gif_path):
            return

    if not _MATRIX_AVAILABLE:
        print(f"[gif_player-stub] Showing: {gif_name}")
        stop_event.wait()   # just hold until told to stop
        print(f"[gif_player-stub] Stopped: {gif_name}")
        return

    # ── Real matrix display ──
    try:
        from PIL import Image as PILImage

        options = RGBMatrixOptions()
        options.rows               = MATRIX_ROWS
        options.cols               = MATRIX_COLS
        options.hardware_mapping   = HW_MAPPING
        options.gpio_slowdown      = GPIO_SLOWDOWN
        options.led_rgb_sequence   = RGB_SEQUENCE
        options.pixel_mapper_config = PIXEL_MAPPER
        options.brightness         = BRIGHTNESS

        matrix = RGBMatrix(options=options)

        # Black out screen first (clean transition)
        blank = PILImage.open(os.path.join(GIFS_DIR, "black.png"))
        blank.thumbnail((matrix.width, matrix.height), PILImage.LANCZOS)
        matrix.SetImage(blank.convert("RGB"))
        time.sleep(0.05)

        # Pre-load all frames into memory for smooth playback
        gif = PILImage.open(gif_path)
        num_frames = gif.n_frames
        canvases = []
        for i in range(num_frames):
            gif.seek(i)
            frame = gif.copy()
            frame.thumbnail((matrix.width, matrix.height), PILImage.LANCZOS)
            canvas = matrix.CreateFrameCanvas()
            canvas.SetImage(frame.convert("RGB"))
            canvases.append(canvas)
        gif.close()

        print(f"[gif_player] Loaded {num_frames} frames for '{gif_name}'")

        # Loop until stop_event is set
        cur_frame = 0
        while not stop_event.is_set():
            matrix.SwapOnVSync(canvases[cur_frame], framerate_fraction=10)
            cur_frame = (cur_frame + 1) % num_frames

        # Black out when done
        blank = PILImage.open(os.path.join(GIFS_DIR, "black.png"))
        blank.thumbnail((matrix.width, matrix.height), PILImage.LANCZOS)
        matrix.SetImage(blank.convert("RGB"))

    except Exception as e:
        print(f"[gif_player] ERROR: {e}")


def _show_static(image_path: str, stop_event: threading.Event):
    """Shows a static image (used for black.png blanking)."""
    if not _MATRIX_AVAILABLE:
        print(f"[gif_player-stub] Showing static: {image_path}")
        stop_event.wait()
        return
    try:
        from PIL import Image as PILImage
        options = RGBMatrixOptions()
        options.rows = MATRIX_ROWS
        options.cols = MATRIX_COLS
        options.hardware_mapping = HW_MAPPING
        options.gpio_slowdown = GPIO_SLOWDOWN
        matrix = RGBMatrix(options=options)
        img = PILImage.open(image_path)
        img.thumbnail((matrix.width, matrix.height), PILImage.LANCZOS)
        matrix.SetImage(img.convert("RGB"))
        stop_event.wait()
    except Exception as e:
        print(f"[gif_player] Static image error: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def show(gif_name: str):
    """
    Switch the LED matrix to display a new GIF.
    If the same GIF is already showing, does nothing.
    Stops the previous GIF first, then starts the new one.

    Args:
        gif_name: Name without extension, e.g. "default", "thinking", "hello"
    """
    global _current_gif, _stop_event, _thread

    with _lock:
        if gif_name == _current_gif:
            return  # already showing this one

        # Stop the current thread if running
        if _thread and _thread.is_alive():
            _stop_event.set()
            _thread.join(timeout=2.0)

        # Start new thread
        _stop_event = threading.Event()
        _current_gif = gif_name
        _thread = threading.Thread(
            target=_run_matrix,
            args=(gif_name, _stop_event),
            daemon=True
        )
        _thread.start()
        print(f"[gif_player] Switching to: {gif_name}")


def stop():
    """Black out the LED matrix and stop the display thread."""
    global _current_gif, _stop_event, _thread

    with _lock:
        if _thread and _thread.is_alive():
            _stop_event.set()
            _thread.join(timeout=2.0)
        _current_gif = None
        print("[gif_player] Display stopped (blanked)")


def show_for_intent(intent: str):
    """
    Convenience function — looks up the right GIF for an intent and shows it.

    Example:
        gif_player.show_for_intent("chant")   # shows chant.gif
        gif_player.show_for_intent("farewell") # shows goodbye.gif
    """
    show(gif_for_intent(intent))


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== gif_player.py standalone test ===")
    print(f"GIFs folder: {GIFS_DIR}")
    print(f"Matrix available: {_MATRIX_AVAILABLE}")
    print()

    test_gifs = ["default", "thinking", "hello", "goodbye", "chant", "dance"]
    for name in test_gifs:
        print(f"Showing: {name} for 2 seconds...")
        show(name)
        time.sleep(2)

    stop()
    print("Test complete.")