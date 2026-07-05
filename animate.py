import csv
import socket
import time
import os
import threading

# ============================================================
#  BK2L animate.py
#  Based on the proven BK2L3 animate.py.
#  Sends servo commands over UDP to the Weigl motor controllers.
#
#  HOW IT WORKS:
#  - Each animation is a CSV file in the "animations/" folder
#  - Each row in the CSV = one frame of movement
#  - Each column = one servo channel (a percentage 0-100)
#  - Commands are sent over Ethernet to two Weigl controllers
#
#  HOW TO CALL FROM YOUR CODE:
#    import animate
#    animate.play("hello")        # plays hello_animation.csv
#    animate.play("wave")         # plays wave_animation.csv
#    animate.home()               # moves all servos to home position
#
#  TO RUN AS A STANDALONE TEST:
#    python animate.py
# ============================================================

# ---- Weigl Network Config ----
# These IP addresses are link-local (no router needed, direct Ethernet cable)
# DO NOT change these unless the Weigl controllers were reconfigured
WEIGL1_IP   = '169.254.43.97'
WEIGL2_IP   = '169.254.43.98'
WEIGL_PORT  = 5559

# ---- Animations folder ----
# This path is relative to wherever you run the script from.
# If you run from the repo root, this points to animations/ in the same folder.
ANIMATIONS_DIR = os.path.join(os.path.dirname(__file__), "animations")

# ---- Create UDP sockets once at import time ----
# UDP is "fire and forget" — no handshake, just send the packet.
# This is fast enough for real-time servo control.
_sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---- Thread-safety lock ----
# Prevents two animations from running at the same time and sending
# conflicting commands to the motors (e.g. if called from different threads)
_lock = threading.Lock()

def _send_frame(row):

    """
    Takes one row from the CSV (one frame) and sends each servo value
    as a UDP command to the correct Weigl controller.

    The command format is:  !esl{channel}%{value}#
      - channel = which servo port (1-16 on each Weigl)
      - value   = position as a percentage (0-100) of the servo's range

    Column 0 in the CSV is the frame number — we skip it.
    Columns 1-16  → Weigl 1
    Columns 17+   → Weigl 2 (channel resets back to 1 on the second Weigl)
    """
    for col_index, value in enumerate(row):
        if col_index == 0 or value.strip() == "":
            continue
        try:
            if col_index <= 16:
                command = f'!esl{col_index}%{value}#'
                _sock1.sendto(command.encode(), (WEIGL1_IP, WEIGL_PORT))
            else:
                channel = col_index - 16
                command = f'!esl{channel}%{value}#'
                _sock2.sendto(command.encode(), (WEIGL2_IP, WEIGL_PORT))
        except OSError:
            # Motors not reachable (normal when testing without the robot!)
            # Just skip silently and keep going
            pass



def play(action, stop_event=None):
    """
    Plays an animation by name.

    Args:
        action (str): The animation name. For example, "hello" will load
                      animations/hello_animation.csv
        stop_event (threading.Event, optional): If provided, the animation
                      will stop early when this event is set. Useful for
                      interrupting a looping animation (like "talking")
                      when TTS finishes.

    Example:
        animate.play("wave")
        animate.play("talking", stop_event=my_event)
    """
    animate_file = action + "_animation.csv"
    file_loc = os.path.join(ANIMATIONS_DIR, animate_file)

    if not os.path.exists(file_loc):
        print(f"[animate] WARNING: '{animate_file}' not found in {ANIMATIONS_DIR}")
        print(f"[animate] Available: {_list_available()}")
        return

    # "talking" is sped up because the CSV is very long
    fade_time = 1/10 if action == "talking" else 1/30

    print(f"[animate] Playing: {animate_file}")

    with _lock:
        with open(file_loc, 'r') as f:
            reader = csv.reader(f)
            # Skip the header row (column names)
            next(reader, None)

            for row in reader:
                # If the caller set a stop event (e.g. TTS finished), exit early
                if stop_event and stop_event.is_set():
                    print(f"[animate] Stop event received — interrupting '{action}'")
                    break
                _send_frame(row)
                time.sleep(fade_time)

    print(f"[animate] Finished: {animate_file}")


def home():
    """
    Moves all servos to the home (neutral) position.
    Reads from animations/home_profile.csv — a single-row CSV with
    the resting position for every servo channel.

    Call this on startup before any animations run, so the robot
    starts from a known safe position.
    """
    file_loc = os.path.join(ANIMATIONS_DIR, "home_profile.csv")

    if not os.path.exists(file_loc):
        print(f"[animate] WARNING: home_profile.csv not found — skipping homing")
        return

    print("[animate] Homing motors...")

    with open(file_loc, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            _send_frame(row)

    # Give servos time to physically reach home position before
    # anything else moves them
    time.sleep(2.0)
    print("[animate] Homing complete")


def play_loop(action, stop_event):
    """
    Loops an animation until stop_event is set.
    Useful for "thinking" or "talking" animations that should
    keep playing while TTS or LLM processing is happening in the background.

    Args:
        action (str): Animation name (same as play())
        stop_event (threading.Event): Loop runs until this is set.

    Example:
        stop = threading.Event()
        t = threading.Thread(target=animate.play_loop, args=("thinking", stop))
        t.start()
        # ... do LLM call here ...
        stop.set()
        t.join()
    """
    print(f"[animate] Looping: {action} (until stop event)")
    while not stop_event.is_set():
        play(action, stop_event=stop_event)
    print(f"[animate] Loop stopped: {action}")


def _list_available():
    """Returns a list of available animation names (without path or suffix)."""
    if not os.path.exists(ANIMATIONS_DIR):
        return []
    files = os.listdir(ANIMATIONS_DIR)
    return [f.replace("_animation.csv", "") for f in files if f.endswith("_animation.csv")]


# ---- Standalone test ----
if __name__ == "__main__":
    print("=== animate.py standalone test ===")
    print(f"Available animations: {_list_available()}")
    print("Running: home → hello → wave")
    home()
    play("hello")
    play("wave")
    print("Test complete.")