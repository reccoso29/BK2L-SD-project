"""Knightro Integrated Demo — Face Recognition + Voice + TTS + Animations + LED Eyes

This script ties together ALL subsystems:
  1. Camera opens and runs face detection/tracking/recognition
  2. Wake word ("Hey Knightro") activates the system
  3. Motors home to neutral position on startup
  4. LED matrix shows matching GIF eye animations throughout
  5. When a known person is detected, Knightro asks for confirmation
  6. Commands are routed through the interaction pipeline
  7. Knightro speaks (Piper TTS) AND moves AND shows eye GIF simultaneously

Usage:
    cd into repo root (BK2L-SD-project/)
    python3 "Interaction Pipeline/integrated_demo.py"

    Press 'q' in the camera window to quit.
    Press 'w' to simulate wake word.

Requirements (run in conda cv_env on Pi):
    pip install piper-tts sounddevice soundfile opencv-python mediapipe pillow
    dlib installed via conda
    rgi-rgb-led-matrix installed (see gif_player.py header)
    animate.py and gif_player.py must be in the repo root
"""

import os
import sys
import time
import threading
import queue

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_CV_SRC       = os.path.join(_PROJECT_ROOT, "Computer Vision", "src")
_CV_ROOT      = os.path.join(_PROJECT_ROOT, "Computer Vision")

sys.path.insert(0, _CV_SRC)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import cv2

from face_detection  import FaceDetector
from face_tracker    import FaceTracker
from face_recognizer import FaceRecognizer

import intent_detection
import interaction_router
import safety_filters
import system_state
import tts

# ── Animate (servo motors) ────────────────────────────────────────────────────
try:
    import animate
    _ANIMATE_AVAILABLE = True
    print("[demo] animate.py loaded — motors ACTIVE")
except ImportError:
    _ANIMATE_AVAILABLE = False
    print("[demo] WARNING: animate.py not found — motors DISABLED")
    class _AnimateStub:
        def home(self): print("[animate-stub] home()")
        def play(self, action, stop_event=None): print(f"[animate-stub] play('{action}')")
        def play_loop(self, action, stop_event):
            print(f"[animate-stub] play_loop('{action}')")
            stop_event.wait()
    animate = _AnimateStub()

# ── GIF player (LED matrix eyes) ─────────────────────────────────────────────
try:
    import gif_player
    _GIF_AVAILABLE = True
    print("[demo] gif_player.py loaded — LED eyes ACTIVE")
except ImportError:
    _GIF_AVAILABLE = False
    print("[demo] WARNING: gif_player.py not found — LED eyes DISABLED")
    class _GifStub:
        def show(self, name): print(f"[gif-stub] show('{name}')")
        def show_for_intent(self, intent): print(f"[gif-stub] show_for_intent('{intent}')")
        def stop(self): print("[gif-stub] stop()")
    gif_player = _GifStub()

# ── Settings ──────────────────────────────────────────────────────────────────
CAMERA_INDEX          = 0
RECOGNITION_COOLDOWN  = 10.0
GREETING_DELAY_FRAMES = 3
STT_MODE              = "whisper"   # "typed" or "whisper"
WAKE_WORD_MODE        = True

# ── Intent → animation mapping ────────────────────────────────────────────────
INTENT_ANIMATION_MAP = {
    "greeting":      "hello",
    "farewell":      "goodbye",
    "chant":         "chant",
    "dance":         "dance",
    "goknights":     "goknights",
    "known_person":  "hello",
    "identity":      "hello",
    "directions":    "wave",
    "ucf_trivia":    "wave",
    "knightro_info": "wave",
    "unknown":       "thinking",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _play_animation_with_speech(intent: str, speech_text: str):
    """
    Runs TTS + servo animation + GIF eye simultaneously.
    - GIF switches immediately to match the intent
    - Servo animation runs in background thread
    - TTS blocks until speech finishes, then stops servo animation
    - After speech, GIF returns to default (idle eyes)
    """
    # Switch eyes to match what Knightro is doing
    gif_player.show_for_intent(intent)

    animation_name = INTENT_ANIMATION_MAP.get(intent)

    if not animation_name:
        tts.speak(speech_text)
        gif_player.show("default")
        return

    if intent == "dance":
        # Dance: let full animation finish, speech runs alongside
        anim_thread = threading.Thread(
            target=animate.play, args=(animation_name,), daemon=True
        )
        anim_thread.start()
        tts.speak(speech_text)
        anim_thread.join()
        gif_player.show("default")
        return

    # All other intents: stop animation when speech ends
    stop_event  = threading.Event()
    anim_thread = threading.Thread(
        target=animate.play, args=(animation_name, stop_event), daemon=True
    )
    anim_thread.start()
    tts.speak(speech_text)
    stop_event.set()
    anim_thread.join(timeout=1.0)

    # Return eyes to default after speaking
    gif_player.show("default")


def _play_thinking_loop_while(fn, *args, **kwargs):
    """
    Loops the thinking animation AND thinking GIF while fn() runs.
    Returns whatever fn() returned.
    """
    result_box = [None]
    error_box  = [None]

    def worker():
        try:
            result_box[0] = fn(*args, **kwargs)
        except Exception as e:
            error_box[0] = e

    stop_event  = threading.Event()
    work_thread = threading.Thread(target=worker, daemon=True)
    anim_thread = threading.Thread(
        target=animate.play_loop, args=("thinking", stop_event), daemon=True
    )

    gif_player.show("thinking")   # eyes show thinking GIF
    anim_thread.start()
    work_thread.start()
    work_thread.join()
    stop_event.set()
    anim_thread.join(timeout=1.0)
    gif_player.show("default")    # back to idle eyes

    if error_box[0]:
        raise error_box[0]
    return result_box[0]


# ═════════════════════════════════════════════════════════════════════════════
class KnightroDemo:
# ═════════════════════════════════════════════════════════════════════════════

    def __init__(self):
        print("=" * 55)
        print("  Knightro Integrated Demo")
        print("  Face + Voice + TTS + Motors + LED Eyes")
        print("=" * 55)
        print()

        # Show default eyes immediately on startup
        gif_player.show("default")

        # Home motors to safe neutral position
        print("[demo] Homing motors...")
        animate.home()

        print("[demo] Initializing face detection...")
        self.detector = FaceDetector(min_confidence=0.5)

        print("[demo] Initializing face tracker...")
        self.tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=15)

        from pathlib import Path
        from face_database import FaceDatabase

        cv_data_dir = Path(_CV_ROOT) / "data"
        db_path     = cv_data_dir / "dlib_face_embeddings.enc"
        key_path    = cv_data_dir / ".dlib_encryption_key"

        print(f"[demo] Face database: {db_path} (exists: {db_path.exists()})")
        db = FaceDatabase(db_path=db_path, key_path=key_path)

        print("[demo] Initializing face recognizer...")
        self.recognizer = FaceRecognizer(similarity_threshold=0.970, db=db)

        print(f"[demo] System: {'ONLINE' if system_state.is_online() else 'OFFLINE'}")
        print(f"[demo] Enrolled: {self.recognizer.enrolled_count} people")
        print()

        self._greeted: dict           = {}
        self._recognized_tracks: set  = set()
        self._activated               = not WAKE_WORD_MODE
        self._busy                    = False
        self._input_queue: queue.Queue = queue.Queue()
        self._status_msg = 'IDLE — Say "Hey Knightro!" or press W' if WAKE_WORD_MODE else "ACTIVE"

    # ──────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[demo] ERROR: Could not open camera!")
            return

        print("[demo] Running. Controls:  q=quit  w=wake word")
        self._cap = cap

        if STT_MODE == "typed":
            threading.Thread(target=self._input_reader, daemon=True).start()

        if WAKE_WORD_MODE and not self._activated:
            threading.Thread(target=self._wait_for_wake_word, daemon=True).start()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame         = cv2.flip(frame, 1)
                boxes         = self.detector.detect(frame)
                active_tracks = self.tracker.update(boxes)

                if self._activated and not self._busy:
                    for track in active_tracks:
                        if track.track_id not in self._recognized_tracks:
                            if track.frames_seen >= GREETING_DELAY_FRAMES:
                                self._recognized_tracks.add(track.track_id)
                                self._busy = True
                                threading.Thread(
                                    target=self._recognize_and_greet,
                                    args=(frame.copy(), track),
                                    daemon=True,
                                ).start()

                display = self._draw_overlay(frame, active_tracks)
                cv2.imshow("Knightro Demo", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w') and not self._activated:
                    self._activate()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print("[demo] Shutting down...")
            animate.home()
            gif_player.stop()
            print("[demo] Done. Go Knights!")

    # ──────────────────────────────────────────────────────────────────────────
    # Input
    # ──────────────────────────────────────────────────────────────────────────

    def _input_reader(self):
        while True:
            try:
                line = input().strip().lower()
                self._input_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                break

    def _get_user_input(self, prompt: str, timeout: float = 30.0) -> str | None:
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

        if STT_MODE == "typed":
            print(f"{prompt}", end="", flush=True)
            try:
                return self._input_queue.get(timeout=timeout)
            except queue.Empty:
                print("\n[demo] Timed out.")
                return None
        else:
            import speech_to_text
            result = speech_to_text.speech_to_text()
            if result["error"]:
                print(f"[demo] STT error: {result['reason']}")
                return None
            return result["text"]

    # ──────────────────────────────────────────────────────────────────────────
    # Wake word
    # ──────────────────────────────────────────────────────────────────────────

    def _activate(self):
        self._status_msg = "ACTIVE — Scanning..."
        print("[demo] ACTIVATED!")
        _play_animation_with_speech(
            "greeting", "Hey! I'm here! Let me see who I'm talking to!"
        )
        time.sleep(0.5)
        self._activated = True
        self._recognized_tracks.clear()
        self._greeted.clear()

    def _deactivate(self):
        self._activated  = False
        self._busy       = False
        self._recognized_tracks.clear()
        self._greeted.clear()
        self._status_msg = 'IDLE — Say "Hey Knightro!" or press W'
        gif_player.show("default")   # back to idle eyes
        print("[demo] Back to idle.")
        if WAKE_WORD_MODE:
            threading.Thread(target=self._wait_for_wake_word, daemon=True).start()

    def _wait_for_wake_word(self):
        WAKE_PHRASES = [
            "hey knightro", "knightro", "hey nitro", "nitro",
            "hey knight", "hey night row"
        ]
        while not self._activated:
            self._status_msg = 'IDLE — Say "Hey Knightro!" or press W'
            print("[demo] Waiting for wake word...")
            text = self._get_user_input("", timeout=8.0)
            if self._activated:
                return
            if text is None:
                continue
            if any(p in text for p in WAKE_PHRASES):
                self._activate()
                return
            else:
                print(f"[demo] Heard '{text}', not wake word.")

    # ──────────────────────────────────────────────────────────────────────────
    # Face recognition + greeting
    # ──────────────────────────────────────────────────────────────────────────

    def _recognize_and_greet(self, frame, track):
        MAX_RETRIES = 3
        best_result = None

        # Show thinking eyes while recognizing
        gif_player.show("thinking")

        for attempt in range(MAX_RETRIES):
            if attempt > 0 and hasattr(self, '_cap') and self._cap is not None:
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                boxes = self.detector.detect(frame)
                if not boxes:
                    time.sleep(0.3)
                    continue
                largest    = max(boxes, key=lambda b: b.width * b.height)
                box_to_use = largest
            else:
                box_to_use = track.bbox

            margin = 20
            y1 = max(0, box_to_use.y  - margin)
            y2 = min(frame.shape[0], box_to_use.y2 + margin)
            x1 = max(0, box_to_use.x  - margin)
            x2 = min(frame.shape[1], box_to_use.x2 + margin)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            result = self.recognizer.recognize(face_crop)
            print(f"[demo] Attempt {attempt+1}/{MAX_RETRIES}: "
                  f"name={result.name}, distance={result.distance:.3f}, "
                  f"known={result.is_known}")

            if result.is_known:
                best_result = result
                break
            else:
                if best_result is None or result.distance < best_result.distance:
                    best_result = result
                time.sleep(0.3)

        if best_result is None:
            gif_player.show("default")
            self._busy = False
            return

        if best_result.is_known:
            self._status_msg = f"Recognized: {best_result.name}?"
            self._confirm_and_greet(best_result.name)
        else:
            self._status_msg = "Unknown visitor"
            self._greet_unknown_and_listen()

    def _confirm_and_greet(self, name: str, guess_count: int = 1, wrong_guesses: set = None):
        if wrong_guesses is None:
            wrong_guesses = set()
        MAX_GUESSES = 2

        _play_animation_with_speech("greeting", f"Hello! Are you {name}?")
        self._status_msg = f"Asking: Are you {name}?"
        time.sleep(0.5)

        text = self._get_user_input("You (yes/no): ", timeout=15.0)
        if text is None:
            tts.speak("I didn't catch that. Nice to meet you anyway! Go Knights!")
            self._finish_interaction()
            return

        yes_words = ["yes", "yeah", "yep", "yup", "that's me", "correct", "si", "ye"]
        no_words  = ["no", "nah", "nope", "wrong", "not me", "negative"]

        if any(w in text for w in yes_words):
            self._status_msg = f"Greeting {name}"
            greeting = self._greet_known(name)
            _play_animation_with_speech("known_person", greeting)
            self._greeted[name] = time.time()
            self._listen_for_commands()

        elif any(w in text for w in no_words):
            wrong_guesses.add(name)
            if guess_count >= MAX_GUESSES:
                tts.speak("Sorry about that! How can I help?")
                self._listen_for_commands()
            else:
                tts.speak("My apologies! Let me try again.")
                next_name = self._get_next_match(wrong_guesses)
                if next_name:
                    self._confirm_and_greet(next_name, guess_count + 1, wrong_guesses)
                else:
                    tts.speak("I don't think we've met! Welcome to UCF!")
                    self._listen_for_commands()
        else:
            _play_animation_with_speech(
                "known_person", f"I'll take that as a yes! Welcome, {name}! Go Knights!"
            )
            self._greeted[name] = time.time()
            self._listen_for_commands()

    def _greet_unknown_and_listen(self):
        import random
        greetings = [
            "Hey there! Welcome to UCF! I'm Knightro!",
            "What's up! Welcome to the Engineering building! Go Knights!",
            "Hey! I don't think we've met. I'm Knightro, UCF's mascot!",
            "Welcome, fellow Knight! How can I help you today?",
        ]
        _play_animation_with_speech("greeting", random.choice(greetings))
        tts.speak("You can say 'try again' if you think I should know you!")
        self._listen_for_commands()

    def _get_next_match(self, exclude: set) -> str | None:
        for name in self.recognizer._enrolled:
            if name not in exclude:
                return name
        return None

    def _greet_known(self, name: str) -> str:
        import random
        return random.choice([
            f"Hey, {name}! Great to see you! Welcome back! Go Knights!",
            f"Well well well, if it isn't {name}! Good to see you around campus!",
            f"Look who it is! Welcome, {name}! Charge On!",
            f"Hey there, {name}! Always good to see a familiar face!",
            f"{name}! My favorite Knight! How's it going?",
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # Command handling
    # ──────────────────────────────────────────────────────────────────────────

    def _listen_for_commands(self):
        self._status_msg = "Listening..."
        print("\n[demo] Listening... (command, 'try again', or 'goodbye')")

        text = self._get_user_input("You: ", timeout=20.0)
        if text is None:
            _play_animation_with_speech(
                "farewell", "Looks like you're done. See you around! Go Knights!"
            )
            self._finish_interaction()
            return

        retry_phrases = ["try again", "scan again", "look again", "retry",
                         "re-scan", "rescan", "check again"]
        if any(p in text for p in retry_phrases):
            tts.speak("Let me take another look!")
            self._recognized_tracks.clear()
            self._greeted.clear()
            self.tracker.reset()
            self._busy = False
            return

        is_safe, _ = safety_filters.check_input_safety(text)
        if not is_safe:
            tts.speak("I can't help with that, but ask me about UCF!")
            self._listen_for_commands()
            return

        intent = intent_detection.detect_intent(text)

        # Unknown intent with LLM → thinking animation + eyes while waiting
        if intent == "unknown" and system_state.is_online():
            import online_llm
            response = _play_thinking_loop_while(
                online_llm.handle_unknown_with_cloud, text
            )
            if response is None:
                import offline_interactions
                response = offline_interactions.handle_unknown_offline(text)
        else:
            response = interaction_router.route_intent(
                intent, text, input_passed_safety=True
            )

        is_safe, _ = safety_filters.check_output_safety(response)
        if not is_safe:
            response = "Let me think of something else. Ask me about UCF!"

        print(f"[demo] Knightro: {response}")
        _play_animation_with_speech(intent, response)

        if intent == "farewell":
            self._finish_interaction()
            return

        self._listen_for_commands()

    def _finish_interaction(self):
        self._busy = False
        if WAKE_WORD_MODE:
            self._deactivate()
        else:
            self._status_msg = "ACTIVE — Scanning..."
            self._recognized_tracks.clear()
            self._greeted.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Camera overlay
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame, tracks):
        display = frame.copy()

        for track in tracks:
            box = track.bbox
            if self._activated:
                if hasattr(track, 'identity') and track.identity:
                    color, label = (0, 199, 255), track.identity
                elif track.is_recognized:
                    color, label = (255, 255, 255), "Visitor"
                else:
                    color, label = (128, 128, 128), "Detecting..."
            else:
                color, label = (128, 128, 128), ""

            cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), color, 2)
            if label:
                cv2.putText(display, label, (box.x, max(0, box.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Status bar
        bar_color = (0, 255, 0) if self._activated else (0, 165, 255)
        cv2.putText(display, self._status_msg, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

        # Hardware status
        motor_color = (0, 255, 0) if _ANIMATE_AVAILABLE else (0, 0, 255)
        gif_color   = (0, 255, 0) if _GIF_AVAILABLE   else (0, 0, 255)
        cv2.putText(display,
                    f"Motors: {'ON' if _ANIMATE_AVAILABLE else 'OFF'}  "
                    f"Eyes: {'ON' if _GIF_AVAILABLE else 'OFF'}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, motor_color, 1)

        cv2.putText(display,
                    f"Enrolled: {self.recognizer.enrolled_count} | q=quit  w=wake",
                    (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = KnightroDemo()
    demo.run()