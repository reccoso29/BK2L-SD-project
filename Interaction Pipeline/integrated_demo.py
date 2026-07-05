"""Knightro Integrated Demo
Face Recognition + Voice + TTS + Animations + GIF Eyes + Audio Clips

How to run:
    cd into repo root (BK2L-SD-project/)
    python3 "Interaction Pipeline/integrated_demo.py"

    Press 'q' to quit
    Press 'w' to simulate the wake word
"""

import os
import sys
import time
import threading
import queue

# ── Tell Python where our other files are ────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_CV_SRC       = os.path.join(_PROJECT_ROOT, "Computer Vision", "src")
_CV_ROOT      = os.path.join(_PROJECT_ROOT, "Computer Vision")

sys.path.insert(0, _CV_SRC)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import cv2

from face_detection  import FaceDetector, draw_boxes
from face_tracker    import FaceTracker
from face_recognizer import FaceRecognizer, RecognitionResult

import intent_detection
import interaction_router
import safety_filters
import system_state
import tts
import audio_clips

# ── Load animate.py (servo motors) ───────────────────────────────────────────
try:
    import animate
    _ANIMATE_AVAILABLE = True
    print("[demo] animate.py loaded — motors ACTIVE")
except ImportError:
    _ANIMATE_AVAILABLE = False
    print("[demo] WARNING: animate.py not found — motors DISABLED (using stub)")
    class _AnimateStub:
        def home(self): print("[animate-stub] home()")
        def play(self, action, stop_event=None): print(f"[animate-stub] play('{action}')")
        def play_loop(self, action, stop_event):
            print(f"[animate-stub] play_loop('{action}')")
            stop_event.wait()
    animate = _AnimateStub()

# ── Load gif_player.py (LED eye display) ─────────────────────────────────────
try:
    import gif_player
    _GIF_AVAILABLE = True
    print("[demo] gif_player.py loaded — LED eyes ACTIVE")
except ImportError:
    _GIF_AVAILABLE = False
    print("[demo] WARNING: gif_player.py not found — LED eyes DISABLED (using stub)")
    class _GifStub:
        def show(self, name): print(f"[gif-stub] show('{name}')")
        def stop(self): print("[gif-stub] stop()")
    gif_player = _GifStub()

# ── Settings ──────────────────────────────────────────────────────────────────
CAMERA_INDEX          = 0
RECOGNITION_COOLDOWN  = 10.0
GREETING_DELAY_FRAMES = 3
STT_MODE              = "whisper"
WAKE_WORD_MODE        = True

# ── The Master Coordination Table ────────────────────────────────────────────
COORDINATION_MAP = {
    "wake_word":      ("hello",     "hello"),
    "scanning":       ("thinking",  "thinking"),
    "known_person":   ("hello",     "Heart_Eyes"),
    "unknown_person": ("hello",     "hello"),
    "chant":          ("chant",     "chant"),
    "goknights":      ("goknights", "goknights"),
    "dance":          ("dance",     "dance"),
    "directions":     ("wave",      "wave"),
    "farewell":       ("goodbye",   "goodbye"),
    "thinking":       ("thinking",  "thinking"),
    "error":          ("error",     "Swirly_Eyes"),
    "football":       ("ucf",       "Fire_Eyes"),
    "weather":        ("wave",      "weather"),
    "next_game":      ("wave",      "next game"),
    "idle":           (None,        "default"),
}


def _do_moment(moment_name, speech_text=None, audio_clip=None, loop_anim=False, bell=True):
    """
    The main coordination function!
    Plays the right animation + GIF eyes + speech all at the same time.

    bell=False means don't ring the bell after speaking.
    Use this when Knightro is about to scan a face or wait for yes/no,
    because the bell tells the user to talk and that would be confusing!
    """
    anim_name, gif_name = COORDINATION_MAP.get(moment_name, (None, "default"))

    gif_player.show(gif_name)

    if anim_name is None:
        if speech_text:
            tts.speak(speech_text, bell=bell)
        if audio_clip:
            audio_clips.play(audio_clip)
        gif_player.show("default")
        return

    stop_event = threading.Event()
    if loop_anim:
        anim_thread = threading.Thread(
            target=animate.play_loop,
            args=(anim_name, stop_event),
            daemon=True
        )
    else:
        anim_thread = threading.Thread(
            target=animate.play,
            args=(anim_name, stop_event),
            daemon=True
        )
    anim_thread.start()

    if speech_text:
        tts.speak(speech_text, bell=bell)

    if audio_clip:
        audio_clips.play(audio_clip)

    stop_event.set()
    anim_thread.join(timeout=2.0)
    gif_player.show("default")


def _do_thinking_while(fn, *args, **kwargs):
    """
    Shows thinking animation + eyes while waiting for a slow function.
    Like Knightro scratching his head while thinking of an answer!
    Returns whatever fn() returned.
    """
    result_box = [None]
    error_box  = [None]

    def worker():
        try:
            result_box[0] = fn(*args, **kwargs)
        except Exception as e:
            error_box[0] = e

    gif_player.show("thinking")
    stop_event  = threading.Event()
    work_thread = threading.Thread(target=worker, daemon=True)
    anim_thread = threading.Thread(
        target=animate.play_loop,
        args=("thinking", stop_event),
        daemon=True
    )

    anim_thread.start()
    work_thread.start()
    work_thread.join()
    stop_event.set()
    anim_thread.join(timeout=2.0)
    gif_player.show("default")

    if error_box[0]:
        raise error_box[0]
    return result_box[0]


# ═════════════════════════════════════════════════════════════════════════════
class KnightroDemo:
# ═════════════════════════════════════════════════════════════════════════════

    def __init__(self):
        print("=" * 55)
        print("  Knightro Integrated Demo")
        print("  Face + Voice + Motors + Eyes + Audio")
        print("=" * 55)
        print()

        gif_player.show("default")
        print("[demo] Playing startup sound...")
        audio_clips.play_async("startup")

        print("[demo] Homing motors to safe position...")
        animate.home()

        print("[demo] Setting up face detection...")
        self.detector = FaceDetector(min_confidence=0.5)

        print("[demo] Setting up face tracker...")
        self.tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=15)

        from pathlib import Path
        from face_database import FaceDatabase

        cv_data_dir = Path(_CV_ROOT) / "data"
        db_path     = cv_data_dir / "dlib_face_embeddings.enc"
        key_path    = cv_data_dir / ".dlib_encryption_key"

        print(f"[demo] Face database: {db_path} (exists: {db_path.exists()})")
        db = FaceDatabase(db_path=db_path, key_path=key_path)

        print("[demo] Setting up face recognizer...")
        self.recognizer = FaceRecognizer(similarity_threshold=0.970, db=db)

        print(f"[demo] Internet: {'ONLINE' if system_state.is_online() else 'OFFLINE'}")
        print(f"[demo] Enrolled people: {self.recognizer.enrolled_count}")
        print()

        self._greeted: dict            = {}
        self._recognized_tracks: set   = set()
        self._activated                = not WAKE_WORD_MODE
        self._busy                     = False
        self._input_queue: queue.Queue = queue.Queue()
        self._status_msg = 'IDLE — Say "Hey Knightro!" or press W' if WAKE_WORD_MODE else "ACTIVE"

    # ──────────────────────────────────────────────────────────────────────────
    # Main camera loop
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("[demo] ERROR: Could not open camera!")
            return

        print("[demo] Camera is running!")
        print("  q = quit  |  w = simulate wake word")
        print()

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
            animate.home()
            gif_player.stop()
            print("[demo] Done. Go Knights!")

    # ──────────────────────────────────────────────────────────────────────────
    # Input handling
    # ──────────────────────────────────────────────────────────────────────────

    def _input_reader(self):
        while True:
            try:
                line = input().strip().lower()
                self._input_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                break

    def _get_speech_input(self, timeout: float = 20.0) -> str | None:
        """
        Listens for the user to say a COMMAND (not a wake word).
        This is used ONLY during active conversation.
        If it can't hear anything, it plays the helmet sound.
        """
        if STT_MODE == "typed":
            try:
                return self._input_queue.get(timeout=timeout)
            except queue.Empty:
                print("\n[demo] Timed out.")
                return None
        else:
            import speech_to_text
            result = speech_to_text.speech_to_text()
            if result["error"]:
                print(f"[demo] Microphone error: {result['reason']}")
                # Small pause so it doesnt overlap with previous audio
                time.sleep(0.5)
                # Play helmet sound — user tried to talk but we couldnt hear them
                _do_moment("error", audio_clip="cant_hear")
                return None
            return result["text"]

    def _get_wake_word_input(self, timeout: float = 2.0) -> str | None:
        """
        Listens for the WAKE WORD only.
        This NEVER plays the helmet sound when it times out —
        timing out here is totally normal and expected!
        Think of it like a motion sensor — it doesnt beep every time
        nobody walks by, it only reacts when someone actually does!
        """
        if STT_MODE == "typed":
            try:
                return self._input_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        else:
            import speech_to_text
            result = speech_to_text.speech_to_text()
            if result["error"]:
                # Timeout during wake word listening is NORMAL — just return None quietly
                return None
            return result["text"]

    # ──────────────────────────────────────────────────────────────────────────
    # Wake word
    # ──────────────────────────────────────────────────────────────────────────

    def _activate(self):
        self._status_msg = "ACTIVE — Scanning..."
        print("[demo] Wake word detected — ACTIVATING!")

        # Set busy=True RIGHT AWAY so the wake word listener stops immediately!
        # Think of it like hanging a do not disturb sign on the door
        self._busy = True

        # Small pause to let camera settle before speaking
        # This helps with glitchiness on laptop
        # On the Pi without a screen this wont matter at all
        time.sleep(0.3)

        # Speak BEFORE turning on face recognition
        # NO BELL here because we are about to scan a face not wait for a command!
        _do_moment("wake_word",
                   speech_text="Hello knight fan! Let me see who I am talking to!",
                   bell=False)

        # Give the Pi a moment to finish audio completely before face scanning
        time.sleep(1.0)

        # NOW turn on face recognition
        self._activated = True
        self._recognized_tracks.clear()
        self._greeted.clear()

        # Let face recognition take over
        self._busy = False

    def _deactivate(self):
        self._activated  = False
        self._busy       = False
        self._recognized_tracks.clear()
        self._greeted.clear()
        self._status_msg = 'IDLE — Say "Hey Knightro!" or press W'
        gif_player.show("default")
        print("[demo] Back to idle.")
        if WAKE_WORD_MODE:
            threading.Thread(target=self._wait_for_wake_word, daemon=True).start()

    def _wait_for_wake_word(self):
        """
        Keeps listening for Hey Knightro in the background.

        Key things:
        - Uses _get_wake_word_input (NOT _get_speech_input) so NO helmet sound on timeout
        - Checks self._busy so it stops immediately when Knightro is doing something
        - Uses short 2 second timeout so it reacts quickly to busy flag changes
        """
        WAKE_PHRASES = [
            "hey knightro", "knightro", "hey nitro", "nitro",
            "hey knight", "hey night row"
        ]
        while not self._activated:
            # If Knightro is busy talking or scanning, just wait!
            # Dont try to listen — that would cause the helmet sound to play
            if self._busy:
                time.sleep(0.5)
                continue

            self._status_msg = 'IDLE — Say "Hey Knightro!" or press W'
            print("[demo] Listening for wake word...")

            # Use wake word specific input — this NEVER plays helmet sound on timeout!
            text = self._get_wake_word_input(timeout=2.0)

            # Check if W was pressed while we were listening
            if self._activated:
                return

            if text is None:
                continue  # timed out quietly, just keep looping

            if any(phrase in text for phrase in WAKE_PHRASES):
                self._activate()
                return
            else:
                print(f"[demo] Heard '{text}' — not the wake word.")

    # ──────────────────────────────────────────────────────────────────────────
    # Face recognition
    # ──────────────────────────────────────────────────────────────────────────

    def _recognize_and_greet(self, frame, track):
        MAX_RETRIES = 3
        best_result = None

        gif_player.show("thinking")
        animate.play("thinking")

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

        # No bell — we are waiting for yes/no not a new command
        _do_moment("wake_word", speech_text=f"Hello! Are you {name}?", bell=False)
        self._status_msg = f"Asking: Are you {name}?"
        time.sleep(0.5)

        # Use speech input here — this CAN play helmet sound if cant hear yes/no
        text = self._get_speech_input(timeout=15.0)
        if text is None:
            tts.speak("I didn't catch that. Nice to meet you anyway! Go Knights!")
            self._finish_interaction()
            return

        yes_words = ["yes", "yeah", "yep", "yup", "that's me", "correct", "si", "ye"]
        no_words  = ["no", "nah", "nope", "wrong", "not me", "negative"]

        if any(w in text for w in yes_words):
            print(f"[demo] Confirmed: {name}")
            self._status_msg = f"Greeting {name}"
            greeting = self._greet_known(name)
            _do_moment("known_person", speech_text=greeting)
            self._greeted[name] = time.time()
            self._listen_for_commands()

        elif any(w in text for w in no_words):
            wrong_guesses.add(name)
            if guess_count >= MAX_GUESSES:
                tts.speak("Sorry about that! How can I help you today?")
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
            _do_moment("known_person",
                       speech_text=f"I'll take that as a yes! Welcome, {name}! Go Knights!")
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
        _do_moment("unknown_person", speech_text=random.choice(greetings))
        tts.speak("You can say try again if you think I should know you!")
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
            f"Well well well if it isn't {name}! Good to see you around campus!",
            f"Look who it is! Welcome, {name}! Charge On!",
            f"Hey there, {name}! Always good to see a familiar face!",
            f"{name}! My favorite Knight! How's it going?",
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # Command handling
    # ──────────────────────────────────────────────────────────────────────────

    def _listen_for_commands(self):
        self._status_msg = "Listening..."
        print("\n[demo] Listening for a command...")

        # Use speech input here — this CAN play helmet sound if cant hear
        text = self._get_speech_input(timeout=20.0)

        if text is None:
            # No response — say goodbye and go back to idle
            gif_player.show("goodbye")
            animate.play("goodbye")
            audio_clips.play("farewell")
            gif_player.show("default")
            self._finish_interaction()
            return

        retry_phrases = ["try again", "scan again", "look again", "retry", "rescan"]
        if any(phrase in text for phrase in retry_phrases):
            tts.speak("Let me take another look!")
            self._recognized_tracks.clear()
            self._greeted.clear()
            self.tracker.reset()
            self._busy = False
            return

        is_safe, _ = safety_filters.check_input_safety(text)
        if not is_safe:
            _do_moment("error",
                       speech_text="I can't help with that, but ask me about UCF!")
            self._listen_for_commands()
            return

        intent = intent_detection.detect_intent(text)

        if intent == "unknown" and system_state.is_online():
            import online_llm
            response = _do_thinking_while(online_llm.handle_unknown_with_cloud, text)
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

        print(f"[demo] Knightro says: {response}")

        # CHANT — only audio clip, no robot voice, no bell
        if intent == "chant":
            gif_player.show("chant")
            animate.play("chant")
            audio_clips.play("chant")
            gif_player.show("default")

        # GO KNIGHTS — only audio clip, no robot voice, no bell
        elif intent == "goknights":
            gif_player.show("goknights")
            animate.play("goknights")
            audio_clips.play("goknights")
            gif_player.show("default")

        # DANCE
        elif intent == "dance":
            _do_moment("dance", speech_text=response)

        # FAREWELL — only audio clip, no robot voice
        elif intent == "farewell":
            gif_player.show("goodbye")
            animate.play("goodbye")
            audio_clips.play("farewell")
            gif_player.show("default")
            self._finish_interaction()
            return

        # FOOTBALL — fire eyes + ucf animation + robot voice + touchdown clip
        elif intent == "ucf_trivia" and any(
            word in text.lower()
            for word in ["football", "touchdown", "score", "game", "bowl", "stadium"]
        ):
            _do_moment("football", speech_text=response, audio_clip="touchdown")

        # WEATHER
        elif any(word in text.lower()
                 for word in ["weather", "rain", "sunny", "temperature", "forecast"]):
            _do_moment("weather", speech_text=response)

        # NEXT GAME
        elif any(word in text.lower()
                 for word in ["next game", "schedule", "play next", "when do they play"]):
            _do_moment("next_game", speech_text=response)

        # DIRECTIONS
        elif intent == "directions":
            _do_moment("directions", speech_text=response)

        # UCF TRIVIA
        elif intent == "ucf_trivia":
            _do_moment("directions", speech_text=response)

        # GREETING/IDENTITY
        elif intent in ["greeting", "identity", "knightro_info"]:
            _do_moment("wake_word", speech_text=response)

        # EVERYTHING ELSE
        else:
            _do_moment("thinking", speech_text=response)

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
                    color = (0, 199, 255)
                    label = track.identity
                elif track.is_recognized:
                    color = (255, 255, 255)
                    label = "Visitor"
                else:
                    color = (128, 128, 128)
                    label = "Detecting..."
            else:
                color = (128, 128, 128)
                label = ""

            cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), color, 2)
            if label:
                cv2.putText(display, label, (box.x, max(0, box.y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bar_color = (0, 255, 0) if self._activated else (0, 165, 255)
        cv2.putText(display, self._status_msg, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

        hw = f"Motors: {'ON' if _ANIMATE_AVAILABLE else 'OFF'}  Eyes: {'ON' if _GIF_AVAILABLE else 'OFF'}"
        cv2.putText(display, hw, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0) if _ANIMATE_AVAILABLE else (0, 0, 255), 1)

        info = f"Enrolled: {self.recognizer.enrolled_count} | q=quit  w=wake"
        cv2.putText(display, info, (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return display


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = KnightroDemo()
    demo.run()