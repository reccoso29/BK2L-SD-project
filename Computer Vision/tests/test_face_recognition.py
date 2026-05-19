"""
test_face_recognition.py — Full real-time face recognition demo.

This is the culmination of SD-7 through SD-11. It opens the webcam,
detects faces, tracks them, recognizes enrolled faculty, and displays
personalized labels — exactly what Knightro will do in deployment.

Run this from the project root:
    python tests/test_face_recognition.py

Prerequisites:
  - At least one person enrolled (run `python src/enroll.py --enroll --name "Your Name"`)
  - All dependencies installed (pip install -r requirements.txt)

What you'll see:
  - Bounding boxes around detected faces
  - Track IDs that persist across frames
  - Names of recognized faculty (from the encrypted database)
  - "Unknown" for people not enrolled
  - FPS counter, active tracks, and enrolled count

Press 'q' to quit, 's' to save a snapshot, 'r' to reload the database
(if you enrolled someone in another terminal while this is running).
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from face_detection import FaceDetector
from face_tracker import FaceTracker, Track
from face_recognizer import FaceRecognizer, RecognitionResult

# UCF colors
UCF_GOLD = (4, 201, 255)
UCF_BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 200)


# ---------------------------------------------------------------------------
# FPS counter (reused from earlier tests)
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._frame_times: list = []
        self._last_time = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_time is not None:
            self._frame_times.append(now - self._last_time)
            if len(self._frame_times) > self._window_size:
                self._frame_times.pop(0)
        self._last_time = now
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0 else 0.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_recognition_results(
    frame: np.ndarray,
    tracks: list,
    recognition_cache: dict,
) -> np.ndarray:
    """Draw bounding boxes with recognition results.

    Args:
        frame: The raw camera frame.
        tracks: List of active Track objects from the tracker.
        recognition_cache: Dict mapping track_id -> RecognitionResult.

    Returns:
        A new frame with all annotations drawn.
    """
    out = frame.copy()

    for track in tracks:
        box = track.bbox
        result = recognition_cache.get(track.track_id)

        # Pick color and label based on recognition result.
        if result and result.is_known:
            color = GREEN
            label = f"{result.name} ({result.confidence:.0%})"
        elif result and not result.is_known:
            color = UCF_GOLD
            label = "Unknown"
        else:
            color = UCF_GOLD
            label = f"#{track.track_id}"

        # Draw bounding box.
        cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2), color, 2)

        # Draw label with background.
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = box.x
        text_y = max(0, box.y - 10)

        # Background rectangle for readability.
        cv2.rectangle(
            out,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color, -1,
        )
        cv2.putText(out, label, (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, UCF_BLACK, 2)

        # Confidence bar under the box (visual indicator).
        if result and result.is_known:
            bar_width = int(box.width * result.confidence)
            bar_y = box.y2 + 5
            cv2.rectangle(out, (box.x, bar_y), (box.x + bar_width, bar_y + 8),
                          GREEN, -1)
            cv2.rectangle(out, (box.x, bar_y), (box.x + box.width, bar_y + 8),
                          color, 1)

    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    print("=== Knightro Face Recognition Demo (SD-11) ===")
    print()

    # Initialize all pipeline components.
    detector = FaceDetector(min_confidence=0.5)
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=15)
    recognizer = FaceRecognizer(similarity_threshold=0.970)

    if recognizer.enrolled_count == 0:
        print("WARNING: No faculty enrolled! Everyone will show as 'Unknown'.")
        print("Enroll someone first: python src/enroll.py --enroll --name \"Your Name\"")
        print()

    print("Controls:")
    print("  q = quit")
    print("  s = save snapshot")
    print("  r = reload database (if you enrolled someone in another terminal)")
    print("  +/- = adjust recognition threshold")
    print()

    fps_counter = FPSCounter(window_size=30)

    # Cache recognition results by track ID so we don't re-recognize
    # the same person every frame. This is the efficiency win that the
    # tracker (SD-8) enables — recognize once, cache the result.
    recognition_cache: dict = {}  # track_id -> RecognitionResult

    # Track which IDs we've already tried to recognize.
    recognized_ids: set = set()

    with detector:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARN: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            # Step 1: Detect faces (SD-7).
            detections = detector.detect(frame)

            # Step 2: Track faces (SD-8).
            active_tracks = tracker.update(detections)

            # Step 3: Recognize NEW faces only (SD-11).
            # We only run the expensive embedding computation on faces
            # we haven't seen before. Once recognized, the result is
            # cached and reused for as long as the track persists.
            current_track_ids = {t.track_id for t in active_tracks}

            # Clean up cache for tracks that are gone.
            stale_ids = set(recognition_cache.keys()) - current_track_ids
            for stale_id in stale_ids:
                del recognition_cache[stale_id]
                recognized_ids.discard(stale_id)

            for track in active_tracks:
                # Only recognize faces we haven't processed yet,
                # and only if they've been visible for a few frames
                # (reduces false triggers from brief detection flickers).
                if (track.track_id not in recognized_ids
                        and track.frames_seen >= 3):
                    box = track.bbox
                    # Crop the face from the frame.
                    margin = 20
                    y1 = max(0, box.y - margin)
                    y2 = min(frame.shape[0], box.y2 + margin)
                    x1 = max(0, box.x - margin)
                    x2 = min(frame.shape[1], box.x2 + margin)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        result = recognizer.recognize(face_crop)
                        recognition_cache[track.track_id] = result
                        recognized_ids.add(track.track_id)

                        if result.is_known:
                            print(f"  Recognized: {result.name} "
                                  f"(distance: {result.distance:.3f}, "
                                  f"confidence: {result.confidence:.0%})")
                        else:
                            print(f"  Unknown face (closest distance: "
                                  f"{result.distance:.3f})")

            # Step 4: Draw everything.
            display = draw_recognition_results(frame, active_tracks,
                                               recognition_cache)

            # Status overlay.
            fps = fps_counter.tick()
            status = (f"FPS: {fps:5.1f}  |  "
                      f"Tracks: {tracker.active_track_count}  |  "
                      f"Enrolled: {recognizer.enrolled_count}  |  "
                      f"Threshold: {recognizer.threshold:.2f}")

            for color, thickness in [(UCF_BLACK, 4), (UCF_GOLD, 2)]:
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)

            cv2.imshow("Knightro Face Recognition (SD-11)", display)

            # Handle keyboard input.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = PROJECT_ROOT / "data" / f"recognition_{int(time.time())}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), display)
                print(f"Saved snapshot to {path}")
            elif key == ord('r'):
                recognizer.reload_database()
                recognition_cache.clear()
                recognized_ids.clear()
                print("Database reloaded and cache cleared.")
            elif key == ord('+') or key == ord('='):
                recognizer.threshold = min(0.99, recognizer.threshold + 0.05)
            elif key == ord('-'):
                recognizer.threshold = max(0.05, recognizer.threshold - 0.05)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())