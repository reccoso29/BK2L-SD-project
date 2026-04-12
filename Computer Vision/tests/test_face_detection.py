"""
test_face_detection.py — Manual test/demo for the face detector.

Run this from the project root with:
    python tests/test_face_detection.py

It opens your webcam, runs face detection on every frame, draws bounding
boxes, and shows the FPS in the corner. Press 'q' to quit.

This is a MANUAL test, not an automated unit test. 
The goal here is to:
  1. Confirm the detector actually works on real webcam input
  2. Get a baseline FPS number on your laptop, which you'll compare against
     the Pi's FPS in Phase 2 (SD-7 Pi deployment step)
  3. Eyeball-verify the bounding boxes look right when you move around
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------
# This test script lives in tests/ but wants to import from src/.
# Rather than making the project a proper installable package (overkill for
# now), we just add src/ to Python's import path. __file__ is the path to
# this script, .parent is tests/, .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from face_detection import FaceDetector, draw_boxes


# ---------------------------------------------------------------------------
# A tiny rolling-average FPS counter
# ---------------------------------------------------------------------------
#
# If we just measured "1 / time_for_this_frame" the number would jitter
# wildly. Averaging over the last N frames gives a stable reading that
# reflects what you'd actually feel when interacting with the robot.

class FPSCounter:
    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._frame_times: list[float] = []
        self._last_time: float | None = None

    def tick(self) -> float:
        """Call this once per frame. Returns the current average FPS."""
        now = time.perf_counter()  # perf_counter is higher-res than time.time()

        if self._last_time is not None:
            delta = now - self._last_time
            self._frame_times.append(delta)
            # Keep only the last N frame times.
            if len(self._frame_times) > self._window_size:
                self._frame_times.pop(0)

        self._last_time = now

        if not self._frame_times:
            return 0.0

        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        if avg_frame_time == 0:
            return 0.0
        return 1.0 / avg_frame_time


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    # Open the default webcam. "0" means "first camera the OS knows about."
    # If you have multiple webcams and this picks the wrong one, try 1 or 2.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Is something else using it?")
        return 1

    print("Webcam opened. Press 'q' in the video window to quit.")
    print("Press 's' to save a snapshot of the current frame.")

    fps_counter = FPSCounter(window_size=30)

    # Use the detector as a context manager so MediaPipe cleans up even if
    # we crash somewhere inside the loop.
    with FaceDetector(min_confidence=0.5) as detector:
        while True:
            # cap.read() returns (success_flag, frame). If the camera fails
            # mid-stream (e.g. unplugged), success_flag is False.
            success, frame = cap.read()
            if not success:
                print("WARN: Failed to read frame from webcam, stopping.")
                break

            # Mirror horizontally so it feels like looking in a mirror.
            # (Standard UX for webcam demos — without it, moving your right
            # hand makes the image move left, which is disorienting.)
            frame = cv2.flip(frame, 1)

            # Run detection.
            boxes = detector.detect(frame)

            # Draw results for visualization.
            display_frame = draw_boxes(frame, boxes)

            # Overlay the FPS and face count in the top-left corner.
            fps = fps_counter.tick()
            status = f"FPS: {fps:5.1f}   Faces: {len(boxes)}"
            cv2.putText(
                display_frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                ( 0, 0,0 ),  #Black
                2,
            )

            # Show the frame in a window.
            cv2.imshow("Knightro Face Detection (SD-7 test)", display_frame)

            # Handle keyboard input. waitKey(1) waits 1ms for a key press;
            # this is also what pumps the window's event loop, so if you
            # remove it the window will freeze.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save a snapshot for reports / debugging.
                snapshot_path = PROJECT_ROOT / "data" / f"snapshot_{int(time.time())}.jpg"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snapshot_path), display_frame)
                print(f"Saved snapshot to {snapshot_path}")

    # Cleanup. Releasing the camera lets other applications use it afterwards;
    # destroyAllWindows closes the OpenCV window.
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    # The `if __name__ == "__main__"` guard means "only run main() if this
    # file is being run directly, not if it's being imported as a module."
    # Standard Python convention.
    sys.exit(main())