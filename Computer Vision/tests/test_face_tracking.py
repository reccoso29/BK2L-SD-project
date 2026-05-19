"""
This builds on the detection demo by adding tracking. You'll see:
  - Bounding boxes drawn around detected faces
  - A track ID number displayed next to each face that PERSISTS across frames
  - The FPS and active track count in the top-left corner
  - Track IDs are color-coded so you can visually distinguish people

Try these things to verify tracking works:
  - Sit in front of the camera — you should get a stable track ID (e.g. #1)
  - Leave the frame and come back — you should get a NEW track ID (#2)
  - Have a second person walk in — they should get their own ID
  - Cover your face briefly and uncover — same ID should persist (grace period)

Press 'q' to quit, 's' to save a snapshot.
"""

import sys
import time
from pathlib import Path

# Import path setup 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from face_detection import FaceDetector
from face_tracker import FaceTracker, Track


# ---------------------------------------------------------------------------
# Color assignment for track IDs
# ---------------------------------------------------------------------------
# We want each track to have its own color so you can visually tell people
# apart. This function generates a deterministic color from a track ID by
# using a simple hash-like trick with the golden ratio. The golden ratio
# approach spreads colors evenly around the hue wheel so consecutive IDs
# don't get similar colors.

# UCF Gold in BGR for the primary display elements.
UCF_GOLD = (4, 201, 255)

def color_for_track(track_id: int) -> tuple:
    """Generate a distinct BGR color for a given track ID.

    Uses the golden ratio to spread hues evenly, so even IDs 1, 2, 3
    get visually distinct colors instead of three shades of red.
    """
    # Golden ratio conjugate — the "most irrational" number, which means
    # multiplying by it and taking the fractional part distributes values
    # as evenly as possible across [0, 1]. Math is cool.
    golden_ratio_conjugate = 0.618033988749895
    hue = (track_id * golden_ratio_conjugate) % 1.0

    # Convert HSV to BGR via OpenCV. We fix saturation and value at high
    # levels so colors are vivid, and only vary the hue.
    hsv = np.array([[[int(hue * 180), 230, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


# ---------------------------------------------------------------------------
# Drawing helper for tracked faces
# ---------------------------------------------------------------------------

def draw_tracked_faces(
    frame: np.ndarray,
    tracks: list,
) -> np.ndarray:
    """Draw bounding boxes with track IDs and color-coding."""
    out = frame.copy()

    for track in tracks:
        box = track.bbox
        color = color_for_track(track.track_id)

        # Draw the bounding box.
        cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2), color, 2)

        # Build the label: track ID + identity if recognized, or just the ID.
        if track.identity:
            label = f"#{track.track_id} {track.identity}"
        else:
            label = f"#{track.track_id}"

        # Add confidence score on second line.
        conf_label = f"{box.confidence:.2f}"

        # Draw a filled background rectangle behind the text so it's readable
        # against any background. This is a common trick in CV demos.
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = box.x
        text_y = max(0, box.y - 10)

        # Background rectangle.
        cv2.rectangle(
            out,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color,
            -1,  # -1 means "filled"
        )

        # Text in black on colored background for readability.
        cv2.putText(
            out, label,
            (text_x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2,
        )

        # Confidence score below the box.
        cv2.putText(
            out, conf_label,
            (box.x, box.y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1,
        )

    return out


# ---------------------------------------------------------------------------
# FPS counter (same as SD-7 test)
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
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    print("Webcam opened. Press 'q' to quit, 's' to save a snapshot.")
    print()
    print("What to look for:")
    print("  - Each face gets a colored box with a track ID (e.g. #1)")
    print("  - The ID persists as long as you stay in frame")
    print("  - Leave and re-enter -> you get a NEW ID")
    print("  - Cover your face briefly -> same ID should survive (grace period)")
    print()

    fps_counter = FPSCounter(window_size=30)
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=10)

    with FaceDetector(min_confidence=0.5) as detector:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARN: Failed to read frame, stopping.")
                break

            # Mirror for natural webcam feel.
            frame = cv2.flip(frame, 1)

            # Step 1: Detect faces (SD-7).
            detections = detector.detect(frame)

            # Step 2: Update tracker with new detections (SD-8).
            active_tracks = tracker.update(detections)

            # Draw tracked faces with IDs and colors.
            display_frame = draw_tracked_faces(frame, active_tracks)

            # Overlay status info in UCF Gold with black outline.
            fps = fps_counter.tick()
            status = f"FPS: {fps:5.1f}   Tracks: {tracker.active_track_count}"

            # Black outline first, then gold text on top.
            for color, thickness in [((0, 0, 0), 4), (UCF_GOLD, 2)]:
                cv2.putText(
                    display_frame, status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, thickness,
                )

            cv2.imshow("Knightro Face Tracking (SD-8 test)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snapshot_path = PROJECT_ROOT / "data" / f"tracking_snapshot_{int(time.time())}.jpg"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snapshot_path), display_frame)
                print(f"Saved snapshot to {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())