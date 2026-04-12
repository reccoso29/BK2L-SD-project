"""
face_detection.py — Lightweight face detection for Knightro.

This module wraps Google's MediaPipe Face Detector behind a simple interface
so the rest of the pipeline doesn't need to know the details.

The job of this file is narrow on purpose: given an image frame, find every
face in it and return a list of bounding boxes. That's it. 
It does NOT:
  - identify who the person is
  - track the same face across multiple frames
  - do anything with the face after detecting it

Satisfies SD-7 in the project plan.

--- Which MediaPipe API? ---
MediaPipe has two face detection APIs floating around:
  - Legacy:  mp.solutions.face_detection  (deprecated, removed in ~0.10.10+)
  - Modern:  mp.tasks.vision.FaceDetector (the new, supported path)

This module uses the modern Tasks API because the legacy one is going away.
The Tasks API requires a model file (`.tflite`) to be downloaded separately;
see get_model_path() below — the first run will auto-download it if missing.

Why MediaPipe at all: benchmarked in Tech Memo 1 at ~14 FPS on Raspberry Pi 4
with Lite models, clearing the 12–15 FPS floor needed for responsive HRI.
OpenCV Haar cascades would be an alternative but are older and worse on
angled faces and people wearing glasses.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------
#
# The Tasks API needs a .tflite model file on disk. Google hosts these on
# their MediaPipe CDN. We download it once and cache it in the models/
# folder next to the project so we don't re-download every run.
#
# "BlazeFace short-range" = the model optimized for faces within ~2 meters,
# which is exactly Knightro's use case. There's also a full-range model for
# up to 5 meters, but we don't need it.

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "blaze_face_short_range.tflite"
)


def get_model_path(model_path: Path | None = None) -> Path:
    """Return path to the face detector model, downloading it if missing.

    Args:
        model_path: Optional explicit path. If None, uses the default location
                    (project_root/models/blaze_face_short_range.tflite).

    Returns:
        Path to the .tflite model file, guaranteed to exist on disk.
    """
    path = model_path if model_path is not None else _DEFAULT_MODEL_PATH

    if not path.exists():
        print(f"[face_detection] Model not found at {path}, downloading...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, str(path))
        print(f"[face_detection] Model downloaded to {path}")

    return path


# ---------------------------------------------------------------------------
# Data type returned by the detector
# ---------------------------------------------------------------------------
#
# A dataclass is just a Python shortcut for "a class that holds some fields."
# Instead of writing __init__ yourself, you list the fields and Python generates
# the boilerplate. This gives us a clean typed object to pass around instead of
# raw tuples like (x, y, w, h, 0.97), which are easy to mess up the order of.

@dataclass
class BoundingBox:
    """A rectangle around a detected face, in pixel coordinates.

    Attributes:
        x: Left edge of the box, in pixels from the left of the image.
        y: Top edge of the box, in pixels from the top of the image.
        width: Width of the box in pixels.
        height: Height of the box in pixels.
        confidence: How sure MediaPipe is this is a face, from 0.0 to 1.0.
                    Values below ~0.5 are usually false positives.
    """
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def x2(self) -> int:
        """Right edge of the box (convenience)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge of the box (convenience)."""
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the box as (cx, cy). Used later by the tracker."""
        return (self.x + self.width // 2, self.y + self.height // 2)


# ---------------------------------------------------------------------------
# The detector class
# ---------------------------------------------------------------------------
#
# We wrap MediaPipe in our own class for three reasons:
#   1. The MediaPipe API is fiddly and we don't want to spread that fiddliness
#      across the whole codebase.
#   2. If we ever swap MediaPipe for something else (BlazeFace, YOLO, whatever),
#      we only need to change this one file — callers won't notice.
#   3. MediaPipe's detector is a stateful object that should be created once
#      and reused for every frame, not re-created per frame (slow).

class FaceDetector:
    """Wraps MediaPipe Face Detection behind a simple detect() method."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        model_path: Path | None = None,
    ):
        """Create a face detector.

        Args:
            min_confidence: Faces detected below this confidence are discarded.
                            0.5 is MediaPipe's recommended default. Raise it
                            (e.g. 0.7) if you're seeing false positives;
                            lower it (e.g. 0.3) if it's missing real faces.
            model_path: Optional override for the .tflite model file location.
                        Leave as None to use the default cached model.
        """
        self._min_confidence = min_confidence

        # Download (first time only) and locate the model file.
        resolved_model_path = get_model_path(model_path)

        # Build the Tasks API detector. The BaseOptions points at the model
        # file on disk; FaceDetectorOptions configures runtime behavior.
        base_options = mp_python.BaseOptions(
            model_asset_path=str(resolved_model_path)
        )
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
            # IMAGE mode = one frame at a time, synchronous. This is what we
            # want because our main loop processes frames in order. The other
            # modes (VIDEO, LIVE_STREAM) are for async pipelines.
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """Find all faces in a single image frame.

        Args:
            frame: A BGR image as a NumPy array, shape (height, width, 3).
                   This is what cv2.VideoCapture gives you by default.
                   BGR means "Blue, Green, Red" — OpenCV's color order, which
                   is backwards from the usual RGB. We convert below.

        Returns:
            A list of BoundingBox objects, one per detected face.
            Returns an empty list if no faces were found.
        """
        # MediaPipe expects RGB, OpenCV gives us BGR. Convert.
        # This is a cheap operation — it's just reordering the color channels.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # The Tasks API wants a mp.Image wrapper, not a raw numpy array.
        # SRGB means "standard RGB" — the color space we just converted to.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run the actual detection.
        result = self._detector.detect(mp_image)

        if not result.detections:
            return []

        # Unlike the legacy API, the Tasks API already gives us pixel
        # coordinates directly (not fractions). We still clamp to image
        # bounds because MediaPipe can return boxes that extend slightly
        # past the edge when a face is near the frame border.
        img_height, img_width = frame.shape[:2]
        boxes: List[BoundingBox] = []

        for detection in result.detections:
            bbox = detection.bounding_box
            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            # Clamp width/height so the box doesn't extend past the image.
            w = min(bbox.width, img_width - x)
            h = min(bbox.height, img_height - y)

            # detection.categories is a list of Category objects; for face
            # detection there's usually just one, with a score.
            confidence = (
                float(detection.categories[0].score)
                if detection.categories else 0.0
            )

            boxes.append(BoundingBox(
                x=int(x), y=int(y), width=int(w), height=int(h),
                confidence=confidence,
            ))

        return boxes

    def close(self) -> None:
        """Release MediaPipe resources. Call this when you're done with the
        detector — usually at program shutdown. You can also use the detector
        as a context manager (`with FaceDetector() as d:`) and skip this."""
        self._detector.close()

    # Context manager support — lets you write `with FaceDetector() as d:`
    # and have close() called automatically when the block exits, even on error.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# Drawing helper — useful for the test script and debugging
# ---------------------------------------------------------------------------

def draw_boxes(
    frame: np.ndarray,
    boxes: List[BoundingBox],
    color: tuple[int, int, int] = (0, 199, 255),  # gold in BGR
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a frame. Returns a NEW frame, leaves original alone.

    This is a convenience function for visualizing detection results during
    development and testing. The real runtime on the Pi won't draw anything —
    it'll just consume the bounding boxes directly.
    """
    # .copy() because we don't want to mutate the caller's frame unexpectedly.
    # If they need to do something else with the original frame, drawing on it
    # would be a nasty surprise.
    out = frame.copy()

    for box in boxes:
        # Draw the rectangle.
        cv2.rectangle(
            out,
            (box.x, box.y),
            (box.x2, box.y2),
            color,
            thickness,
        )

        # Draw the confidence score just above the box.
        label = f"{box.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (box.x, max(0, box.y - 5)),  # max() prevents text going off-screen
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # font scale
            color,
            1,    # line thickness
        )

    return out