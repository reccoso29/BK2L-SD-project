"""
This module takes the bounding boxes from face_detection.py and
figures out which detections across consecutive frames belong to the same person.
It assigns each person a stable integer track ID that persists for
as long as they stay in the camera's view.

Why this matters:
  - Recognition is expensive. Tracking lets us recognize a face
    ONCE when it first appears, attach the identity to its track ID, and
    then cheaply follow the box without re-running recognition every frame.
  - The greeting system needs to know "this is a NEW person" vs.
    "this person has been here." Without tracking, every frame looks new.

How it works (IoU tracking):
  IoU = "Intersection over Union" — a measure of how much two rectangles
  overlap, from 0.0 (no overlap) to 1.0 (identical boxes). Between two
  consecutive frames at 30 FPS, the same person's face barely moves, so
  the IoU between their box in frame N and frame N+1 is very high (often
  >0.7). We use this to match detections across frames.

  For each new frame:
    1. Compute the IoU between every new detection and every existing track.
    2. Greedily match each new detection to the existing track with the
       highest IoU, as long as it's above a minimum threshold (default 0.3).
    3. Unmatched detections become new tracks.
    4. Existing tracks with no match are kept alive for a few frames (in case
       the detector momentarily lost them), then marked as lost.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from face_detection import BoundingBox

###################### Track data structure. #####################

@dataclass
class Track:
    """A tracked face with a persistent identity across frames.

    Attributes:
        track_id:      Unique integer ID for this track. Once assigned, it
                       never changes and is never reused by another person.
        bbox:          The most recent bounding box for this face.
        frames_seen:   How many total frames this track has been matched in.
                       Useful for filtering out brief false-positive flickers.
        frames_missed: How many consecutive frames since the last match.
                       When this exceeds max_missed_frames, the track is
                       considered lost and removed.
        is_recognized: Whether the recognition system (SD-11) has identified
                       this face. Set externally by the recognition pipeline.
        identity:      The name returned by recognition, or None if not yet
                       recognized or if the person is unknown. Set externally.
    """
    track_id: int
    bbox: BoundingBox
    frames_seen: int = 1
    frames_missed: int = 0
    is_recognized: bool = False
    identity: Optional[str] = None

###################### IoU calculation.  #####################
#
# This is the core math behind the tracker. If you've never seen IoU before,
# here's the intuition:
#
#   Imagine two rectangles on a table.
#   - "Intersection" = the area where they overlap.
#   - "Union" = the total area covered by both rectangles combined.
#   - IoU = intersection / union.
#
#   If the rectangles are identical, IoU = 1.0.
#   If they don't touch at all, IoU = 0.0.
#   Anything in between means partial overlap.
#
# Between consecutive video frames at 30 FPS (33ms apart), a person's face
# moves only a few pixels, so the IoU between their bounding box in frame N
# and frame N+1 is very high — typically 0.7 or above. That's what makes
# this simple approach work so well for our use case.

def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        A float between 0.0 (no overlap) and 1.0 (identical boxes).
    """
    # Find the coordinates of the intersection rectangle.
    # The intersection's left edge is the rightmost of the two left edges.
    # The intersection's right edge is the leftmost of the two right edges.
    # Same logic for top and bottom.
    inter_x1 = max(box_a.x, box_b.x)
    inter_y1 = max(box_a.y, box_b.y)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    # If the boxes don't overlap, the intersection width or height will be
    # negative. Clamp to zero in that case.
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Union = area_A + area_B - intersection (we subtract intersection
    # because it's counted in both areas).
    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    union_area = area_a + area_b - inter_area

    # Guard against division by zero (shouldn't happen with real faces,
    # but defensive coding is free).
    if union_area == 0:
        return 0.0

    return inter_area / union_area


###################### The tracker #####################

class FaceTracker:
    """Tracks faces across video frames using IoU matching.

    Usage:
        tracker = FaceTracker()

        # Each frame:
        detections = detector.detect(frame)
        active_tracks = tracker.update(detections)

        # active_tracks is a list of Track objects, each with a stable ID.
        for track in active_tracks:
            print(f"Track {track.track_id} at {track.bbox.center}")
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 10,
    ):
        """Create a face tracker.

        Args:
            iou_threshold: Minimum IoU to consider two boxes as the same
                           person. Lower = more forgiving of fast movement
                           or jittery detection. Higher = stricter matching.
                           0.3 is conservative and works well for people
                           walking at normal speed toward a stationary camera.

            max_missed_frames: How many consecutive frames a track can go
                               unmatched before it's considered lost and
                               removed. This handles momentary detection
                               failures (e.g., the person blinks and
                               MediaPipe loses them for 2 frames). At 30 FPS,
                               10 frames = about 0.33 seconds of grace period.
        """
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._next_id = 1          # Next track ID to assign (never reused)
        self._tracks: Dict[int, Track] = {}  # track_id -> Track

    def update(self, detections: List[BoundingBox]) -> List[Track]:
        """Process a new frame's detections and return updated tracks.

        This is the main method you call every frame. It:
          1. Matches new detections to existing tracks via IoU.
          2. Creates new tracks for unmatched detections.
          3. Ages out tracks that haven't been matched recently.

        Args:
            detections: List of BoundingBox objects from FaceDetector.detect().

        Returns:
            List of currently active Track objects (matched this frame or
            still within the grace period). Each Track has a stable track_id
            that persists across frames.
        """
        if not self._tracks:
            # No existing tracks — every detection is a new track.
            for det in detections:
                self._create_track(det)
            return list(self._tracks.values())

        if not detections:
            # No detections this frame — increment missed count for all tracks.
            self._age_tracks()
            return list(self._tracks.values())

        # --- Step 1: Compute IoU between every detection and every track ---
        #
        # We build a list of (iou_score, track_id, detection_index) tuples,
        # then sort by IoU descending. This lets us greedily match the best
        # pairs first.
        #
        # "Greedy matching" means: take the highest-IoU pair, lock it in,
        # then take the next highest that doesn't reuse a track or detection
        # that's already matched. It's not globally optimal (the Hungarian
        # algorithm would be), but it's fast, simple, and plenty good enough
        # when you only have 1-3 faces at a time, which is Knightro's case.

        iou_pairs: List[Tuple[float, int, int]] = []
        track_ids = list(self._tracks.keys())

        for det_idx, detection in enumerate(detections):
            for track_id in track_ids:
                track = self._tracks[track_id]
                iou = compute_iou(track.bbox, detection)
                if iou >= self._iou_threshold:
                    iou_pairs.append((iou, track_id, det_idx))

        # Sort by IoU descending — best matches first.
        iou_pairs.sort(key=lambda x: x[0], reverse=True)

        # --- Step 2: Greedy matching ---
        matched_track_ids: set = set()
        matched_det_indices: set = set()

        for iou_score, track_id, det_idx in iou_pairs:
            # Skip if either the track or detection is already matched.
            if track_id in matched_track_ids or det_idx in matched_det_indices:
                continue

            # Match found — update the track with the new bounding box.
            track = self._tracks[track_id]
            track.bbox = detections[det_idx]
            track.frames_seen += 1
            track.frames_missed = 0

            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)

        # --- Step 3: Create new tracks for unmatched detections ---
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_det_indices:
                self._create_track(detection)

        # --- Step 4: Age unmatched tracks and remove lost ones ---
        unmatched_track_ids = set(self._tracks.keys()) - matched_track_ids
        lost_ids = []
        for track_id in unmatched_track_ids:
            track = self._tracks[track_id]
            track.frames_missed += 1
            if track.frames_missed > self._max_missed_frames:
                lost_ids.append(track_id)

        for track_id in lost_ids:
            del self._tracks[track_id]

        return list(self._tracks.values())

    def get_new_tracks(self, detections: List[BoundingBox]) -> List[Track]:
        """Convenience: return only the tracks that were JUST created this frame.

        This is useful for the recognition pipeline — you only want to run
        recognition on faces that just appeared, not faces you've already
        identified. Call this AFTER update().

        Note: this is a simple implementation that checks frames_seen == 1.
        """
        return [t for t in self._tracks.values() if t.frames_seen == 1]

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Look up a specific track by its ID. Returns None if not found."""
        return self._tracks.get(track_id)

    @property
    def active_track_count(self) -> int:
        """How many tracks are currently active."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all tracks. Useful for testing or when restarting the system."""
        self._tracks.clear()
        self._next_id = 1

    # --- Private helpers ---

    def _create_track(self, bbox: BoundingBox) -> Track:
        """Create a new track with a fresh unique ID."""
        track = Track(track_id=self._next_id, bbox=bbox)
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _age_tracks(self) -> None:
        """Increment missed count for ALL tracks and remove lost ones."""
        lost_ids = []
        for track_id, track in self._tracks.items():
            track.frames_missed += 1
            if track.frames_missed > self._max_missed_frames:
                lost_ids.append(track_id)
        for track_id in lost_ids:
            del self._tracks[track_id]