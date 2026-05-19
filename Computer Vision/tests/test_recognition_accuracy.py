"""
test_recognition_accuracy.py — Structured accuracy testing for face recognition.

This script runs a guided test session that measures:
  - True Positive Rate (TPR): enrolled person correctly recognized
  - False Positive Rate (FPR): unknown person incorrectly recognized
  - True Negative Rate (TNR): unknown person correctly rejected
  - False Negative Rate (FNR): enrolled person incorrectly rejected

It also records every distance measurement to help find the optimal
recognition threshold.

Run from the project root:
    python tests/test_recognition_accuracy.py

The script walks you through multiple test rounds:
  Round 1: Enrolled person sits in front of camera (tests TPR/FNR)
  Round 2: Unknown person sits in front of camera (tests TNR/FPR)
  Round 3: (Optional) Both people together

After all rounds, it prints a full results table and saves it to a file.

Satisfies SD-12 in the project plan.
"""

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from face_detection import FaceDetector
from face_embedder import FaceEmbedder
from face_recognizer import FaceRecognizer


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestAttempt:
    """A single recognition attempt during testing."""
    round_name: str
    expected_name: str       # who should be recognized (or "Unknown")
    actual_name: str         # who was actually recognized (or "Unknown")
    distance: float
    threshold: float
    is_correct: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestResults:
    """Aggregated results from all test rounds."""
    attempts: list = field(default_factory=list)

    def add(self, attempt: TestAttempt):
        self.attempts.append(attempt)

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def correct(self) -> int:
        return sum(1 for a in self.attempts if a.is_correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def get_round(self, round_name: str) -> list:
        return [a for a in self.attempts if a.round_name == round_name]

    def compute_metrics(self, round_name: str = None) -> dict:
        """Compute TP, FP, TN, FN counts and rates."""
        attempts = self.get_round(round_name) if round_name else self.attempts

        tp = sum(1 for a in attempts
                 if a.expected_name != "Unknown" and a.actual_name == a.expected_name)
        fp = sum(1 for a in attempts
                 if a.expected_name == "Unknown" and a.actual_name != "Unknown")
        tn = sum(1 for a in attempts
                 if a.expected_name == "Unknown" and a.actual_name == "Unknown")
        fn = sum(1 for a in attempts
                 if a.expected_name != "Unknown" and a.actual_name != a.expected_name)

        total = len(attempts)
        return {
            "total": total,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "tnr": tn / (fp + tn) if (fp + tn) > 0 else 0.0,
            "fnr": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
        }

    def get_distances(self, round_name: str = None) -> list:
        attempts = self.get_round(round_name) if round_name else self.attempts
        return [a.distance for a in attempts]


# ---------------------------------------------------------------------------
# Single test capture
# ---------------------------------------------------------------------------

def run_test_capture(
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    expected_name: str,
    round_name: str,
    num_samples: int = 10,
    delay: float = 0.5,
) -> list[TestAttempt]:
    """Run a series of recognition attempts from the webcam.

    Opens the webcam, waits for the test subject to be in position,
    then captures num_samples recognition attempts with a short delay
    between each.

    Args:
        detector: FaceDetector instance.
        recognizer: FaceRecognizer instance.
        expected_name: Who should be recognized ("Unknown" for unknown people).
        round_name: Label for this test round.
        num_samples: How many recognition samples to capture.
        delay: Seconds between samples.

    Returns:
        List of TestAttempt results.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return []

    print(f"\n  Camera is open. Position the test subject in front of the camera.")
    print(f"  Press SPACE when ready to start capturing {num_samples} samples.")
    print(f"  Press 'q' to skip this round.")

    attempts: list[TestAttempt] = []
    waiting_for_start = True
    last_sample_time = 0.0
    embedder_used = False

    with detector:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            boxes = detector.detect(frame)

            # Draw UI
            display = frame.copy()
            for box in boxes:
                cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2),
                              (4, 201, 255), 2)

            if waiting_for_start:
                status = f"[{round_name}] Press SPACE to start | Faces: {len(boxes)}"
            else:
                status = f"[{round_name}] Sampling: {len(attempts)}/{num_samples}"

            for color, thickness in [((0, 0, 0), 4), ((4, 201, 255), 2)]:
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

            cv2.imshow("Accuracy Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("  Round skipped.")
                break
            elif key == ord(' ') and waiting_for_start:
                waiting_for_start = False
                print(f"  Capturing {num_samples} samples...")

            # Capture samples
            if not waiting_for_start and len(attempts) < num_samples:
                now = time.time()
                if now - last_sample_time >= delay and len(boxes) >= 1:
                    # Use the largest face
                    box = max(boxes, key=lambda b: b.width * b.height)
                    margin = 20
                    y1 = max(0, box.y - margin)
                    y2 = min(frame.shape[0], box.y2 + margin)
                    x1 = max(0, box.x - margin)
                    x2 = min(frame.shape[1], box.x2 + margin)
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        result = recognizer.recognize(crop)
                        actual = result.name if result.is_known else "Unknown"
                        is_correct = (actual == expected_name)

                        attempt = TestAttempt(
                            round_name=round_name,
                            expected_name=expected_name,
                            actual_name=actual,
                            distance=result.distance,
                            threshold=recognizer.threshold,
                            is_correct=is_correct,
                        )
                        attempts.append(attempt)
                        last_sample_time = now

                        marker = "✓" if is_correct else "✗"
                        print(f"    {marker} Sample {len(attempts)}: "
                              f"expected={expected_name}, got={actual}, "
                              f"distance={result.distance:.3f}")

            if not waiting_for_start and len(attempts) >= num_samples:
                break

    cap.release()
    cv2.destroyAllWindows()
    return attempts


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: TestResults, threshold: float):
    """Print a formatted accuracy report."""

    print("\n")
    print("=" * 65)
    print("  KNIGHTRO FACE RECOGNITION — ACCURACY TEST REPORT")
    print("=" * 65)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Threshold: {threshold}")
    print(f"  Total samples: {results.total}")
    print(f"  Overall accuracy: {results.accuracy:.1%}")
    print("=" * 65)

    # Per-round breakdown
    round_names = sorted(set(a.round_name for a in results.attempts))
    for rname in round_names:
        metrics = results.compute_metrics(rname)
        distances = results.get_distances(rname)
        print(f"\n  Round: {rname}")
        print(f"  {'─' * 55}")
        print(f"  Samples: {metrics['total']}")
        print(f"  True Positives:  {metrics['tp']}    "
              f"True Negatives:  {metrics['tn']}")
        print(f"  False Positives: {metrics['fp']}    "
              f"False Negatives: {metrics['fn']}")
        print(f"  TPR (sensitivity): {metrics['tpr']:.1%}    "
              f"TNR (specificity): {metrics['tnr']:.1%}")
        print(f"  FPR:               {metrics['fpr']:.1%}    "
              f"FNR:               {metrics['fnr']:.1%}")
        if distances:
            print(f"  Distance — min: {min(distances):.3f}, "
                  f"max: {max(distances):.3f}, "
                  f"mean: {np.mean(distances):.3f}, "
                  f"std: {np.std(distances):.3f}")

    # Overall metrics
    overall = results.compute_metrics()
    print(f"\n  OVERALL")
    print(f"  {'─' * 55}")
    print(f"  Accuracy:    {overall['accuracy']:.1%}")
    print(f"  TPR:         {overall['tpr']:.1%}")
    print(f"  FPR:         {overall['fpr']:.1%}")
    print(f"  TNR:         {overall['tnr']:.1%}")
    print(f"  FNR:         {overall['fnr']:.1%}")

    # Distance analysis for threshold tuning
    known_distances = [a.distance for a in results.attempts
                       if a.expected_name != "Unknown"]
    unknown_distances = [a.distance for a in results.attempts
                         if a.expected_name == "Unknown"]

    if known_distances and unknown_distances:
        print(f"\n  THRESHOLD ANALYSIS")
        print(f"  {'─' * 55}")
        print(f"  Known person distances:   "
              f"min={min(known_distances):.3f}, "
              f"max={max(known_distances):.3f}, "
              f"mean={np.mean(known_distances):.3f}")
        print(f"  Unknown person distances: "
              f"min={min(unknown_distances):.3f}, "
              f"max={max(unknown_distances):.3f}, "
              f"mean={np.mean(unknown_distances):.3f}")

        gap = min(unknown_distances) - max(known_distances)
        suggested = (max(known_distances) + min(unknown_distances)) / 2

        if gap > 0:
            print(f"\n  Gap between known max and unknown min: {gap:.3f}")
            print(f"  ✓ CLEAN SEPARATION — suggested threshold: {suggested:.3f}")
        else:
            print(f"\n  ⚠ OVERLAP — known max ({max(known_distances):.3f}) > "
                  f"unknown min ({min(unknown_distances):.3f})")
            print(f"  Suggested threshold (midpoint): {suggested:.3f}")
            print(f"  Consider re-enrolling with more varied captures.")

    print(f"\n{'=' * 65}")


def save_report(results: TestResults, threshold: float):
    """Save results to a JSON file for later analysis."""
    save_dir = PROJECT_ROOT / "data"
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"accuracy_test_{timestamp}.json"

    report_data = {
        "timestamp": timestamp,
        "threshold": threshold,
        "total_samples": results.total,
        "overall_accuracy": results.accuracy,
        "metrics": results.compute_metrics(),
        "attempts": [
            {
                "round": a.round_name,
                "expected": a.expected_name,
                "actual": a.actual_name,
                "distance": a.distance,
                "threshold": a.threshold,
                "correct": a.is_correct,
            }
            for a in results.attempts
        ],
    }

    with open(save_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nResults saved to {save_path}")


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 65)
    print("  KNIGHTRO FACE RECOGNITION — ACCURACY TEST (SD-12)")
    print("=" * 65)
    print()

    # Ask for threshold
    print("Current default threshold is 1.1")
    threshold_input = input("Enter threshold to test (or press Enter for 1.1): ").strip()
    threshold = float(threshold_input) if threshold_input else 1.1

    detector = FaceDetector(min_confidence=0.5)
    recognizer = FaceRecognizer(similarity_threshold=threshold)

    if recognizer.enrolled_count == 0:
        print("\nERROR: No faculty enrolled. Enroll at least one person first.")
        print("  python src/enroll.py --enroll --name \"Your Name\"")
        return 1

    print(f"\nEnrolled faculty: {recognizer.enrolled_count}")
    print(f"Threshold: {threshold}")

    results = TestResults()

    # --- Round 1: Enrolled person ---
    print("\n" + "=" * 65)
    print("  ROUND 1: ENROLLED PERSON")
    print("=" * 65)

    enrolled_name = input("\nEnter the enrolled person's name exactly as enrolled: ").strip()
    if not enrolled_name:
        print("Skipping Round 1.")
    else:
        print(f"\nTesting: {enrolled_name} (should be RECOGNIZED)")
        print(f"We'll capture 10 samples. Have {enrolled_name} sit in front of the camera.")
        attempts = run_test_capture(
            detector, recognizer,
            expected_name=enrolled_name,
            round_name="Enrolled Person",
            num_samples=10,
            delay=0.5,
        )
        for a in attempts:
            results.add(a)
        correct = sum(1 for a in attempts if a.is_correct)
        print(f"\n  Round 1 result: {correct}/{len(attempts)} correct")

    # --- Round 2: Unknown person ---
    print("\n" + "=" * 65)
    print("  ROUND 2: UNKNOWN PERSON")
    print("=" * 65)

    proceed = input("\nReady to test an unknown person? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipping Round 2.")
    else:
        print("\nTesting: Unknown person (should be REJECTED)")
        print("Have the unenrolled person sit in front of the camera.")
        attempts = run_test_capture(
            detector, recognizer,
            expected_name="Unknown",
            round_name="Unknown Person",
            num_samples=10,
            delay=0.5,
        )
        for a in attempts:
            results.add(a)
        correct = sum(1 for a in attempts if a.is_correct)
        print(f"\n  Round 2 result: {correct}/{len(attempts)} correct")

    # --- Round 3: Varied conditions (optional) ---
    print("\n" + "=" * 65)
    print("  ROUND 3: VARIED CONDITIONS (Optional)")
    print("=" * 65)

    proceed = input("\nTest enrolled person with varied conditions? (y/n): ").strip().lower()
    if proceed == 'y':
        print(f"\nHave {enrolled_name} try different conditions:")
        print("  - Different distance from camera")
        print("  - With/without glasses")
        print("  - Different angles")
        print("  - Different lighting (turn away from/toward the light)")
        attempts = run_test_capture(
            detector, recognizer,
            expected_name=enrolled_name,
            round_name="Varied Conditions",
            num_samples=10,
            delay=0.5,
        )
        for a in attempts:
            results.add(a)
        correct = sum(1 for a in attempts if a.is_correct)
        print(f"\n  Round 3 result: {correct}/{len(attempts)} correct")

    # --- Report ---
    if results.total > 0:
        print_report(results, threshold)
        save_report(results, threshold)
    else:
        print("\nNo test data collected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())