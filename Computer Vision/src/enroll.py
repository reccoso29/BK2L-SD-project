"""
This script is the admin-facing tool described in the Privacy Policy.
It handles:
  - Enrolling a new faculty member (capturing face, computing embedding, saving to encrypted database)
  - Removing an enrolled faculty member
  - Listing currently enrolled faculty

Usage:
    # Enroll a new faculty member (opens webcam for photo capture)
    python src/enroll.py --enroll --name "Dr. Smith"

    # Enroll using a photo file instead of webcam
    python src/enroll.py --enroll --name "Dr. Smith" --photo path/to/photo.jpg

    # Remove a faculty member
    python src/enroll.py --remove --name "Dr. Smith"

    # List all enrolled faculty
    python src/enroll.py --list

Depends on: 
    > face_detection.py
    > face_embedder.py
    > face_database.py
"""

import argparse
import sys
import time
from pathlib import Path
 
# Add src/ to path so we can import our modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
 
import cv2
import numpy as np
 
from face_detection import FaceDetector, BoundingBox
from face_embedder import FaceEmbedder
 
 
# ---------------------------------------------------------------------------
# Face capture from webcam
# ---------------------------------------------------------------------------
 
def capture_faces_from_webcam(
    detector: FaceDetector,
    num_captures: int = 5,
    delay_between: float = 0.5,
) -> list[np.ndarray]:
    """Open the webcam and capture multiple face crops.
 
    The script guides the user through the capture process:
    press SPACE to capture each frame, or 'a' for automatic mode.
 
    Args:
        detector: A FaceDetector instance.
        num_captures: How many face images to capture (default 5).
        delay_between: Seconds between auto-captures.
 
    Returns:
        A list of BGR face crops (numpy arrays), one per capture.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return []
 
    print(f"\nWebcam opened. We need to capture {num_captures} face images.")
    print("Position the faculty member in front of the camera.")
    print("Controls:")
    print("  SPACE = capture a frame")
    print("  'a'   = auto-capture mode (captures every 0.5s)")
    print("  'q'   = quit without saving")
    print()
 
    face_crops: list[np.ndarray] = []
    auto_mode = False
    last_capture_time = 0.0
 
    with detector:
        while len(face_crops) < num_captures:
            success, frame = cap.read()
            if not success:
                print("WARN: Failed to read frame.")
                break
 
            frame = cv2.flip(frame, 1)
            boxes = detector.detect(frame)
 
            # Draw UI
            display = frame.copy()
            status = f"Captured: {len(face_crops)}/{num_captures}"
            if auto_mode:
                status += " [AUTO]"
 
            # Draw boxes around detected faces
            for box in boxes:
                color = (4, 201, 255)  # UCF Gold
                cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2), color, 2)
 
            # Status text with black outline + gold fill
            for color, thickness in [((0, 0, 0), 4), ((4, 201, 255), 2)]:
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
 
            if len(boxes) == 0:
                cv2.putText(display, "No face detected - adjust position",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
            elif len(boxes) > 1:
                cv2.putText(display, "Multiple faces - only one person please",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
 
            cv2.imshow("Knightro Enrollment", display)
 
            # Handle captures
            should_capture = False
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord('q'):
                print("Enrollment cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return []
            elif key == ord(' '):
                should_capture = True
            elif key == ord('a'):
                auto_mode = not auto_mode
                print(f"Auto-capture {'ON' if auto_mode else 'OFF'}")
 
            if auto_mode and (time.time() - last_capture_time) >= delay_between:
                should_capture = True
 
            # Capture if conditions met: exactly one face detected
            if should_capture and len(boxes) == 1:
                box = boxes[0]
                # Crop the face from the frame with a small margin.
                margin = 20
                y1 = max(0, box.y - margin)
                y2 = min(frame.shape[0], box.y2 + margin)
                x1 = max(0, box.x - margin)
                x2 = min(frame.shape[1], box.x2 + margin)
                crop = frame[y1:y2, x1:x2].copy()
 
                face_crops.append(crop)
                last_capture_time = time.time()
                print(f"  Captured {len(face_crops)}/{num_captures}")
 
                # Brief green flash to show capture happened
                cv2.rectangle(display, (0, 0),
                              (display.shape[1], display.shape[0]),
                              (0, 255, 0), 10)
                cv2.imshow("Knightro Enrollment", display)
                cv2.waitKey(200)
 
    cap.release()
    cv2.destroyAllWindows()
    return face_crops
 
 
def capture_face_from_photo(
    detector: FaceDetector,
    photo_path: str,
) -> list[np.ndarray]:
    """Load a photo file and extract the face crop.
 
    Args:
        detector: A FaceDetector instance.
        photo_path: Path to a photo file (jpg, png, etc.)
 
    Returns:
        A list containing a single face crop, or empty if no face found.
    """
    image = cv2.imread(photo_path)
    if image is None:
        print(f"ERROR: Could not read image at {photo_path}")
        return []
 
    with detector:
        boxes = detector.detect(image)
 
    if len(boxes) == 0:
        print("ERROR: No face detected in the photo.")
        return []
    if len(boxes) > 1:
        print(f"WARNING: {len(boxes)} faces found, using the largest one.")
        # Pick the largest face by area
        boxes.sort(key=lambda b: b.width * b.height, reverse=True)
 
    box = boxes[0]
    margin = 20
    y1 = max(0, box.y - margin)
    y2 = min(image.shape[0], box.y2 + margin)
    x1 = max(0, box.x - margin)
    x2 = min(image.shape[1], box.x2 + margin)
    crop = image[y1:y2, x1:x2].copy()
 
    print(f"Face detected with confidence {box.confidence:.2f}")
    return [crop]
 
 
# ---------------------------------------------------------------------------
# Enrollment logic
# ---------------------------------------------------------------------------
 
def enroll_faculty(name: str, photo_path: str | None = None) -> bool:
    """Enroll a faculty member in the face recognition system.
 
    Args:
        name: The faculty member's name (used for greetings).
        photo_path: Optional path to a photo. If None, uses webcam.
 
    Returns:
        True if enrollment succeeded, False otherwise.
    """
    print(f"\n=== Enrolling: {name} ===")
 
    detector = FaceDetector(min_confidence=0.5)
    embedder = FaceEmbedder()
 
    # Step 1: Capture face images
    if photo_path:
        crops = capture_face_from_photo(detector, photo_path)
    else:
        crops = capture_faces_from_webcam(detector, num_captures=10)
 
    if not crops:
        print("ERROR: No faces captured. Enrollment cancelled.")
        return False
 
    print(f"\nComputing embeddings from {len(crops)} capture(s)...")
 
    # Step 2: Compute embedding for each capture
    embeddings = []
    for i, crop in enumerate(crops):
        emb = embedder.embed(crop)
        embeddings.append(emb)
        print(f"  Embedding {i+1}/{len(crops)} computed (dim={len(emb)})")
 
    # Step 3: Store ALL embeddings (multi-template approach).
    # Instead of averaging into one embedding, we keep all of them.
    # During recognition, a new face must match a MAJORITY of these
    # templates to be considered a match. This dramatically reduces
    # false positives because an unknown person would need to
    # coincidentally match most of the templates, not just the average.
    print(f"\n{len(embeddings)} templates computed for {name}")
 
    # Step 4: Save to encrypted database.
    try:
        from face_database import FaceDatabase
        db = FaceDatabase()
        db.add_face(name, embeddings)
        db.save()
        print(f"SUCCESS: {name} enrolled with {len(embeddings)} templates "
              f"in encrypted database.")
    except ImportError:
        save_path = PROJECT_ROOT / "data" / "enrollments"
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / f"{name.replace(' ', '_')}.npy"
        np.save(str(file_path), np.array(embeddings))
        print(f"NOTE: face_database module not found (SD-10 not yet built).")
        print(f"Saved raw embeddings to {file_path}")
 
    return True
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Knightro Faculty Face Recognition — Enrollment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/enroll.py --enroll --name "Dr. Smith"
  python src/enroll.py --enroll --name "Dr. Smith" --photo headshot.jpg
  python src/enroll.py --remove --name "Dr. Smith"
  python src/enroll.py --list
        """,
    )
 
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--enroll", action="store_true",
                        help="Enroll a new faculty member")
    action.add_argument("--remove", action="store_true",
                        help="Remove an enrolled faculty member")
    action.add_argument("--list", action="store_true",
                        help="List all enrolled faculty")
 
    parser.add_argument("--name", type=str,
                        help="Faculty member's name (required for --enroll and --remove)")
    parser.add_argument("--photo", type=str, default=None,
                        help="Path to a photo file (optional, for --enroll)")
 
    args = parser.parse_args()
 
    if args.enroll:
        if not args.name:
            print("ERROR: --name is required for enrollment.")
            return 1
        success = enroll_faculty(args.name, args.photo)
        return 0 if success else 1
 
    elif args.remove:
        if not args.name:
            print("ERROR: --name is required for removal.")
            return 1
        try:
            from face_database import FaceDatabase
            db = FaceDatabase()
            db.remove_face(args.name)
            db.save()
            print(f"SUCCESS: {args.name} removed from database.")
        except ImportError:
            # Fallback: try to delete the .npy file
            file_path = PROJECT_ROOT / "data" / "enrollments" / f"{args.name.replace(' ', '_')}.npy"
            if file_path.exists():
                file_path.unlink()
                print(f"Removed {file_path}")
            else:
                print(f"No enrollment found for {args.name}")
        return 0
 
    elif args.list:
        try:
            from face_database import FaceDatabase
            db = FaceDatabase()
            names = db.list_enrolled()
            if names:
                print(f"\nEnrolled faculty ({len(names)}):")
                for name in sorted(names):
                    print(f"  - {name}")
            else:
                print("No faculty currently enrolled.")
        except ImportError:
            # Fallback: list .npy files
            enroll_dir = PROJECT_ROOT / "data" / "enrollments"
            if enroll_dir.exists():
                files = list(enroll_dir.glob("*.npy"))
                if files:
                    print(f"\nEnrolled faculty ({len(files)}):")
                    for f in sorted(files):
                        print(f"  - {f.stem.replace('_', ' ')}")
                else:
                    print("No faculty currently enrolled.")
            else:
                print("No faculty currently enrolled.")
        return 0
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())