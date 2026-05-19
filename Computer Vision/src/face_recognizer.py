"""
This module ties together the entire face recognition pipeline:
  - Takes a cropped face image from face_detection.py
  - Computes its embedding using face_embedder.py
  - Compares it against all enrolled faculty from face_database.py
  - Returns the best match or "unknown"

"1:N" means: one detected face compared against N enrolled people.
This is different from "1:1" verification (where you already know who
the person claims to be and just want to confirm). In our case, Knightro doesn't know who just walked up,
it has to figure it out by checking everyone in the database.

The math:
  - Each face is represented as a 512-number vector (the "embedding").
  - Two embeddings of the same person will be close together in
    512-dimensional space (small Euclidean distance).
  - Two embeddings of different people will be far apart (large distance).
  - We compute the distance between the new face and every enrolled face,
    find the smallest one, and if it's below a threshold, it's a match.

Satisfies requirement 4.4 from M2 (refuses to identify unknown people).
"""

from dataclasses import dataclass
from typing import Dict, Optional
 
import numpy as np
 
from face_embedder import FaceEmbedder
from face_database import FaceDatabase
 
##################### Recognition result #####################
 
@dataclass
class RecognitionResult:
    """The result of trying to recognize a single face.
 
    Attributes:
        name: The matched faculty member's name, or None if unknown.
        distance: The average cosine similarity between the detected face
                  and the best match's templates. Higher = more similar.
                  Range: -1.0 to 1.0 (1.0 = identical).
        is_known: True if the face matched someone in the database.
        confidence: A rough confidence score from 0.0 to 1.0.
    """
    name: Optional[str]
    distance: float  # actually cosine similarity (higher = better)
    is_known: bool
 
    @property
    def confidence(self) -> float:
        """Convert cosine similarity to a 0-1 confidence score.
 
        Cosine similarity ranges from -1 to 1. We map:
          similarity 1.0 -> confidence 1.0
          similarity 0.0 -> confidence 0.0
          similarity < 0 -> confidence 0.0
        """
        return max(0.0, min(1.0, self.distance))
 
 
###################### The recognizer #####################
 
class FaceRecognizer:
    """Recognizes faces by comparing against enrolled faculty embeddings.
 
    Usage:
        recognizer = FaceRecognizer()
 
        # Given a face crop (BGR numpy array from the camera):
        result = recognizer.recognize(face_crop)
        if result.is_known:
            print(f"Hello, {result.name}! (confidence: {result.confidence:.0%})")
        else:
            print("Hello, visitor!")
 
    The recognizer loads the enrolled faculty from the encrypted database
    on creation. If you enroll new faculty while the recognizer is running,
    call reload_database() to pick up the changes.
    """
 
    def __init__(
        self,
        similarity_threshold: float = 0.45,
        min_match_ratio: float = 0.6,
        db: Optional[FaceDatabase] = None,
        embedder: Optional[FaceEmbedder] = None,
    ):
        """Create a face recognizer.
 
        Args:
            similarity_threshold: Minimum cosine similarity to consider a
                single template as matching. Range is -1.0 to 1.0 where
                1.0 = identical, 0.0 = unrelated, -1.0 = opposite.
                Default 0.45 — tune during SD-12 accuracy testing.
 
                ArcFace was trained with angular margin loss, meaning it
                was specifically designed to maximize the ANGULAR separation
                between different people. Cosine similarity measures exactly
                this — the angle between two vectors — making it the correct
                metric for this model.
 
            min_match_ratio: Minimum fraction of enrolled templates that
                must match for a person to be recognized. Default 0.6
                means at least 3 out of 5 templates must be above the
                similarity threshold.
 
            db: Optional FaceDatabase instance.
            embedder: Optional FaceEmbedder instance.
        """
        self._threshold = similarity_threshold
        self._min_match_ratio = min_match_ratio
        self._embedder = embedder or FaceEmbedder()
        self._db = db or FaceDatabase()
 
        self._enrolled: Dict[str, list] = self._db.get_all()
        print(f"[face_recognizer] Ready. {len(self._enrolled)} faculty enrolled. "
              f"Similarity threshold: {self._threshold}, "
              f"Min match ratio: {self._min_match_ratio}")
 
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
 
        Returns a value from -1.0 to 1.0:
          1.0 = identical direction (same person)
          0.0 = perpendicular (unrelated)
         -1.0 = opposite direction
 
        Since our embeddings are L2-normalized (unit vectors), cosine
        similarity simplifies to just the dot product.
        """
        return float(np.dot(a, b))
 
    def recognize(self, face_bgr: np.ndarray) -> RecognitionResult:
        """Recognize a single face against all enrolled faculty.
 
        Uses cosine similarity with two levels of filtering:
        1. Majority vote: the face must match enough of a person's templates.
        2. Margin check: when multiple people pass the majority vote, the
           best match must be significantly better than the second-best.
           If the margin is too small, the result is "Unknown" to avoid
           misidentification.
 
        Args:
            face_bgr: A BGR image containing the cropped face.
 
        Returns:
            A RecognitionResult with the best match (or unknown).
        """
        if not self._enrolled:
            return RecognitionResult(name=None, distance=0.0, is_known=False)
 
        query_embedding = self._embedder.embed(face_bgr)
 
        # Step 1: Compute scores for every enrolled person.
        # For each person, we compute:
        #   - max_similarity: the highest similarity across all their templates
        #     (how well the BEST template matches)
        #   - avg_similarity: the average similarity across all templates
        #   - match_ratio: fraction of templates above the threshold
        candidates = []
 
        for name, templates in self._enrolled.items():
            similarities = [
                self._cosine_similarity(query_embedding, tmpl)
                for tmpl in templates
            ]
 
            max_similarity = max(similarities)
            avg_similarity = sum(similarities) / len(similarities)
            matches = sum(1 for s in similarities if s >= self._threshold)
            match_ratio = matches / len(templates) if templates else 0.0
 
            candidates.append({
                "name": name,
                "max_sim": max_similarity,
                "avg_sim": avg_similarity,
                "match_ratio": match_ratio,
            })
 
        # Step 2: Sort by max_similarity (best single-template match first).
        # Using max instead of average because the BEST template match is
        # more reliable than the average — it represents the angle where
        # this person's face most closely matches the enrollment.
        candidates.sort(key=lambda c: c["max_sim"], reverse=True)
 
        best = candidates[0]
 
        # Step 3: Check if the best candidate passes the majority vote.
        if best["match_ratio"] < self._min_match_ratio:
            return RecognitionResult(
                name=None,
                distance=best["avg_sim"],
                is_known=False,
            )
 
        # Step 4: Margin check — if there's a second candidate, make sure
        # the best is clearly better. This prevents misidentification when
        # two enrolled people have similar embeddings.
        if len(candidates) >= 2:
            second = candidates[1]
            margin = best["max_sim"] - second["max_sim"]
 
            # If the margin is too small, the system can't confidently
            # distinguish between the two people. Return the best match
            # ONLY if there's enough separation.
            min_margin = 0.003  # tunable — very small but meaningful in cosine space
            if margin < min_margin and second["match_ratio"] >= self._min_match_ratio:
                # Both candidates are viable and too close to call.
                # Return Unknown to avoid misidentification.
                return RecognitionResult(
                    name=None,
                    distance=best["avg_sim"],
                    is_known=False,
                )
 
        return RecognitionResult(
            name=best["name"],
            distance=best["avg_sim"],
            is_known=True,
        )
 
    def reload_database(self) -> None:
        """Reload enrolled faculty from the encrypted database."""
        self._db = FaceDatabase()
        self._enrolled = self._db.get_all()
        print(f"[face_recognizer] Reloaded. {len(self._enrolled)} faculty enrolled.")
 
    @property
    def enrolled_count(self) -> int:
        """How many faculty are currently enrolled."""
        return len(self._enrolled)
 
    @property
    def threshold(self) -> float:
        """The current distance threshold."""
        return self._threshold
 
    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update the distance threshold at runtime.
 
        Useful during SD-12 accuracy testing when you want to try
        different thresholds without restarting.
        """
        self._threshold = value
        print(f"[face_recognizer] Threshold updated to {value}")