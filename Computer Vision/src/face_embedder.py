"""
This module takes a cropped face image and produces a numeric embedding vector (512 floating-point numbers) that mathematically represents that face. 
Two photos of the same person produce embeddings that are close together; two different people produce embeddings that are far apart.

The model used is ArcFace (ResNet100 backbone):
  - Input: 112x112 RGB face image, properly aligned
  - Output: 512-dimensional embedding vector
  - Accuracy: state-of-the-art on LFW benchmark

IMPORTANT: Face alignment dramatically improves accuracy. This moduleincludes an alignment step that uses 
the eye positions from the facecrop to rotate and center the face before computing the embedding.
Without alignment, two photos of the same person from different angles can produce embeddings that are far apart, making recognition unreliable.

"""

from pathlib import Path
from typing import Optional, Tuple
import urllib.request

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for face embeddings. "
        "Install it with: pip install onnxruntime"
    )


############################ Model file management ###########################

_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/arcface/model/arcfaceresnet100-8.onnx"
)

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_recognition_model.onnx"
)

# Input dimensions expected by the model.
_INPUT_SIZE = (112, 112)  # width, height


def get_model_path(model_path: Optional[Path] = None) -> Path:
    """Return path to the face embedding model, downloading if missing."""
    path = model_path if model_path is not None else _DEFAULT_MODEL_PATH

    if not path.exists():
        print(f"[face_embedder] Model not found at {path}, downloading...")
        print(f"[face_embedder] This is a one-time download (~120 MB)...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, str(path))
        print(f"[face_embedder] Model downloaded to {path}")

    return path


# ########################### Face alignment ###########################

# Face recognition models are trained on ALIGNED faces — images where the
# eyes are at consistent horizontal positions and the face is centered.
# If we feed in a tilted or off-center face, the embedding quality degrades
# significantly and two photos of the SAME person can look different.
#
# Alignment steps:
#   1. Detect the eye positions in the face crop using a simple eye cascade.
#   2. Compute the angle between the eyes.
#   3. Rotate the image so the eyes are horizontal.
#   4. Crop and resize to 112x112.
#
# If eye detection fails (e.g., person wearing sunglasses), we fall back
# to a simple center-crop without rotation.

# Load OpenCV's pre-trained eye detector (Haar cascade).
# This is a lightweight classical detector — not a neural network.
# It ships with OpenCV, so no extra download needed.

_EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
_eye_cascade = cv2.CascadeClassifier(_EYE_CASCADE_PATH)


def _detect_eyes(face_gray: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Detect the two eyes in a grayscale face image.

    Returns:
        A tuple of ((left_eye_x, left_eye_y), (right_eye_x, right_eye_y)),
        or None if two eyes couldn't be found.
    """
    eyes = _eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
    )

    if len(eyes) < 2:
        return None

    # Sort by x-coordinate to identify left and right eyes.
    # "Left eye" = the eye on the LEFT side of the image (viewer's perspective).
    eyes_sorted = sorted(eyes, key=lambda e: e[0])

    # Get the center of each eye rectangle.
    left_eye = eyes_sorted[0]
    right_eye = eyes_sorted[1]

    left_center = (left_eye[0] + left_eye[2] // 2,
                   left_eye[1] + left_eye[3] // 2)
    right_center = (right_eye[0] + right_eye[2] // 2,
                    right_eye[1] + right_eye[3] // 2)

    return (left_center, right_center)


def align_face(face_bgr: np.ndarray) -> np.ndarray:
    """Align a face crop so the eyes are horizontal and centered.

    This is the key preprocessing step that makes embeddings consistent
    across different head angles and camera positions.

    Args:
        face_bgr: A BGR face crop of any size.

    Returns:
        A 112x112 BGR image with the face aligned and centered.
    """
    h, w = face_bgr.shape[:2]

    # Convert to grayscale for eye detection.
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    eyes = _detect_eyes(gray)

    if eyes is not None:
        left_eye, right_eye = eyes

        # Compute the angle between the eyes.
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Compute the center point between the eyes.
        eye_center = (int((left_eye[0] + right_eye[0]) // 2),
                      int((left_eye[1] + right_eye[1]) // 2))

        # Rotate the image to make the eyes horizontal.
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
        aligned = cv2.warpAffine(face_bgr, rotation_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    else:
        # Eye detection failed — use the image as-is.
        aligned = face_bgr

    # Resize to the model's expected input size.
    result = cv2.resize(aligned, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    return result


############################ Preprocessing for ONNX model ###########################

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR face crop into the format the ONNX model expects.

    This includes alignment, color conversion, normalization, and
    tensor reshaping.

    Args:
        face_bgr: A BGR image (numpy array, shape HxWx3) containing just
                  the face, already cropped from the full frame.

    Returns:
        A numpy array of shape (1, 3, 112, 112) with float32 values
        normalized to [-1, 1], ready to feed into the ONNX model.
    """
    # Step 1: Align the face (rotate so eyes are horizontal, resize to 112x112).
    face_aligned = align_face(face_bgr)

    # Step 2: BGR -> RGB.
    face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)

    # Step 3: Convert to float and normalize to [-1, 1].
    face_float = face_rgb.astype(np.float32) / 255.0
    face_normalized = (face_float - 0.5) / 0.5

    # Step 4: Transpose from HxWxC to CxHxW (channels first).
    face_transposed = face_normalized.transpose(2, 0, 1)

    # Step 5: Add batch dimension.
    face_batch = np.expand_dims(face_transposed, axis=0)

    return face_batch

############################ The embedder class ###########################

class FaceEmbedder:
    """Computes face embeddings using an ONNX model with face alignment.

    Usage:
        embedder = FaceEmbedder()
        embedding = embedder.embed(face_crop_bgr)
        # embedding is a numpy array of shape (512,)
    """

    def __init__(self, model_path: Optional[Path] = None):
        """Create a face embedder.

        Args:
            model_path: Optional path to a .onnx face recognition model.
                        If None, downloads/uses the default ArcFace model.
        """
        resolved_path = get_model_path(model_path)

        self._session = ort.InferenceSession(
            str(resolved_path),
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        output_shape = self._session.get_outputs()[0].shape
        self._embedding_dim = output_shape[-1] if output_shape else None

        print(f"[face_embedder] Model loaded. Input: {self._input_name}, "
              f"Output dim: {self._embedding_dim}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Compute the embedding vector for a single face image.

        Includes automatic face alignment for best accuracy.

        Args:
            face_bgr: A BGR image containing the cropped face.
                      Can be any size — alignment and resizing happen internally.

        Returns:
            A 1D numpy array of length 512 (the embedding vector).
            L2-normalized to unit length.
        """
        input_tensor = preprocess_face(face_bgr)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: input_tensor},
        )

        embedding = outputs[0].squeeze()

        # L2-normalize.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @property
    def embedding_dim(self) -> Optional[int]:
        """The dimensionality of the embedding vectors this model produces."""
        return self._embedding_dim