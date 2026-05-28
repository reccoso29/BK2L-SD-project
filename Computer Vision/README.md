<div align="center">

#  Knightro Face Recognition

### Software subsystem for the Knightro interactive animatronic mascot

*Recognizes enrolled UCF faculty to enable personalized greetings*

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![dlib](https://img.shields.io/badge/dlib-19.24.6-red.svg)](http://dlib.net/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/status-Sprint%202-yellow.svg)](#-progress-tracker)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey.svg)](#)

</div>

---

## 📖 Overview

This is the software subsystem responsible for the **face recognition pipeline** of the Knightro animatronic — a senior design project at the University of Central Florida. When an enrolled faculty member walks up to Knightro, the system detects their face, looks up their identity from a small encrypted on-device database, and triggers a personalized greeting through the animatronic's speech and motion controllers.

The full pipeline runs **locally on a Raspberry Pi** with no cloud dependency. Privacy and consent are first-class design constraints, not afterthoughts.

---

## 📑 Table of Contents

- [Technology Journey](#-technology-journey)
- [Project Layout](#-project-layout)
- [Setup](#-setup-first-time)
- [Enrollment](#-enrollment)
- [Running the Tests](#-running-the-tests)
- [Baseline Benchmarks](#-baseline-benchmarks)
- [Progress Tracker](#-progress-tracker)
- [References](#-references)

---

## 🔄 Technology Journey

### Sprint 1: dlib → ONNX Runtime

During Sprint 1, the original plan was to use the `face_recognition` library (which depends on `dlib`) for computing face embeddings. However, `dlib` 20.0.1 could not compile on macOS 15 (Sequoia) + Apple Silicon due to missing C++ standard library headers and a removed macOS header (`<fp.h>`) in dlib's bundled libpng. Multiple workarounds were attempted — none fully resolved the build.

The team pivoted to **ONNX Runtime** with an **ArcFace ResNet100** model, which provides the same functionality with zero compilation required.

### Sprint 2: ONNX Runtime → dlib (via conda)

During integration testing, the ArcFace/ONNX model revealed a critical limitation: **cosine similarity scores for different people were too close together** (all scoring 0.97–0.99), making it nearly impossible to distinguish enrolled individuals. The margin between the correct match and the second-best candidate was as low as 0.003, causing frequent misidentification.

After investigation, the team discovered that **dlib 19.24.6** installs cleanly via conda (`conda install -c conda-forge dlib`) on macOS, Windows, and Linux — bypassing all the compilation issues from Sprint 1. Testing confirmed dramatically better separation with dlib's 128-dimensional Euclidean distance embeddings.

| | Sprint 1 (ONNX/ArcFace) | Sprint 2 (dlib) |
|---|---|---|
| **Embedding dim** | 512 | 128 |
| **Distance metric** | Cosine similarity (higher = better) | Euclidean distance (lower = better) |
| **Same-person distance** | 0.97–0.99 | 0.17–0.44 |
| **Different-person distance** | 0.96–0.98 | 0.70–0.95 |
| **Separation gap** | ~0.003 (unusable) | ~0.30+ (excellent) |
| **Install method** | `pip install onnxruntime` | `conda install -c conda-forge dlib` |
| **Recognition reliability** | Frequent confusion between enrolled people | One-shot recognition, no confusion |

The old ONNX database is preserved as a backup (`data/face_embeddings.enc`). The active system uses the dlib database (`data/dlib_face_embeddings.enc`).

---

## 📁 Project Layout

```
Computer Vision/
├── src/                                # Python modules
│   ├── face_detection.py               # SD-7: MediaPipe BlazeFace detection
│   ├── face_tracker.py                 # SD-8: IoU-based multi-face tracking
│   ├── face_embedder.py                # SD-9: dlib 128-dim face embeddings
│   ├── face_database.py                # SD-10: Fernet-encrypted embedding storage
│   ├── face_recognizer.py              # SD-11: Euclidean distance recognition
│   └── enroll.py                       # SD-9: Faculty enrollment tool (webcam + photo)
├── tests/                              # Manual and automated tests
│   ├── test_face_detection.py          # SD-7: webcam demo + FPS baseline
│   ├── test_face_tracking.py           # SD-8: multi-face tracking demo
│   ├── test_face_recognition.py        # SD-11: real-time recognition demo
│   └── test_recognition_accuracy.py    # SD-12: accuracy benchmarks
├── data/                               # (auto-created, gitignored except DB)
│   ├── dlib_face_embeddings.enc        # Active dlib enrollment database
│   ├── .dlib_encryption_key            # Encryption key for dlib database
│   ├── face_embeddings.enc             # Legacy ONNX database (backup)
│   └── .encryption_key                 # Legacy ONNX encryption key (backup)
├── models/                             # (auto-created) cached MediaPipe model
├── Privacy_policy.md                   # SD-6: Biometric data privacy policy
├── PERFORMANCE_TEST_PLAN.md            # SD-50: Performance testing plan
├── requirements.txt                    # Python dependencies
└── README.md                           # You are here
```

---

## 🚀 Setup (first time)

### Option A: conda (recommended — required for dlib)

```bash
# Install Miniforge if you don't have conda
# macOS: brew install --cask miniforge
# Windows: download from https://conda-forge.org/download/

# Create environment
conda create -n cv_env python=3.12 -y
conda activate cv_env

# Install dlib and OpenCV via conda (pre-built, no compilation)
conda install -c conda-forge dlib=19.24.6 face_recognition opencv -y

# Install remaining dependencies
pip install mediapipe onnxruntime cryptography
```

### Option B: pip only (no dlib — uses legacy ONNX model)

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

> [!NOTE]
> Option B uses the ONNX/ArcFace model which has limited separation between enrolled people. Option A (conda + dlib) is strongly recommended for reliable recognition.

---

## 👤 Enrollment

### Enroll a faculty member (webcam)

```bash
python src/enroll.py --enroll --name "Dr. Smith"
```

The webcam opens with on-screen guidance for capturing 10 images from different angles (straight, left, right, up, down, 45°). Press SPACE to capture each frame or 'a' for auto-capture mode.

### Enroll from a photo

```bash
python src/enroll.py --enroll --name "Dr. Smith" --photo headshot.jpg
```

### List enrolled faculty

```bash
python src/enroll.py --list
```

### Remove a faculty member

```bash
python src/enroll.py --remove --name "Dr. Smith"
```

### Test separation between enrolled people

```bash
python src/enroll.py --test
```

This prints Euclidean distances between all enrolled pairs. You want avg distance > 0.6 (marked `[GOOD]`).

---

## 🧪 Running the Tests

### SD-7: Face Detection

```bash
python tests/test_face_detection.py
```

Opens webcam, draws bounding boxes, shows FPS. Press `q` to quit.

### SD-8: Face Tracking

```bash
python tests/test_face_tracking.py
```

Shows persistent track IDs across frames with color-coded boxes.

### SD-11: Face Recognition (real-time)

```bash
python tests/test_face_recognition.py
```

Shows recognized names with green boxes, "Unknown" for non-enrolled faces. Controls: `q` quit, `s` snapshot, `r` reload database, `+/-` adjust threshold.

### SD-12: Accuracy Benchmarks

```bash
python tests/test_recognition_accuracy.py
```

---

## 📊 Baseline Benchmarks

### Face Detection (SD-7)

| Metric                       | 💻 MacBook Air M1 | 🥧 Raspberry Pi 5 (Phase 2) |
|------------------------------|--------------------|-----------------------------|
| Sustained FPS                | **30** (camera-capped) | _TBD_                   |
| Confidence on frontal face   | **> 0.90**         | _TBD_                       |
| Robust to ±45° head rotation | ✅ Yes             | _TBD_                       |

### Face Recognition (SD-11/12) — dlib

| Metric                           | Result            |
|----------------------------------|-------------------|
| Same-person avg distance         | **0.42–0.46**     |
| Different-person avg distance    | **0.70–0.91**     |
| Separation gap                   | **0.25–0.30+**    |
| Match threshold                  | **0.50**          |
| One-shot recognition             | ✅ Yes            |
| Cross-person confusion           | ❌ None observed  |

---

## ✅ Progress Tracker

| Ticket | Description                    | Status |
|--------|--------------------------------|--------|
| SD-6   | Privacy & Consent Policy       | ✅ Done |
| SD-7   | Face Detection (laptop)        | ✅ Done — 30 FPS on MacBook Air M1 |
| SD-8   | Face Tracking                  | ✅ Done — IoU-based multi-face tracker |
| SD-9   | Faculty Enrollment Process     | ✅ Done — webcam + photo enrollment with quality checks |
| SD-10  | Encrypted Embeddings Storage   | ✅ Done — Fernet encryption, separate dlib database |
| SD-11  | 1:N Recognition Comparison     | ✅ Done — dlib Euclidean distance, one-shot recognition |
| SD-12  | Accuracy Tests (laptop)        | ✅ Done — 0.30+ separation gap between enrolled people |
| SD-13  | Integration with greeting flow | 🔄 In Progress — integrated demo with TTS + interaction pipeline |
| SD-50  | Performance test plan          | ✅ Done (plan) / ⬜ Sprint 2 (execute on Pi) |
| SD-54  | Safety validation plan         | ✅ Done (plan) / ⬜ Sprint 2 (execute on Pi) |

---

## 🔒 Privacy Architecture

All face data is processed and stored **locally on-device**:

- **No raw images stored** — only 128-dimensional numerical embeddings
- **Fernet encryption** (AES-128-CBC) for the embedding database
- **Enrolled faculty have full data removal rights** (`enroll.py --remove`)
- **No cloud transmission** — all processing stays on the Raspberry Pi
- **Separate encryption keys** for dlib and legacy ONNX databases

See `Privacy_policy.md` for the full biometric data privacy policy.

---

## 📚 References

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR 2015.*
- King, D. E. (2009). Dlib-ml: A machine learning toolkit. *Journal of Machine Learning Research, 10*, 1755-1758.
- Google AI Edge. *MediaPipe Face Detector documentation.* https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
- Geitgey, A. *face_recognition library documentation.* https://github.com/ageitgey/face_recognition
- Hernandez, A. *Knightro Tech Memo 1: Raspberry Pi 4 Computer Vision Benchmarks.*

---

<div align="center">

**🛡️ Charge On!** ⚔️

</div>