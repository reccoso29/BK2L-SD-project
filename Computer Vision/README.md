<div align="center">

# 🛡️ Knightro Face Recognition

### Software subsystem for the Knightro interactive animatronic mascot

*Recognizes enrolled UCF faculty to enable personalized greetings*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/status-Sprint%201-yellow.svg)](#-progress-tracker)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey.svg)](#)

</div>

---

## 📖 Overview

This is the software subsystem responsible for the **face recognition pipeline** of the Knightro animatronic — a senior design project at the University of Central Florida. When an enrolled faculty member walks up to Knightro, the system detects their face, looks up their identity from a small encrypted on-device database, and triggers a personalized greeting through the animatronic's speech and motion controllers.

The full pipeline runs **locally on a Raspberry Pi** with no cloud dependency. Privacy and consent are first-class design constraints, not afterthoughts.

---

## 📑 Table of Contents

- [Project Layout](#-project-layout)
- [Setup](#-setup-first-time)
- [Running the Tests](#-running-the-tests)
- [Baseline Benchmarks](#-baseline-benchmarks)
- [Progress Tracker](#-progress-tracker)
- [Known Issues](#-known-issues)
- [References](#-references)

---

## 📁 Project Layout

```
Computer Vision/
├── src/                           # Python modules
│   └── face_detection.py          # SD-7: MediaPipe face detection
├── tests/                         # Manual and automated tests
│   └── test_face_detection.py     # SD-7: webcam demo + FPS baseline
├── models/                        # (auto-created) cached MediaPipe model files
├── data/                          # (auto-created) snapshots, encrypted embeddings
├── requirements.txt               # Python dependencies
└── README.md                      # You are here
```

---

## 🚀 Setup (first time)

### 1. Create a virtual environment

A venv keeps this project's dependencies isolated from your system Python.

```bash
python -m venv .venv
```

Then activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

You'll know it worked when your prompt shows `(.venv)` at the front.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs only the **core** packages (`opencv-python`, `mediapipe`, `numpy`) needed for SD-7 and SD-8. The recognition packages (`face_recognition`, `dlib`, `cryptography`) are commented out and will be enabled when we start SD-9.

> [!TIP]
> See [Known Issues](#-known-issues) below for the reason `dlib` is deferred and how to enable it later.

---

## 🧪 Running the Tests

### SD-7: Face Detection Webcam Demo

Opens your webcam, draws boxes around detected faces, and shows FPS and face count in the corner. This is the **laptop baseline benchmark** that Phase 2 will compare against Pi performance.

```bash
python tests/test_face_detection.py
```

| Key | Action |
|-----|--------|
| `q` | Quit the demo |
| `s` | Save a snapshot of the current frame to `data/` |

> [!NOTE]
> On first run, this will download the MediaPipe face detector model (~230 KB) to `models/blaze_face_short_range.tflite`. After that it runs entirely offline.

> [!WARNING]
> **macOS users:** the first run will trigger a camera permission popup. Click **Allow**. If you're running from a terminal *inside* an editor (VS Code, etc.), the permission prompt is for **that app**, not Terminal — and you may need to fully quit and relaunch the editor after granting permission. See [Known Issues](#-known-issues) for details.

---

## 📊 Baseline Benchmarks

### SD-7 — Face Detection (laptop)

Recorded on the development laptop for the EDS comparison table. Phase 2 will re-run the same test on the Raspberry Pi and add a second column.

| Metric                       | 💻 Laptop (MacBook Air M1) | 🥧 Raspberry Pi (Phase 2) |
|------------------------------|----------------------------|---------------------------|
| Sustained FPS                | **30** (camera-capped)     | _TBD_                     |
| Confidence on frontal face   | **> 0.90**                 | _TBD_                     |
| Robust to ±45° head rotation | ✅ Yes                     | _TBD_                     |
| Hardware acceleration        | Apple M1 GPU               | _TBD_                     |

> [!NOTE]
> **Interpretation:** The laptop sustained the full 30 FPS that the webcam delivers, meaning the detector is **not the bottleneck on the laptop** — it could likely go faster with a higher-frame-rate camera. This is the expected ceiling for laptop performance and gives us a clean reference point for the Pi comparison.

---

## ✅ Progress Tracker

| Ticket  | Description                              | Status                                              |
|---------|------------------------------------------|-----------------------------------------------------|
| SD-6    | Privacy & Consent Policy                 | On Hold                                         |
| SD-7    | Face Detection (laptop)                  | ✅ **Done** — verified on MacBook Air M1, 30 FPS    |
| SD-8    | Face Tracking                            | 🔜 **Next up**                                       |
| SD-9    | Faculty Enrollment Process               | ⬜ Sprint 1 _(needs `dlib` — see Known Issues)_     |
| SD-10   | Encrypted Embeddings Storage             | ⬜ Sprint 1                                          |
| SD-11   | 1:N Recognition Comparison               | ⬜ Sprint 1                                          |
| SD-12   | Accuracy Tests (laptop baseline)         | ⬜ Sprint 1                                          |
| SD-13   | Integration with greeting flow           | ⬜ Sprint 2 _(needs Pi hardware)_                    |
| SD-50   | Performance test plan                    | ⬜ Sprint 1 (plan) / Sprint 2 (execute)              |
| SD-54   | Safety validation plan                   | ⬜ Sprint 1 (plan) / Sprint 2 (execute)              |

---

## ⚠️ Known Issues

<details>
<summary><strong>🔧 dlib / face_recognition won't install on a fresh macOS</strong></summary>

<br>

`face_recognition` depends on `dlib`, which is a C++ library that pip has to compile from source. The compile requires a working C++ toolchain and standard C++ headers (`<string>`, `<iosfwd>`, etc.). On a fresh macOS install those aren't present and the build fails with errors like:

```
fatal error: 'string' file not found
fatal error: 'iosfwd' file not found
ERROR: Failed building wheel for dlib
```

### Why we're sidestepping this for now

The recognition packages aren't needed until SD-9. To keep SD-7 and SD-8 unblocked, the relevant lines in `requirements.txt` are commented out. You can install all the core dependencies and run the face detection demo without ever touching `dlib`.

### The fix (when we get to SD-9)

**1.** Install Apple's command line tools, which provides the missing C++ compiler and headers:

```bash
xcode-select --install
```

A popup will appear; click **Install**. The download is ~3 GB and takes 10–20 minutes depending on your connection.

**2.** Once that finishes, uncomment the two `face_recognition` and `cryptography` lines at the bottom of `requirements.txt`.

**3.** Re-run:

```bash
pip install -r requirements.txt
```

The `dlib` build will now succeed (it'll still take a few minutes — it's a big library — but it won't error out).

> [!IMPORTANT]
> **For Raspberry Pi (Phase 2):** The Pi has its C++ toolchain available by default, so no equivalent step is needed there. However, the `dlib` compile itself takes **30–60 minutes** on a Pi. We just need to plan accordingly when deploying.

</details>

<br>

<details>
<summary><strong>📷 macOS camera permissions can fail silently</strong></summary>

<br>

If you see this even though the script reports `Webcam opened`:

```
WARN: Failed to read frame from webcam, stopping.
```

…that's macOS rejecting the camera read silently because the calling app doesn't have camera permission. Annoyingly, OpenCV reports the camera as "open" even when it can't actually read frames.

### The fix

**1.** Go to **System Settings → Privacy & Security → Camera**.

**2.** Find the terminal app you're using (`Terminal`, `Code`, `iTerm2`, etc.) and make sure it's enabled.

**3.** **Fully quit and relaunch** that app (Cmd+Q, not just close the window). macOS only re-checks permissions when an app fully restarts.

**4.** Reactivate your venv and try the script again.

</details>

---

## 📚 References

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR 2015.*
- Google AI Edge. *MediaPipe Face Detector documentation.* https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
- Geitgey, A. (2020). *`face_recognition` Python library documentation.*
- Hernandez, A. *Knightro Tech Memo 1: Raspberry Pi 4 Computer Vision Benchmarks.*

---

<div align="center">

**🛡️ Charge On!** ⚔️

*Built for the BK2L Senior Design Project at the University of Central Florida*

</div>