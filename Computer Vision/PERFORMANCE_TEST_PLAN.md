# Knightro Face Recognition — Performance Test Plan

**Ticket:** SD-50
**Author:** Angel Hernandez
**Status:** Plan authored (Sprint 1) — execution deferred to Sprint 2 (requires Raspberry Pi 5)
**Related requirements:** TPM 7.5.1, Tech Memo 1

---

## 1. Purpose

This document defines the performance tests for the Knightro face recognition
subsystem. The goal is to verify that the system meets real-time interaction
requirements when running on the target hardware (Raspberry Pi 5, 8 GB).

All tests in this plan will be executed during Sprint 2 once the Raspberry Pi
is available. Laptop baseline numbers from Sprint 1 are included for
comparison.

## 2. Test Environment

| Item | Sprint 1 (Baseline) | Sprint 2 (Target) |
|------|---------------------|--------------------|
| Hardware | MacBook Air M1, 8 GB | Raspberry Pi 5, 8 GB |
| OS | macOS Sequoia | Raspberry Pi OS (64-bit) |
| Python | 3.14 | 3.11+ |
| Camera | Built-in FaceTime HD (720p, 30 FPS) | TBD (USB or Pi Camera Module) |
| Cooling | N/A (passive) | Active cooling fan (required per Tech Memo 1) |
| Power | AC adapter | Official Pi 5 27W USB-C PSU |

## 3. Metrics and Pass/Fail Thresholds

Each metric is derived from project requirements (M3 Concept Design Report
section 7.5.1) and Tech Memo 1 benchmarks.

### 3.1 Face Detection FPS

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Sustained FPS (face detection only) | ≥ 12 FPS | Tech Memo 1: below 12 FPS creates a "floaty" or "disconnected" feeling for users. MediaPipe Lite achieved 12–16 FPS on Pi 4; Pi 5 should match or exceed. |
| Sustained FPS (detection + tracking) | ≥ 12 FPS | Tracking adds negligible overhead (verified on laptop: 0 FPS drop). |

**Laptop baseline:** 30 FPS (camera-capped, detection is not the bottleneck).

### 3.2 Recognition Latency

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Time from "face appears" to "name returned" | ≤ 2000 ms | M3 section 7.5.1 requires facial recognition accuracy ≥70% and motion within 5 seconds. The recognition pipeline must complete well within that window. Includes: detection (1 frame) + tracking stabilization (3 frames) + embedding computation + database comparison. |
| Embedding computation time (single face) | ≤ 500 ms | The ArcFace ONNX model is the heaviest compute step. Must not block the interaction pipeline. |

**How to measure:** The test script timestamps each pipeline stage and reports
per-stage and total latency.

### 3.3 Memory Usage

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Peak RSS memory (full pipeline running) | ≤ 2 GB | Pi 5 has 8 GB total. Other subsystems (conversational AI, TTS, animation control) share the same device. Face recognition should not consume more than 25% of available RAM. |
| Steady-state memory after 10 minutes | No unbounded growth | Detects memory leaks. A slow leak that adds 10 MB/hour would eventually crash the Pi during a multi-hour deployment. |

### 3.4 CPU Temperature Under Load

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| CPU temperature after 10 minutes sustained | ≤ 75°C | Tech Memo 1 flagged 60–70°C as the operating range under vision load. The Pi 5 throttles at 85°C. A 75°C ceiling provides a 10°C margin. |
| Thermal throttling events in 30 minutes | 0 events | Any throttling means FPS will drop unpredictably during interaction. |

**Measurement:** `vcgencmd measure_temp` polled every 10 seconds during the
test run, logged to a CSV file.

### 3.5 System Uptime / Stability

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Continuous operation without crash | ≥ 60 minutes | Knightro will operate unattended for hours. The face recognition pipeline must not crash, hang, or degrade over an extended session. |

## 4. Test Procedures

### Test P-1: Face Detection FPS on Pi

**Objective:** Measure sustained detection FPS on the target hardware.

**Steps:**
1. Boot the Pi, activate the venv, ensure the camera is connected.
2. Run `python tests/test_face_detection.py`.
3. Position a person approximately 1 meter from the camera (Knightro's interaction zone).
4. Let the test run for 60 seconds after the FPS counter stabilizes.
5. Record the sustained FPS value from the on-screen display.
6. Repeat with 0, 1, 2, and 3 faces in frame.

**Pass criteria:** ≥ 12 FPS with 1 face, ≥ 10 FPS with 2 faces.

**Record:**
| Condition | FPS | Pass/Fail |
|-----------|-----|-----------|
| 0 faces | | |
| 1 face at 1m | | |
| 2 faces | | |
| 3 faces | | |

---

### Test P-2: Detection + Tracking FPS on Pi

**Objective:** Verify tracking does not degrade FPS.

**Steps:**
1. Run `python tests/test_face_tracking.py`.
2. Position 1 person at 1 meter.
3. Record FPS after 30 seconds.
4. Have a second person walk in. Record FPS.
5. Have person 1 leave and return. Verify new track ID assigned.

**Pass criteria:** FPS within 1 FPS of detection-only result.

---

### Test P-3: Recognition Latency on Pi

**Objective:** Measure end-to-end time from face appearance to name returned.

**Steps:**
1. Enroll at least 1 faculty member on the Pi.
2. Run `python tests/test_face_recognition.py`.
3. Have the enrolled person walk into frame from off-screen.
4. Time from the moment they enter frame to the moment their name appears on screen.
5. Repeat 10 times and compute the average.

**Pass criteria:** Average latency ≤ 2000 ms, no single trial > 3000 ms.

**Record:**
| Trial | Latency (ms) | Correct ID? |
|-------|-------------|-------------|
| 1 | | |
| ... | | |
| 10 | | |
| **Average** | | |

---

### Test P-4: Memory Usage

**Objective:** Verify the pipeline fits within the Pi's memory budget.

**Steps:**
1. Before starting: record baseline memory with `free -m`.
2. Start the recognition demo.
3. Record memory usage at: startup, 1 min, 5 min, 10 min.
4. Check for unbounded growth by comparing the 1-min and 10-min values.

**Pass criteria:** Peak RSS ≤ 2 GB. Growth between 1-min and 10-min ≤ 50 MB.

---

### Test P-5: CPU Temperature Under Load

**Objective:** Verify the cooling solution prevents thermal throttling.

**Steps:**
1. Ensure the active cooling fan is installed and running.
2. Start the recognition demo with 1 enrolled person.
3. Run for 30 minutes with continuous interaction (people walking up periodically).
4. Log `vcgencmd measure_temp` every 10 seconds to a CSV.
5. Check for any temperature readings above 75°C.
6. Check system log for throttling events: `vcgencmd get_throttled`.

**Pass criteria:** Max temp ≤ 75°C. Zero throttling events.

---

### Test P-6: Extended Stability Run

**Objective:** Verify the system doesn't crash or degrade over time.

**Steps:**
1. Start the full recognition pipeline.
2. Let it run for 60 minutes with periodic interaction (someone walks up every 5 minutes).
3. At 60 minutes, verify: FPS is still within pass criteria, no error messages in console, memory has not grown unboundedly, the system correctly recognizes enrolled faculty.

**Pass criteria:** No crashes, no hangs, FPS within 2 FPS of the 1-minute reading.

## 5. Laptop Baseline Results (Sprint 1)

These numbers were recorded during SD-7 and SD-8 testing and serve as the
upper bound for comparison.

| Metric | Laptop Result |
|--------|---------------|
| Face detection FPS | 30 (camera-capped) |
| Detection + tracking FPS | 30 (no degradation) |
| Detection confidence (frontal) | > 0.90 |
| Robust to ±45° rotation | Yes |
| Hardware acceleration | Apple M1 GPU (Metal) |
| Recognition accuracy (SD-12) | 95% (TPR 100%, TNR 90%) |
| Recognition threshold | 0.970 (cosine similarity) |

## 6. Reporting

After execution, results will be compiled into a table comparing laptop
baseline vs. Pi performance and included in the EDS. Any metric that fails
will generate a Jira bug ticket with priority based on the margin of failure.
