# RDK X5 Vision Pipeline
> AI Vision Pipeline for Retail Analytics on RDK X5 + Pi Camera Module 2  
> Powered by D-Robotics BPU (Bayes-E architecture)

---

## 🎯 Use Case
Retail shelf analytics — detecting people walking past, measuring attention (head turns), dwell time, and demographic insights (age/gender) in real time on edge hardware.

---

## 📋 Table of Contents
- [Hardware](#hardware)
- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
  - [YOLOv8s — Person Detection](#1-yolov8s--person-detection)
  - [YOLOv8n-face — Face Detection](#2-yolov8n-face--face-detection)
  - [YOLOv8s-pose — Pose Estimation](#3-yolov8s-pose--pose-estimation)
  - [InsightFace GenderAge — Age & Gender](#4-insightface-genderage--age--gender)
- [Calibration Details](#calibration-details)
- [Performance Summary](#performance-summary)
- [How to Use on Board](#how-to-use-on-board)
- [Project Structure](#project-structure)
- [Toolchain Info](#toolchain-info)

---

## Hardware

| Component | Details |
|-----------|---------|
| Board | RDK X5 |
| BPU Architecture | Bayes-E |
| BPU Cores | 2 |
| Camera | Pi Camera Module 2 (MIPI CSI) |
| Camera Output Format | NV12 |
| OS | Ubuntu 22.04 aarch64 |
| Runtime Version | hrt_model_exec v1.24.5 |

---

## Pipeline Overview

```
Pi Camera Module 2
  NV12 stream (640x640)
        │
        ▼
┌───────────────────────────────┐
│  Stage 1: Detection           │
│  ├─ YOLOv8s      → Person BBox + Track ID │
│  └─ YOLOv8n-face → Face BBox + Landmarks  │
└───────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────────────┐   ┌──────────────────────┐
│   Stage 2a   │   │      Stage 2b        │
│ YOLOv8s-pose │   │  InsightFace         │
│ 17 Keypoints │   │  Age + Gender        │
│ Body pose    │   │  from face crop      │
└──────────────┘   └──────────────────────┘
        │                   │
        └────────┬───────────┘
                 ▼
        📊 Analytics Output
        - Traffic count
        - Attention rate (% who looked)
        - Dwell time per person
        - Age/Gender demographics
        - Pose-based engagement score
```

---

## Models

### 1. YOLOv8s — Person Detection

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/yolov8s.bin` |
| **Task** | Object Detection — Person class (COCO) |
| **Architecture** | YOLOv8 Small (CSPDarknet + PAN + Detect head) |
| **Parameters** | 11.2M |
| **GFLOPs** | 28.6 |
| **Input Name** | `images` |
| **Input Shape** | `1 × 3 × 640 × 640` |
| **Input Type (Runtime)** | NV12 — direct from camera pyramid |
| **Input Type (Training)** | RGB, NCHW |
| **Output Shape** | `1 × 84 × 8400` |
| **Output Format** | 84 = 4 bbox coords + 80 class scores; 8400 anchors |
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 2 (main BPU + post-process BPU) |
| **Cosine Similarity** | **0.999772** 🟢 |
| **BPU FPS** | ~88 FPS |
| **BPU Latency** | ~11.3 ms |
| **File Size (.bin)** | 13 MB |
| **DataType** | int8 (BPU), float32 (CPU nodes) |

**CPU nodes (ARM):**
- `/model.22/Reshape` × 3 — reshape DFL output
- `/model.22/Concat` — concat predictions
- `/model.22/dfl/Reshape`, `/Transpose`, `/Softmax` — DFL decoding

**Post-processing parameters:**
```python
conf_threshold = 0.25
iou_threshold  = 0.45
classes        = [0]  # person only
input_size     = (640, 640)
```

---

### 2. YOLOv8n-face — Face Detection

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/yolov8n-face.bin` |
| **Task** | Face Detection |
| **Architecture** | YOLOv8 Nano |
| **Parameters** | 3.2M |
| **GFLOPs** | 8.7 |
| **Input Name** | `images` |
| **Input Shape** | `1 × 3 × 640 × 640` |
| **Input Type (Runtime)** | NV12 — direct from camera pyramid |
| **Input Type (Training)** | RGB, NCHW |
| **Output Shape** | `1 × 84 × 8400` |
| **Output Format** | 84 = 4 bbox + 80 class scores; 8400 anchors |
| **Normalization** | `pixel / 255.0` |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Cosine Similarity** | **0.999706** 🟢 |
| **BPU FPS** | ~100+ FPS |
| **File Size (.bin)** | 4.9 MB |
| **DataType** | int8 (BPU), float32 (CPU nodes) |

**Post-processing parameters:**
```python
conf_threshold = 0.4   # higher threshold for face
iou_threshold  = 0.5
input_size     = (640, 640)
```

**⚠️ Note:** Base weights are YOLOv8n (general detector), not face-specific.  
For production, replace with dedicated face weights such as `yolov8-face` from akanametov/yolo-face.

---

### 3. YOLOv8s-pose — Pose Estimation

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/yolov8s-pose.bin` |
| **Task** | Human Pose Estimation — 17 keypoints (COCO) |
| **Architecture** | YOLOv8 Small Pose |
| **Parameters** | 11.6M |
| **GFLOPs** | 30.2 |
| **Input Name** | `images` |
| **Input Shape** | `1 × 3 × 640 × 640` |
| **Input Type (Runtime)** | NV12 — direct from camera pyramid |
| **Input Type (Training)** | RGB, NCHW |
| **Output Shape** | `1 × 56 × 8400` |
| **Output Format** | 56 = 4 bbox + 1 obj_score + 17×3 keypoints (x, y, conf) |
| **Normalization** | `pixel / 255.0` |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O1 ⚠️ (O3 causes hbdk-cc internal crash on this model) |
| **Compile Mode** | latency |
| **Subgraphs** | 2 |
| **Cosine Similarity** | **0.999903** 🟢 |
| **BPU FPS (subgraph 0)** | ~21 FPS |
| **BPU FPS (subgraph 1)** | ~121 FPS |
| **BPU Latency (subgraph 0)** | ~47.3 ms |
| **BPU Latency (subgraph 1)** | ~8.2 ms |
| **File Size (.bin)** | 15 MB |
| **DataType** | int8 / int16 (BPU), float32 (CPU nodes) |

**17 Keypoints (COCO order):**
```
0: nose          1: left_eye       2: right_eye
3: left_ear      4: right_ear      5: left_shoulder
6: right_shoulder 7: left_elbow   8: right_elbow
9: left_wrist   10: right_wrist   11: left_hip
12: right_hip   13: left_knee     14: right_knee
15: left_ankle  16: right_ankle
```

**Engagement detection logic:**
```python
# Person is "looking at shelf" if:
# nose keypoint is visible AND facing forward
nose_conf    = keypoints[0][2]
l_shoulder_x = keypoints[5][0]
r_shoulder_x = keypoints[6][0]
shoulder_width = abs(r_shoulder_x - l_shoulder_x)
# if shoulder_width > threshold → person facing camera
```

**Known Issues:**
- O3 causes compiler crash (`calculate size exceed peak dim`) → must use O1
- O1 is ~4× slower than O3 would be; upgrade hbdk when fixed

---

### 4. InsightFace GenderAge — Age & Gender

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/genderage.bin` |
| **Source** | InsightFace `buffalo_l` pack — `genderage.onnx` |
| **Task** | Age estimation + Gender classification |
| **Architecture** | MobileNet-based lightweight classifier |
| **Input Name** | `data` |
| **Input Shape** | `1 × 3 × 96 × 96` |
| **Input Type (Runtime)** | NV12 — cropped face region |
| **Input Type (Training)** | RGB, NCHW |
| **Output Shape** | `1 × 3` |
| **Output Format** | `[gender_prob, age_value, padding]` |
| **Output Decoding** | `gender = 'Male' if output[0] > 0.5 else 'Female'`; `age = output[1] * 100` |
| **Normalization** | `pixel / 255.0` |
| **Original ONNX Opset** | 12 (downgraded to 11 for compatibility) |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 1 (almost fully on BPU) |
| **Cosine Similarity** | **0.995687** 🟢 |
| **BPU FPS** | **~10,116 FPS** 🚀 |
| **BPU Latency** | **~98.8 microseconds** 🚀 |
| **File Size (.bin)** | 579 KB |
| **DataType** | int8 (BPU), float32 (CPU: final Concat + Reshape) |

**Usage — crop face before inference:**
```python
# 1. Get face bbox from YOLOv8n-face
x1, y1, x2, y2 = face_bbox

# 2. Crop and resize
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (96, 96))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

# 3. Infer
output = model.infer(face_crop)  # shape: [1, 3]
gender = 'Male' if output[0][0] > 0.5 else 'Female'
age    = int(output[0][1] * 100)
```

**Accuracy benchmarks (InsightFace buffalo_l):**
- Gender accuracy: ~97%
- Age MAE: ~3-4 years

---

## Calibration Details

| Parameter | Value |
|-----------|-------|
| **Total Images** | 54 (34 COCO val2017 + 20 random/picsum) |
| **640×640 format** | Used for yolov8s, yolov8n-face, yolov8s-pose |
| **96×96 format** | Used for genderage |
| **Pixel Format** | uint8, RGB channel order |
| **File Format** | Raw binary `.bin` (H × W × C, uint8) |
| **Method Selected** | max-percentile (percentile = 0.99995) |
| **Per-Channel Quantization** | False |
| **Asymmetric Quantization** | False |
| **Batch Size** | 8 (auto-reset to 1 for pose/genderage) |

---

## Performance Summary

| Model | Task | Cosine Sim | FPS | Latency | Size |
|-------|------|-----------|-----|---------|------|
| yolov8s.bin | Person Detection | 0.999772 | ~88 | ~11.3 ms | 13 MB |
| yolov8n-face.bin | Face Detection | 0.999706 | ~100+ | <10 ms | 4.9 MB |
| yolov8s-pose.bin | Pose Estimation | 0.999903 | ~21 | ~47.3 ms | 15 MB |
| genderage.bin | Age + Gender | 0.995687 | ~10,116 | ~0.1 ms | 579 KB |

> All models run on BPU (Bayes-E). CPU nodes are minimal and run on ARM Cortex-A55.

---

## How to Use on Board

### 1. Copy models to board
```bash
scp models/bin/*.bin supersensor@172.20.10.2:/home/supersensor/models/
```

### 2. Check model info
```bash
ssh supersensor@172.20.10.2
hrt_model_exec model_info --model_file=models/yolov8s.bin
```

### 3. Benchmark latency (single thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=1 \
  --frame_count=200
```

### 4. Benchmark FPS (multi thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=8 \
  --frame_count=200
```

### 5. Disable CPU frequency scaling for best performance
```bash
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

---

## Project Structure

```
rdkx5_vision/
├── models/
│   ├── onnx/
│   │   ├── yolov8s.onnx              (43 MB)
│   │   ├── yolov8n-face.onnx         (12 MB)
│   │   ├── yolov8s-pose.onnx         (45 MB)
│   │   └── insightface/
│   │       ├── genderage.onnx        (1.3 MB) — original opset 12
│   │       ├── genderage_op11.onnx   (1.3 MB) — converted opset 11
│   │       ├── det_10g.onnx          (17 MB)  — InsightFace detector
│   │       └── 2d106det.onnx         (4.8 MB) — 106 landmarks
│   ├── bin/                          ← Ready to deploy on RDK X5
│   │   ├── yolov8s.bin               (13 MB)
│   │   ├── yolov8n-face.bin          (4.9 MB)
│   │   ├── yolov8s-pose.bin          (15 MB)
│   │   └── genderage.bin             (579 KB)
│   └── configs/
│       ├── yolov8s.yaml
│       ├── yolov8n-face.yaml
│       ├── yolov8s-pose.yaml
│       └── genderage.yaml
├── scripts/
│   ├── export_yolo.py                ← Step 1: Export PyTorch → ONNX
│   ├── convert_model.sh              ← Step 2: Convert ONNX → .bin
│   └── test_inference.sh             ← Step 3: Test on board
├── src/pipeline/                     ← C++ pipeline (coming soon)
├── data/
│   ├── calibration/                  ← Raw images (.jpg)
│   ├── calibration_processed/        ← 640×640 uint8 binary
│   └── calibration_processed_96/     ← 96×96 uint8 binary (for genderage)
└── docs/
    └── 01_setup.md
```

---

## Toolchain Info

| Tool | Version |
|------|---------|
| hb_mapper | 1.24.3 |
| hbdk | 3.49.15 |
| hbdk runtime | 3.15.55.0 |
| horizon_nn | 1.1.0 |
| Docker Image | openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 |
| PyTorch | 2.11.0 |
| Ultralytics | 8.4.39 |
| ONNX | 1.21.0 |
| InsightFace | 0.7.3 |
| Python | 3.12.1 |

---

## License
Models are based on open-source weights:
- YOLOv8: [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- InsightFace: [MIT License](https://github.com/deepinsight/insightface/blob/master/LICENSE)