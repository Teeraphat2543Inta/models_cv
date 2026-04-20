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

#---

### 5. Lightweight Head Pose Estimation — Yaw / Pitch / Roll

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/headpose.bin` |
| **Source** | [Shaw-git/Lightweight-Head-Pose-Estimation](https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation) via PINTO Model Zoo |
| **Task** | Head Pose Estimation — yaw, pitch, roll angles |
| **Architecture** | Lightweight MobileNet-based regression network |
| **Input Name** | `input` |
| **Input Shape** | `1 × 3 × 224 × 224` |
| **Input Type (Runtime)** | NV12 — cropped face region |
| **Input Type (Training)** | RGB, NCHW |
| **Output Names** | `roll`, `yaw`, `pitch` |
| **Output Shape** | `[1]` each — scalar angle in degrees |
| **Output Range** | yaw: ±90°, pitch: ±90°, roll: ±90° |
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **Producer** | PyTorch v1.11.0 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 4 (main BPU + 3 post-process BPU) |
| **Cosine Similarity (roll)** | **1.000000** 🔥 |
| **Cosine Similarity (yaw)** | **1.000000** 🔥 |
| **Cosine Similarity (pitch)** | **1.000000** 🔥 |
| **BPU FPS (main subgraph)** | ~6,878 FPS |
| **BPU Latency (main subgraph)** | ~145 microseconds |
| **File Size (.bin)** | 454 KB |
| **Calibration Dataset** | 54 images at 224×224 (COCO val2017 + random) |
| **Calibration Method** | kl (num_bin=1024, max_num_bin=16384) |
| **DataType** | int8 (BPU), float32 (CPU: Softmax × 3) |

**CPU Nodes (run on ARM CPU):**
- `Softmax_140` — softmax for yaw bins
- `Softmax_144` — softmax for pitch bins
- `Softmax_148` — softmax for roll bins

**Angle Interpretation:**
```
yaw   > 0° → head turning right
yaw   < 0° → head turning left
pitch > 0° → head tilting up
pitch < 0° → head tilting down
roll  > 0° → head tilting right
roll  < 0° → head tilting left
```

**Attention Detection Logic:**
```python
# Crop face first using yolov8n-face output
x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (224, 224))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

# Run inference
output = model.infer(face_crop)
yaw   = output['yaw']    # degrees
pitch = output['pitch']  # degrees
roll  = output['roll']   # degrees

# Determine if person is looking at shelf
looking_at_shelf = abs(yaw) < 30 and abs(pitch) < 30
```

---

### 6. OSNet Re-ID — Person Re-Identification

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/osnet_reid.bin` |
| **Source** | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) via PINTO Model Zoo 429_OSNet |
| **Task** | Person Re-Identification — extract 512-dim feature vector per person |
| **Architecture** | OSNet x1.0 (Omni-Scale Network) |
| **Training Dataset** | MSMT17 — large-scale person Re-ID dataset (126,441 images, 4,101 identities) |
| **Input Name** | `base_image` |
| **Input Shape** | `1 × 3 × 256 × 128` |
| **Input Type (Runtime)** | NV12 — cropped person region |
| **Input Type (Training)** | RGB, NCHW |
| **Output Name** | `feature` |
| **Output Shape** | `1 × 512` |
| **Output Format** | 512-dimensional L2-normalized feature vector |
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **Producer** | PyTorch v1.10 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 1 (fully on BPU except final Reshape) |
| **Cosine Similarity** | **0.985353** 🟢 |
| **BPU FPS** | ~868 FPS |
| **BPU Latency** | ~1.152 ms |
| **File Size (.bin)** | 2.7 MB |
| **Calibration Dataset** | 54 images at 256×128 (COCO val2017 + random) |
| **Calibration Method** | max (auto-selected) |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**CPU Nodes (run on ARM CPU):**
- `Relu_394feature_reshape_Reshape_0` — reshape `[1,512,1,1]` → `[1,512]`

**How Re-ID works in pipeline:**
```python
# 1. Get person bbox from YOLOv8s
x1, y1, x2, y2 = person_bbox

# 2. Crop and resize person
person_crop = image[y1:y2, x1:x2]
person_crop = cv2.resize(person_crop, (128, 256))  # W=128, H=256
person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

# 3. Extract feature vector
feature = model.infer(person_crop)  # shape: [1, 512]

# 4. Compare with gallery (cosine similarity)
similarity = cosine_similarity(feature, gallery_feature)
if similarity > 0.7:
    person_id = matched_id   # same person
else:
    person_id = new_id       # new person → assign new ID
```

**Dwell Time Measurement (using Re-ID):**
```python
# Track person across frames
person_tracker = {}  # {person_id: {"first_seen": timestamp, "last_seen": timestamp}}

# On each frame
if person_id in person_tracker:
    person_tracker[person_id]["last_seen"] = current_time
else:
    person_tracker[person_id] = {
        "first_seen": current_time,
        "last_seen": current_time
    }

# Calculate dwell time
dwell_time = person_tracker[person_id]["last_seen"] - person_tracker[person_id]["first_seen"]
if dwell_time > 2.0:  # seconds
    print(f"Person {person_id} is interested (dwell: {dwell_time:.1f}s)")
```

---

---

### 7. FERPlus Emotion Recognition — 8 Emotion Classes

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/emotion.bin` |
| **Source** | [Microsoft FERPlus](https://github.com/microsoft/FERPlus) via PINTO Model Zoo 259_Emotion_FERPlus |
| **Task** | Facial Emotion Recognition — 8 classes |
| **Architecture** | CNN-based deep network (VGG-style) |
| **Training Dataset** | FER2013 + FERPlus soft labels (35,887 images) |
| **Input Name** | `Input3` |
| **Input Shape** | `1 × 1 × 64 × 64` |
| **Input Type** | Grayscale (single channel) |
| **Input Type (Runtime)** | gray — cropped face region converted to grayscale |
| **Input Layout** | NCHW |
| **Output Name** | `tf.identity` |
| **Output Shape** | `1 × 8` |
| **Output Format** | 8 emotion probability scores (softmax) |
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 1 (fully on BPU except final Reshape) |
| **Cosine Similarity** | **0.999995** 🔥 |
| **BPU FPS** | ~684 FPS |
| **BPU Latency** | ~1.46 ms |
| **File Size (.bin)** | 8.7 MB |
| **Calibration Dataset** | 54 grayscale images at 64×64 |
| **Calibration Method** | max-percentile (percentile = 0.99995) |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**8 Emotion Classes (output index):**
```
0: neutral
1: happiness
2: surprise
3: sadness
4: anger
5: disgust
6: fear
7: contempt
```

**CPU Nodes (run on ARM CPU):**
- `MatMul_Gemm__28_transpose_output_reshape` — final reshape only

**Usage — crop face before inference:**
```python
# 1. Get face bbox from YOLOv8n-face
x1, y1, x2, y2 = face_bbox

# 2. Crop, resize, convert to grayscale
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (64, 64))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)  # grayscale

# 3. Run inference
output = model.infer(face_crop)  # shape: [1, 8]

# 4. Get emotion
emotions = ['neutral','happiness','surprise','sadness',
            'anger','disgust','fear','contempt']
emotion_id = output[0].argmax()
emotion = emotions[emotion_id]
confidence = output[0][emotion_id]

# 5. Map to attention signal
interested = emotion in ['happiness', 'surprise', 'neutral']
```

**Retail Analytics Mapping:**
```
happiness  → very interested 😊
surprise   → attracted attention 😮
neutral    → browsing normally 😐
sadness    → not interested 😞
disgust    → negative reaction 😤
anger      → frustrated 😠
contempt   → dismissive 😒
fear       → uncomfortable 😨
```

---

## Final Complete Model Summary

| # | Model | Task | Cosine Sim | FPS | Latency | Size |
|---|-------|------|-----------|-----|---------|------|
| 1 | yolov8s.bin | Person Detection | 0.999772 | ~88 | ~11.3 ms | 13 MB |
| 2 | yolov8n-face.bin | Face Detection | 0.999706 | ~100+ | <10 ms | 4.9 MB |
| 3 | yolov8s-pose.bin | Pose Estimation | 0.999903 | ~21 | ~47.3 ms | 15 MB |
| 4 | genderage.bin | Age + Gender | 0.995687 | ~10,116 | ~0.1 ms | 579 KB |
| 5 | headpose.bin | Head Pose (yaw/pitch/roll) | 1.000000 | ~6,878 | ~0.145 ms | 454 KB |
| 6 | osnet_reid.bin | Person Re-ID (512-dim) | 0.985353 | ~868 | ~1.152 ms | 2.7 MB |
| 7 | emotion.bin | Emotion (8 classes) | 0.999995 | ~684 | ~1.46 ms | 8.7 MB |

**Total pipeline size: ~45 MB**

---

## Complete Retail Analytics Pipeline

```
Camera NV12 Frame (640×640)
           │
           ▼
    ┌─────────────┐
    │  YOLOv8s    │──────────────────► Person BBox
    └─────────────┘                         │
           │                     ┌──────────┴──────────┐
           │                     ▼                     ▼
           │              OSNet Re-ID           YOLOv8s-pose
           │              512-dim feature       17 keypoints
           │              → Track ID            → Body orientation
           │
    ┌─────────────┐
    │YOLOv8n-face │──────────► Face BBox
    └─────────────┘                │
                        ┌──────────┼──────────┬──────────┐
                        ▼          ▼          ▼          ▼
                    headpose   genderage   emotion    (future)
                    yaw/pitch  age/gender  8 classes  gaze
                    roll
                        │          │          │
                        └──────────┴──────────┘
                                   │
                                   ▼
                        📊 Analytics Output
                   ┌──────────────────────────────┐
                   │ person_id:      7             │
                   │ dwell_time:     4.2s          │
                   │ looking:        True          │
                   │ yaw: -12° pitch: 5°           │
                   │ age: 28  gender: Male         │
                   │ emotion:  happiness (87%)     │
                   │ pose:     facing_forward      │
                   └──────────────────────────────┘
```

---


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