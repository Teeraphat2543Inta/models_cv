# RDK X5 Vision Pipeline
> AI Vision Pipeline for Retail Analytics on RDK X5 + Pi Camera Module 2  
> Powered by D-Robotics BPU (Bayes-E architecture)

---

## 🎯 Use Case
Retail shelf analytics — detecting people walking past, measuring attention (head turns), dwell time, and demographic insights (age/gender) in real time on edge hardware.

---

## 📋 Table of Contents
- [Hardware](#hardware)
- [Repository Structure](#repository-structure)
- [Models](#models)
  - [1. YOLOv8s — Person Detection](#1-yolov8s--person-detection)
  - [2. YOLOv8n-face — Face Detection (General)](#2-yolov8n-face--face-detection-general)
  - [3. YOLOv8n-face-widerface — Face Detection ⭐](#3-yolov8n-face-widerface--face-detection-recommended)
  - [4. YOLOv8s-pose — Pose Estimation](#4-yolov8s-pose--pose-estimation)
  - [5. InsightFace GenderAge — Age & Gender](#5-insightface-genderage--age--gender)
  - [6. Headpose — Head Pose Estimation](#6-headpose--head-pose-estimation)
  - [7. OSNet Re-ID — Person Re-Identification](#7-osnet-re-id--person-re-identification)
  - [8. FERPlus Emotion Recognition](#8-ferplus-emotion-recognition)
  - [9. CSRNet — Crowd Density Estimation](#9-csrnet--crowd-density-estimation)
  - [10. Gender MobileNetV3](#10-gender-mobilenetv3--male--female--unknown)
  - [11. Gender MobileNetV2 ⭐](#11-gender-mobilenetv2--male--female--unknown-recommended)
- [Calibration Details](#calibration-details)
- [Final Model Summary](#final-model-summary)
- [Complete Pipeline Flow](#complete-pipeline-flow)
- [How to Use on Board](#how-to-use-on-board)
- [Toolchain Info](#toolchain-info)
- [License](#license)

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
| Board IP | 172.20.10.2 |
| Board User | supersensor |
| Runtime Version | hrt_model_exec v1.24.5 |

---

## Repository Structure

```
models_cv/
├── models/
│   ├── bin/                           ← compiled .bin models for RDK X5 deployment
│   ├── hbm/                           ← .hbm models (identical to .bin, renamed for compatibility)
│   ├── onnx/                          ← ONNX source models (pre-conversion)
│   └── configs/                       ← YAML configs for hb_mapper
├── data/
│   ├── calibration/                   ← raw JPG calibration images (54 files)
│   ├── calibration_processed/         ← 640×640 uint8 RGB binary
│   ├── calibration_processed_96/      ← 96×96 for genderage
│   ├── calibration_processed_224/     ← 224×224 for headpose
│   ├── calibration_processed_320/     ← 320×320 for csrnet
│   ├── calibration_processed_256x128/ ← 256×128 for osnet_reid
│   ├── calibration_processed_64/      ← 64×64 grayscale for emotion
│   └── calibration_gender_float/      ← 224×224 float32 ImageNet-normalized (UTKFace)
├── scripts/
│   ├── export_yolo.py
│   ├── convert_model.sh
│   └── test_inference.sh
└── docs/
    └── 01_setup.md
```

### models/hbm/ — HBM Format Models

All 11 models are available in both `.bin` and `.hbm` format. The `.hbm` files are identical to `.bin` — same content, renamed for team workflow compatibility.

| File | Size | Notes |
|------|------|-------|
| yolov8s.hbm | 13 MB | Person detection |
| yolov8n-face.hbm | 4.9 MB | Face detection (general weights) |
| yolov8n-face-widerface.hbm | ~12 MB | Face detection ⭐ WiderFace + landmarks |
| yolov8s-pose.hbm | 15 MB | Pose estimation 17 keypoints |
| genderage.hbm | 579 KB | Age + gender InsightFace |
| headpose.hbm | 454 KB | Head pose yaw/pitch/roll |
| osnet_reid.hbm | 2.7 MB | Person Re-ID 512-dim |
| emotion.hbm | 8.7 MB | Emotion 8 classes |
| csrnet.hbm | 514 KB | Crowd density map |
| gender_mobilenetv3.hbm | 5.5 MB | Gender 3-class MobileNetV3 |
| gender_mobilenetv2.hbm | 2.9 MB | Gender 3-class MobileNetV2 ⭐ |

**Total: ~66 MB**

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

### 2. YOLOv8n-face — Face Detection (General)

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
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Cosine Similarity** | **0.999706** 🟢 |
| **BPU FPS** | ~100+ FPS |
| **BPU Latency** | <10 ms |
| **File Size (.bin)** | 4.9 MB |
| **DataType** | int8 (BPU), float32 (CPU nodes) |

**Post-processing parameters:**
```python
conf_threshold = 0.4
iou_threshold  = 0.5
input_size     = (640, 640)
```

> ⚠️ Base weights are YOLOv8n (general COCO detector) — not face-specific trained. For better face detection accuracy, use **yolov8n-face-widerface.bin** instead.

---

### 3. YOLOv8n-face-widerface — Face Detection (Recommended)

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/yolov8n-face-widerface.bin` |
| **Source** | [akanametov/yolo-face](https://github.com/akanametov/yolo-face) v1.0.0 — trained on WiderFace dataset |
| **Task** | Face Detection + 5 Facial Landmarks |
| **Architecture** | YOLOv8 Nano Pose |
| **Input Name** | `images` |
| **Input Shape** | `1 × 3 × 640 × 640` |
| **Input Type (Runtime)** | NV12 — direct from camera pyramid |
| **Input Type (Training)** | RGB, NCHW |
| **Output Name** | `output0` |
| **Output Shape** | `1 × 20 × 8400` |
| **Output Format** | 20 = 4 bbox + 1 conf + 5 landmarks × 3 (x, y, conf) |
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **Producer** | PyTorch v2.10.0 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 2 |
| **Cosine Similarity** | **0.999942** 🔥 |
| **BPU FPS (subgraph 0)** | ~83 FPS |
| **BPU FPS (subgraph 1)** | ~317 FPS |
| **BPU Latency (subgraph 0)** | ~12 ms |
| **BPU Latency (subgraph 1)** | ~3.1 ms |
| **File Size (.bin)** | ~12 MB |
| **Calibration Dataset** | 54 images at 640×640 (COCO val2017 + random) |
| **Calibration Method** | max-percentile (percentile = 0.99995) |
| **DataType** | int8 (BPU), float32 (CPU: Reshape × 3, Concat, Softmax) |

**5 Facial Landmarks (output index 5–19):**
```
landmark[0]: left_eye    → output[0, 5:8,  i]  (x, y, conf)
landmark[1]: right_eye   → output[0, 8:11, i]  (x, y, conf)
landmark[2]: nose        → output[0, 11:14, i] (x, y, conf)
landmark[3]: mouth_left  → output[0, 14:17, i] (x, y, conf)
landmark[4]: mouth_right → output[0, 17:20, i] (x, y, conf)
```

**Why use this over yolov8n-face.bin:**
```
yolov8n-face.bin         → general COCO weights, no face-specific training
                         → may miss small/occluded/side-profile faces
yolov8n-face-widerface   → trained on WiderFace (32,203 images, 393,703 face annotations)
                         → significantly better on small, occluded, side-profile faces
                         → includes 5 facial landmarks for face alignment
                         → better cosine similarity: 0.999942 vs 0.999706
```

**Usage:**
```python
output = model.infer(frame)  # shape: [1, 20, 8400]

for i in range(output.shape[2]):
    conf = output[0, 4, i]
    if conf < 0.4:
        continue
    x1, y1, x2, y2 = output[0, :4, i]
    landmarks = output[0, 5:, i].reshape(5, 3)  # [lm_x, lm_y, lm_conf] × 5

    left_eye   = landmarks[0]  # (x, y, conf)
    right_eye  = landmarks[1]
    nose       = landmarks[2]
    mouth_l    = landmarks[3]
    mouth_r    = landmarks[4]
```

**Post-processing parameters:**
```python
conf_threshold = 0.4
iou_threshold  = 0.5
input_size     = (640, 640)
```

> ⭐ **Recommended** over `yolov8n-face.bin` for all production use cases.

---

### 4. YOLOv8s-pose — Pose Estimation

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
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
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
nose_conf      = keypoints[0][2]
l_shoulder_x   = keypoints[5][0]
r_shoulder_x   = keypoints[6][0]
shoulder_width = abs(r_shoulder_x - l_shoulder_x)
# if shoulder_width > threshold → person is facing the camera
```

> ⚠️ Known issue: O3 causes compiler crash (`calculate size exceed peak dim`) — must use O1. O1 is ~4× slower than O3 would be; upgrade hbdk when this is fixed.

---

### 5. InsightFace GenderAge — Age & Gender

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
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **Original ONNX Opset** | 12 (downgraded to 11 for compatibility) |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 1 (almost fully on BPU) |
| **Cosine Similarity** | **0.995687** 🟢 |
| **BPU FPS** | **~10,116 FPS** 🚀 |
| **BPU Latency** | **~98.8 microseconds** 🚀 |
| **File Size (.bin)** | 579 KB |
| **Calibration Dataset** | 54 images at 96×96 (COCO val2017 + random) |
| **Calibration Method** | max-percentile (percentile = 0.99995) |
| **DataType** | int8 (BPU), float32 (CPU: final Concat + Reshape) |
| **Gender Accuracy** | ~97% (InsightFace buffalo_l benchmark) |
| **Age MAE** | ~3-4 years |

**CPU Nodes (run on ARM CPU):**
- Final Concat + Reshape — merge outputs

**Usage — crop face before inference:**
```python
x1, y1, x2, y2 = face_bbox

face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (96, 96))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

output = model.infer(face_crop)  # shape: [1, 3]
gender = 'Male' if output[0][0] > 0.5 else 'Female'
age    = int(output[0][1] * 100)
```

---

### 6. Headpose — Head Pose Estimation

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
yaw   > 0° → head turning right      yaw   < 0° → head turning left
pitch > 0° → head tilting up         pitch < 0° → head tilting down
roll  > 0° → head tilting right      roll  < 0° → head tilting left
```

**Attention Detection Logic:**
```python
x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (224, 224))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

output = model.infer(face_crop)
yaw   = output['yaw']    # degrees
pitch = output['pitch']  # degrees
roll  = output['roll']   # degrees

looking_at_shelf = abs(yaw) < 30 and abs(pitch) < 30
```

---

### 7. OSNet Re-ID — Person Re-Identification

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
- `Relu_394feature_reshape_Reshape_0` — reshape `[1, 512, 1, 1]` → `[1, 512]`

**How Re-ID works in pipeline:**
```python
x1, y1, x2, y2 = person_bbox
person_crop = image[y1:y2, x1:x2]
person_crop = cv2.resize(person_crop, (128, 256))  # W=128, H=256
person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

feature = model.infer(person_crop)  # shape: [1, 512]

similarity = cosine_similarity(feature, gallery_feature)
if similarity > 0.7:
    person_id = matched_id   # same person seen before
else:
    person_id = new_id       # new person → assign new ID
```

**Dwell Time Measurement:**
```python
person_tracker = {}  # {person_id: {"first_seen": timestamp, "last_seen": timestamp}}

if person_id in person_tracker:
    person_tracker[person_id]["last_seen"] = current_time
else:
    person_tracker[person_id] = {
        "first_seen": current_time,
        "last_seen":  current_time
    }

dwell_time = person_tracker[person_id]["last_seen"] - person_tracker[person_id]["first_seen"]
if dwell_time > 2.0:
    print(f"Person {person_id} is interested (dwell: {dwell_time:.1f}s)")
```

---

### 8. FERPlus Emotion Recognition

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/emotion.bin` |
| **Source** | [Microsoft FERPlus](https://github.com/microsoft/FERPlus) via PINTO Model Zoo 259_Emotion_FERPlus |
| **Task** | Facial Emotion Recognition — 8 classes |
| **Architecture** | CNN-based deep network (VGG-style) |
| **Training Dataset** | FER2013 + FERPlus soft labels (35,887 images) |
| **Input Name** | `Input3` |
| **Input Shape** | `1 × 1 × 64 × 64` |
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

**CPU Nodes (run on ARM CPU):**
- `MatMul_Gemm__28_transpose_output_reshape` — final reshape only

**8 Emotion Classes (output index):**
```
0: neutral    1: happiness  2: surprise   3: sadness
4: anger      5: disgust    6: fear       7: contempt
```

**Usage — crop face before inference:**
```python
x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (64, 64))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)  # grayscale

output = model.infer(face_crop)  # shape: [1, 8]

emotions = ['neutral','happiness','surprise','sadness',
            'anger','disgust','fear','contempt']
emotion_id = output[0].argmax()
emotion    = emotions[emotion_id]
confidence = output[0][emotion_id]

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

### 9. CSRNet — Crowd Density Estimation

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/csrnet.bin` |
| **Source** | [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) via PINTO Model Zoo 400_CSRNet |
| **Task** | Crowd Density Estimation — generate density map + people count |
| **Architecture** | CSRNet (Congested Scene Recognition Network) |
| **Training Dataset** | ShanghaiTech Part A + Part B |
| **Input Name** | `input` |
| **Input Shape** | `1 × 3 × 320 × 320` |
| **Input Type (Runtime)** | NV12 — full frame from camera |
| **Input Type (Training)** | RGB, NCHW |
| **Output Name** | `output` |
| **Output Shape** | `1 × 3 × 320 × 320` |
| **Output Format** | Density map — sum of all pixel values = estimated people count |
| **Normalization** | `pixel / 255.0` (data_scale = 0.003921568627) |
| **ONNX Opset** | 11 |
| **Producer** | PyTorch v2.1.0 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 1 (fully on BPU) |
| **Cosine Similarity** | **0.995526** 🟢 |
| **BPU FPS** | ~224 FPS |
| **BPU Latency** | ~4.4 ms |
| **File Size (.bin)** | 514 KB |
| **Calibration Dataset** | 54 images at 320×320 (COCO val2017 + random) |
| **Calibration Method** | max-percentile (percentile = 0.99995) |
| **DataType** | int8 (BPU, fully on BPU) |

> ⚠️ Original model input is 640×640 but forced to 320×320 to fit memory constraints. For production use, consider using the native 640×640 model on a device with more RAM.

**How to get people count from density map:**
```python
density_map = model.infer(frame)  # shape: [1, 3, 320, 320]

count = density_map[0][0].sum()
print(f"Estimated people count: {count:.1f}")
```

**Zone-based density analysis:**
```python
import numpy as np

density = density_map[0][0]  # shape: [320, 320]

h, w = density.shape
zones = {
    'top_left':     density[:h//2, :w//2].sum(),
    'top_right':    density[:h//2, w//2:].sum(),
    'bottom_left':  density[h//2:, :w//2].sum(),
    'bottom_right': density[h//2:, w//2:].sum(),
}

busiest = max(zones, key=zones.get)
print(f"Busiest zone: {busiest} ({zones[busiest]:.1f} people)")
```

---

### 10. Gender MobileNetV3 — Male / Female / Unknown

| Parameter | Value |
|-----------|-------|
| **Files** | `models/bin/gender_mobilenetv3.bin` / `models/hbm/gender_mobilenetv3.hbm` |
| **Task** | Gender Classification — 3 classes |
| **Architecture** | MobileNetV3 Large + Custom Classifier Head |
| **Training Dataset** | UTKFace (23,708 images) |
| **Input Name** | `input` |
| **Input Shape** | `1 × 3 × 224 × 224` |
| **Input Type (Runtime)** | featuremap — pre-normalized float32 from DDR |
| **Input Layout** | NCHW |
| **Normalization** | ImageNet mean=[123.675, 116.28, 103.53] std=[58.395, 57.12, 57.375] |
| **Output Name** | `output` |
| **Output Shape** | `1 × 3` |
| **Output Format** | `[Male_score, Female_score, Unknown_score]` (raw logits) |
| **ONNX Opset** | 11 |
| **Producer** | PyTorch v2.10.0 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Input Source** | DDR (featuremap — not pyramid) |
| **Cosine Similarity** | **0.997586** 🟢 |
| **BPU FPS** | ~1,012 FPS |
| **BPU Latency** | ~1.0 ms |
| **File Size (.bin/.hbm)** | 5.5 MB |
| **Calibration Dataset** | 154 UTKFace images (float32, ImageNet normalized) |
| **Calibration Method** | max-percentile (percentile = 0.99995) |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**3 Output Classes:**
```
0: Male
1: Female
2: Unknown  ← age < 5 years, blurry image, side/back view, low confidence
```

**Model Architecture:**
```
MobileNetV3 Large (backbone, pretrained ImageNet)
    │
    ▼
BatchNorm1d(1280)
Linear(1280 → 512) + ReLU + Dropout(0.4)
Linear(512 → 128)  + ReLU + Dropout(0.3)
Linear(128 → 3)
```

**Training Details:**
```
Dataset:         UTKFace 23,708 images
Train/Val split: 18,967 / 4,741
Epochs:          50 (best at epoch 48)
Optimizer:       AdamW (backbone lr=1e-5, head lr=1e-4)
Scheduler:       CosineAnnealingWarmRestarts
Loss:            CrossEntropyLoss + label_smoothing=0.1
                 weight=[1.0, 1.0, 3.0] ← boost Unknown class
Augmentation:    RandomCrop(224), HFlip, ColorJitter(b=0.4,c=0.4,s=0.3)
                 RandomPerspective, RandomErasing(p=0.2)
Mixup:           alpha=0.3
Mixed Precision: AMP (autocast + GradScaler)
```

**Validation Accuracy (best epoch 48):**
```
Overall:  93.21%
Male:     94.54%
Female:   90.93%
Unknown:  94.74%
```

**Usage — must normalize before inference:**
```python
import cv2
import numpy as np

x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (224, 224))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32)

mean = np.array([123.675, 116.28,  103.53], dtype=np.float32)
std  = np.array([58.395,  57.12,   57.375], dtype=np.float32)
face_crop = (face_crop - mean) / std

face_crop = face_crop.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 224, 224]

output = model.infer(face_crop)  # shape: [1, 3]
classes = ['Male', 'Female', 'Unknown']
scores  = softmax(output[0])
gender  = classes[scores.argmax()] if scores.max() > 0.6 else 'Unknown'

print(f"Gender: {gender} ({scores.max():.1%})")
```

> ⚠️ Input source is **DDR (featuremap)** — must apply ImageNet normalization manually before inference.  
> ⚠️ MobileNetV3 uses Hard-Swish activation which causes some deep intermediate layers to have low Cosine Similarity after PTQ. Use **gender_mobilenetv2** for better quantization quality.

---

### 11. Gender MobileNetV2 — Male / Female / Unknown (Recommended)

| Parameter | Value |
|-----------|-------|
| **Files** | `models/bin/gender_mobilenetv2.bin` / `models/hbm/gender_mobilenetv2.hbm` |
| **Task** | Gender Classification — 3 classes |
| **Architecture** | MobileNetV2 + Custom Classifier Head |
| **Training Dataset** | UTKFace (23,708 images) |
| **Input Name** | `input` |
| **Input Shape** | `1 × 3 × 224 × 224` |
| **Input Type (Runtime)** | featuremap — pre-normalized float32 from DDR |
| **Input Layout** | NCHW |
| **Normalization** | ImageNet mean=[123.675, 116.28, 103.53] std=[58.395, 57.12, 57.375] |
| **Output Name** | `output` |
| **Output Shape** | `1 × 3` |
| **Output Format** | `[Male_score, Female_score, Unknown_score]` (raw logits) |
| **ONNX Opset** | 11 |
| **Producer** | PyTorch v2.10.0 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Input Source** | DDR (featuremap — not pyramid) |
| **Cosine Similarity** | **0.999759** 🔥 |
| **All layers Cosine Sim** | **> 0.983** (every single layer) 🔥 |
| **BPU FPS** | ~1,590 FPS |
| **BPU Latency** | ~628 microseconds |
| **File Size (.bin/.hbm)** | ~2.9 MB |
| **Calibration Dataset** | 154 UTKFace images (float32, ImageNet normalized) |
| **Calibration Method** | max (auto-selected) |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**3 Output Classes:**
```
0: Male
1: Female
2: Unknown  ← age < 5 years, blurry image, side/back view, low confidence
```

**Why MobileNetV2 over MobileNetV3:**
```
MobileNetV3 uses Hard-Swish activation
→ Non-linear, difficult to approximate with int8
→ Some intermediate layers drop to < 0.5 Cosine Similarity after PTQ
→ Potential accuracy loss in real deployment

MobileNetV2 uses ReLU6 activation
→ Simple clipping, quantizes cleanly with int8
→ All layers maintain > 0.983 Cosine Similarity after PTQ
→ More reliable accuracy in real deployment
```

**Model Architecture:**
```
MobileNetV2 (backbone, pretrained ImageNet)
    │
    ▼
GlobalAveragePool → feature vector [1, 1280]
Linear(1280 → 256) + ReLU + Dropout(0.3)
Linear(256 → 3)
```

**Training Details:**
```
Dataset:         UTKFace 23,708 images
Train/Val split: 18,967 / 4,741
Epochs:          50 (best at epoch 40)
Optimizer:       AdamW (backbone lr=1e-5, head lr=1e-4)
Scheduler:       CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
Loss:            CrossEntropyLoss + label_smoothing=0.1
                 weight=[1.0, 1.0, 3.0] ← boost Unknown class
Augmentation:    RandomCrop(224), HFlip, ColorJitter(b=0.3,c=0.3,s=0.2)
                 RandomRotation(10), RandomErasing(p=0.2)
Mixup:           alpha=0.3
Mixed Precision: AMP (autocast + GradScaler)
```

**Validation Accuracy (best epoch 40):**
```
Overall:  92.30%
Male:     94.27%
Female:   89.66%
Unknown:  94.07%
```

**Comparison vs MobileNetV3:**

| Metric | MobileNetV3 | MobileNetV2 | Winner |
|--------|------------|------------|--------|
| Output Cosine Similarity | 0.997586 | **0.999759** | ✅ V2 |
| Minimum layer Cosine Sim | -0.006 | **0.983** | ✅ V2 |
| BPU FPS | ~1,012 | **~1,590** | ✅ V2 |
| BPU Latency | ~1.0 ms | **~0.63 ms** | ✅ V2 |
| File size | 5.5 MB | **2.9 MB** | ✅ V2 |
| Val Accuracy | 93.21% | 92.30% | V3 (slight) |

**Usage — must normalize before inference:**
```python
import cv2
import numpy as np

x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (224, 224))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB).astype(np.float32)

mean = np.array([123.675, 116.28,  103.53], dtype=np.float32)
std  = np.array([58.395,  57.12,   57.375], dtype=np.float32)
face_crop = (face_crop - mean) / std

face_crop = face_crop.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 224, 224]

output = model.infer(face_crop)  # shape: [1, 3]
classes = ['Male', 'Female', 'Unknown']
scores  = softmax(output[0])
gender  = classes[scores.argmax()] if scores.max() > 0.6 else 'Unknown'

print(f"Gender: {gender} ({scores.max():.1%})")
```

> ⚠️ Input source is **DDR (featuremap)** — must apply ImageNet normalization manually before inference.  
> ⭐ **This is the recommended gender model for production deployment on RDK X5.**

---

## Calibration Details

| Parameter | Value |
|-----------|-------|
| **Standard calibration images** | 54 (34 COCO val2017 + 20 random/picsum) |
| **Gender model calibration** | 154 UTKFace face images (float32 ImageNet-normalized) |
| **Standard pixel format** | uint8, RGB channel order |
| **Gender model pixel format** | float32, ImageNet-normalized (mean/std applied) |
| **File format** | Raw binary `.bin` (H × W × C) |
| **Default method** | max-percentile (percentile = 0.99995) |
| **Per-Channel Quantization** | False |
| **Asymmetric Quantization** | False |
| **Batch Size** | 8 (auto-reset to 1 when shape mismatch occurs) |

| Model | Calibration Size | Type | Folder |
|-------|-----------------|------|--------|
| yolov8s | 640×640 | uint8 RGB | `calibration_processed/` |
| yolov8n-face | 640×640 | uint8 RGB | `calibration_processed/` |
| yolov8n-face-widerface | 640×640 | uint8 RGB | `calibration_processed/` |
| yolov8s-pose | 640×640 | uint8 RGB | `calibration_processed/` |
| genderage | 96×96 | uint8 RGB | `calibration_processed_96/` |
| headpose | 224×224 | uint8 RGB | `calibration_processed_224/` |
| emotion | 64×64 | uint8 grayscale | `calibration_processed_64/` |
| osnet_reid | 256×128 | uint8 RGB | `calibration_processed_256x128/` |
| csrnet | 320×320 | uint8 RGB | `calibration_processed_320/` |
| gender_mobilenetv3 | 224×224 | float32 ImageNet-norm | `calibration_gender_float/` |
| gender_mobilenetv2 | 224×224 | float32 ImageNet-norm | `calibration_gender_float/` |

---

## Final Model Summary

| # | Model | Task | Cosine Sim | FPS | Latency | Size |
|---|-------|------|-----------|-----|---------|------|
| 1 | yolov8s.bin | Person Detection | 0.999772 | ~88 | ~11.3 ms | 13 MB |
| 2 | yolov8n-face.bin | Face Detection (general) | 0.999706 | ~100+ | <10 ms | 4.9 MB |
| 3 | **yolov8n-face-widerface.bin** ⭐ | **Face + 5 Landmarks (WiderFace)** | **0.999942** | ~83 | ~12 ms | ~12 MB |
| 4 | yolov8s-pose.bin | Pose Estimation 17kp | 0.999903 | ~21 | ~47.3 ms | 15 MB |
| 5 | genderage.bin | Age + Gender (InsightFace) | 0.995687 | ~10,116 | ~0.1 ms | 579 KB |
| 6 | headpose.bin | Head Pose yaw/pitch/roll | 1.000000 | ~6,878 | ~0.145 ms | 454 KB |
| 7 | osnet_reid.bin | Person Re-ID 512-dim | 0.985353 | ~868 | ~1.152 ms | 2.7 MB |
| 8 | emotion.bin | Emotion 8 classes | 0.999995 | ~684 | ~1.46 ms | 8.7 MB |
| 9 | csrnet.bin | Crowd Density Map | 0.995526 | ~224 | ~4.4 ms | 514 KB |
| 10 | gender_mobilenetv3.bin/.hbm | Gender 3-class (MobileNetV3) | 0.997586 | ~1,012 | ~1.0 ms | 5.5 MB |
| 11 | **gender_mobilenetv2.bin/.hbm** ⭐ | **Gender 3-class (MobileNetV2)** | **0.999759** | **~1,590** | **~0.63 ms** | **2.9 MB** |

**Total pipeline size: ~66 MB**

> ⭐ `yolov8n-face-widerface` recommended over `yolov8n-face` — trained on WiderFace, better accuracy on small/occluded faces, includes 5 facial landmarks.  
> ⭐ `gender_mobilenetv2` recommended over `gender_mobilenetv3` — all layers > 0.983 Cosine Similarity, faster, smaller.

---

## Complete Pipeline Flow

```
Camera NV12 Frame (640×640)
           │
     ┌─────┴──────────────────┐
     ▼                        ▼
 YOLOv8s                  CSRNet (320×320)
 Person BBox               → People Count
 → Track via Re-ID         → Zone Heatmap
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
OSNet Re-ID                       YOLOv8s-pose
512-dim feature                   17 keypoints
→ Person ID                       → Body orientation
→ Dwell time                      → Facing camera?
     │
YOLOv8n-face-widerface ⭐
→ Face BBox
→ 5 Facial Landmarks (eye×2, nose, mouth×2)
     │
     ├──────────┬──────────┬──────────────────────┐
     ▼          ▼          ▼                      ▼
 headpose   genderage  emotion        gender_mobilenetv2 ⭐
 yaw/pitch  age/gender 8 classes      Male/Female/Unknown
 roll       ~97% acc   Cosine 0.9999  Cosine 0.9997
     │
     └─────────────────────────────────────────────┐
                                                   ▼
                                      📊 Analytics Output
                              ┌──────────────────────────────────┐
                              │ traffic_count:    47 people       │
                              │ zone_heatmap:     [zone data]     │
                              │ person_id:        7               │
                              │ dwell_time:       4.2s            │
                              │ looking_at_shelf: True            │
                              │ yaw: -12°  pitch: 5°              │
                              │ age: 28    gender: Male           │
                              │ emotion:   happiness (87%)        │
                              │ pose:      facing_forward         │
                              └──────────────────────────────────┘
```

---

## How to Use on Board

### 1. Copy all models to board
```bash
scp models/bin/*.bin supersensor@172.20.10.2:/home/supersensor/models/
```

### 2. SSH into board
```bash
ssh supersensor@172.20.10.2
```

### 3. Create models directory on board
```bash
mkdir -p /home/supersensor/models
```

### 4. Check model info
```bash
hrt_model_exec model_info --model_file=models/yolov8s.bin
```

### 5. Benchmark latency (single thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=1 \
  --frame_count=200
```

### 6. Benchmark FPS (multi thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=8 \
  --frame_count=200
```

### 7. Test all models at once
```bash
for model in /home/supersensor/models/*.bin; do
    echo "========== $(basename $model) =========="
    hrt_model_exec perf \
      --model_file=$model \
      --thread_num=1 \
      --frame_count=100
    echo ""
done
```

### 8. Set CPU to performance mode (for best benchmark results)
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
| Docker Image | `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8` |
| hrt_model_exec | v1.24.5 (on board) |
| PyTorch | 2.10.0 |
| Ultralytics | 8.4.39 |
| ONNX | 1.21.0 |
| InsightFace | 0.7.3 |
| Python | 3.12 |
| Board OS | Ubuntu 22.04 aarch64 |

---

## License

Models are based on open-source weights:
- YOLOv8: [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- YOLOv8n-face-widerface: [akanametov/yolo-face MIT](https://github.com/akanametov/yolo-face/blob/main/LICENSE)
- InsightFace: [MIT License](https://github.com/deepinsight/insightface/blob/master/LICENSE)
- OSNet: [MIT License](https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE)
- FERPlus: [MIT License](https://github.com/microsoft/FERPlus/blob/master/LICENSE)
- CSRNet: [MIT License](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/LICENSE)
- Lightweight Head Pose: [Shaw-git/Lightweight-Head-Pose-Estimation](https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation)