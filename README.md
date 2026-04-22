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
  - [2. YOLOv8n-face — Face Detection](#2-yolov8n-face--face-detection)
  - [3. YOLOv8s-pose — Pose Estimation](#3-yolov8s-pose--pose-estimation)
  - [4. InsightFace GenderAge — Age & Gender](#4-insightface-genderage--age--gender)
  - [5. Headpose — Head Pose Estimation](#5-headpose--head-pose-estimation)
  - [6. OSNet Re-ID — Person Re-Identification](#6-osnet-re-id--person-re-identification)
  - [7. FERPlus Emotion Recognition](#7-ferplus-emotion-recognition)
  - [8. CSRNet — Crowd Density Estimation](#8-csrnet--crowd-density-estimation)
  - [9. Gender MobileNetV3](#9-gender-mobilenetv3--male--female--unknown)
  - [10. Gender MobileNetV2 ⭐](#10-gender-mobilenetv2--male--female--unknown-recommended)
- [Calibration Details](#calibration-details)
- [Final Model Summary](#final-model-summary)
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

All 10 models are available in both `.bin` and `.hbm` format. The `.hbm` files are identical to `.bin` — same content, renamed for team workflow compatibility.

| File | Size |
|------|------|
| yolov8s.hbm | 13 MB |
| yolov8n-face.hbm | 4.9 MB |
| yolov8s-pose.hbm | 15 MB |
| genderage.hbm | 579 KB |
| headpose.hbm | 454 KB |
| osnet_reid.hbm | 2.7 MB |
| emotion.hbm | 8.7 MB |
| csrnet.hbm | 514 KB |
| gender_mobilenetv3.hbm | 5.5 MB |
| gender_mobilenetv2.hbm | 2.9 MB |

**Total: ~54.2 MB**

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
| **BPU Latency** | <10 ms |
| **File Size (.bin)** | 4.9 MB |
| **DataType** | int8 (BPU), float32 (CPU nodes) |

**Post-processing parameters:**
```python
conf_threshold = 0.4
iou_threshold  = 0.5
input_size     = (640, 640)
```

> ⚠️ Base weights are YOLOv8n (general detector). For production, replace with dedicated face weights such as `yolov8-face` from akanametov/yolo-face.

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
nose_conf      = keypoints[0][2]
l_shoulder_x   = keypoints[5][0]
r_shoulder_x   = keypoints[6][0]
shoulder_width = abs(r_shoulder_x - l_shoulder_x)
# if shoulder_width > threshold → person facing camera
```

> ⚠️ Known issue: O3 causes compiler crash (`calculate size exceed peak dim`) — must use O1. O1 is ~4× slower; upgrade hbdk when fixed.

---

### 4. InsightFace GenderAge — Age & Gender

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/genderage.bin` |
| **Source** | InsightFace `buffalo_l` pack |
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
| **Original ONNX Opset** | 12 (downgraded to 11) |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Cosine Similarity** | **0.995687** 🟢 |
| **BPU FPS** | **~10,116 FPS** 🚀 |
| **BPU Latency** | **~98.8 μs** 🚀 |
| **File Size (.bin)** | 579 KB |
| **DataType** | int8 (BPU), float32 (CPU: final Concat + Reshape) |
| **Gender Accuracy** | ~97% |
| **Age MAE** | ~3-4 years |

**Usage:**
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

### 5. Headpose — Head Pose Estimation

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
| **Normalization** | `pixel / 255.0` |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Subgraphs** | 4 |
| **Cosine Similarity** | **1.000000** 🔥 |
| **BPU FPS** | ~6,878 FPS |
| **BPU Latency** | ~145 μs |
| **File Size (.bin)** | 454 KB |
| **Calibration Method** | kl (num_bin=1024, max_num_bin=16384) |
| **DataType** | int8 (BPU), float32 (CPU: Softmax × 3) |

**Angle interpretation:**
```
yaw   > 0° → turning right    yaw   < 0° → turning left
pitch > 0° → tilting up       pitch < 0° → tilting down
roll  > 0° → tilting right    roll  < 0° → tilting left
```

**Usage:**
```python
x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (224, 224))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

output = model.infer(face_crop)
yaw, pitch, roll = output['yaw'], output['pitch'], output['roll']

looking_at_shelf = abs(yaw) < 30 and abs(pitch) < 30
```

---

### 6. OSNet Re-ID — Person Re-Identification

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/osnet_reid.bin` |
| **Source** | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) via PINTO Model Zoo 429_OSNet |
| **Task** | Person Re-Identification — 512-dim feature vector |
| **Architecture** | OSNet x1.0 (Omni-Scale Network) |
| **Training Dataset** | MSMT17 (126,441 images, 4,101 identities) |
| **Input Name** | `base_image` |
| **Input Shape** | `1 × 3 × 256 × 128` |
| **Input Type (Runtime)** | NV12 — cropped person region |
| **Input Type (Training)** | RGB, NCHW |
| **Output Name** | `feature` |
| **Output Shape** | `1 × 512` |
| **Output Format** | 512-dim L2-normalized feature vector |
| **Normalization** | `pixel / 255.0` |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Cosine Similarity** | **0.985353** 🟢 |
| **BPU FPS** | ~868 FPS |
| **BPU Latency** | ~1.152 ms |
| **File Size (.bin)** | 2.7 MB |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**Usage:**
```python
x1, y1, x2, y2 = person_bbox
person_crop = image[y1:y2, x1:x2]
person_crop = cv2.resize(person_crop, (128, 256))  # W=128, H=256
person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

feature = model.infer(person_crop)  # shape: [1, 512]

similarity = cosine_similarity(feature, gallery_feature)
if similarity > 0.7:
    person_id = matched_id
else:
    person_id = new_id  # assign new ID
```

**Dwell time tracking:**
```python
person_tracker = {}

if person_id in person_tracker:
    person_tracker[person_id]["last_seen"] = current_time
else:
    person_tracker[person_id] = {"first_seen": current_time, "last_seen": current_time}

dwell_time = person_tracker[person_id]["last_seen"] - person_tracker[person_id]["first_seen"]
if dwell_time > 2.0:
    print(f"Person {person_id} is interested (dwell: {dwell_time:.1f}s)")
```

---

### 7. FERPlus Emotion Recognition

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
| **Output Format** | 8 emotion probability scores |
| **Normalization** | `pixel / 255.0` |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Cosine Similarity** | **0.999995** 🔥 |
| **BPU FPS** | ~684 FPS |
| **BPU Latency** | ~1.46 ms |
| **File Size (.bin)** | 8.7 MB |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**8 Emotion Classes:**
```
0: neutral    1: happiness  2: surprise   3: sadness
4: anger      5: disgust    6: fear       7: contempt
```

**Usage:**
```python
x1, y1, x2, y2 = face_bbox
face_crop = image[y1:y2, x1:x2]
face_crop = cv2.resize(face_crop, (64, 64))
face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

output = model.infer(face_crop)  # shape: [1, 8]
emotions = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']
emotion  = emotions[output[0].argmax()]
```

**Retail mapping:**
```
happiness → very interested 😊    surprise → attracted attention 😮
neutral   → browsing normally 😐  sadness  → not interested 😞
disgust   → negative reaction 😤  anger    → frustrated 😠
contempt  → dismissive 😒         fear     → uncomfortable 😨
```

---

### 8. CSRNet — Crowd Density Estimation

| Parameter | Value |
|-----------|-------|
| **File** | `models/bin/csrnet.bin` |
| **Source** | [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) via PINTO Model Zoo 400_CSRNet |
| **Task** | Crowd Density Estimation — density map + people count |
| **Architecture** | CSRNet (Congested Scene Recognition Network) |
| **Training Dataset** | ShanghaiTech Part A + Part B |
| **Input Name** | `input` |
| **Input Shape** | `1 × 3 × 320 × 320` |
| **Input Type (Runtime)** | NV12 — full frame from camera |
| **Input Type (Training)** | RGB, NCHW |
| **Output Name** | `output` |
| **Output Shape** | `1 × 3 × 320 × 320` |
| **Output Format** | Density map — sum of pixel values = people count |
| **Normalization** | `pixel / 255.0` |
| **ONNX Opset** | 11 |
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Cosine Similarity** | **0.995526** 🟢 |
| **BPU FPS** | ~224 FPS |
| **BPU Latency** | ~4.4 ms |
| **File Size (.bin)** | 514 KB |
| **DataType** | int8 (BPU, fully on BPU) |

> ⚠️ Original model input is 640×640 but forced to 320×320 to fit memory constraints.

**Usage:**
```python
density_map = model.infer(frame)  # shape: [1, 3, 320, 320]
count = density_map[0][0].sum()
print(f"Estimated people count: {count:.1f}")

# Zone-based analysis
density = density_map[0][0]  # [320, 320]
h, w = density.shape
zones = {
    'top_left':     density[:h//2, :w//2].sum(),
    'top_right':    density[:h//2, w//2:].sum(),
    'bottom_left':  density[h//2:, :w//2].sum(),
    'bottom_right': density[h//2:, w//2:].sum(),
}
busiest = max(zones, key=zones.get)
```

---

### 9. Gender MobileNetV3 — Male / Female / Unknown

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
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Input Source** | DDR (featuremap — not pyramid) |
| **Cosine Similarity** | **0.997586** 🟢 |
| **BPU FPS** | ~1,012 FPS |
| **BPU Latency** | ~1.0 ms |
| **File Size (.bin/.hbm)** | 5.5 MB |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**3 Output Classes:**
```
0: Male    1: Female    2: Unknown (age < 5, blurry, side/back view)
```

**Model Architecture:**
```
MobileNetV3 Large (pretrained ImageNet)
→ BatchNorm1d(1280) → Linear(1280→512) → ReLU → Dropout(0.4)
→ Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→3)
```

**Training Details:**
```
Dataset:        UTKFace 23,708 images  |  Train/Val: 18,967 / 4,741
Epochs:         50 (best at epoch 48)  |  Optimizer: AdamW
Scheduler:      CosineAnnealingWarmRestarts
Loss:           CrossEntropyLoss (label_smoothing=0.1, weight=[1,1,3])
Augmentation:   RandomCrop, HFlip, ColorJitter, RandomPerspective, RandomErasing, Mixup(0.3)
Mixed Precision: AMP
```

**Validation Accuracy:**
```
Overall: 93.21%  |  Male: 94.54%  |  Female: 90.93%  |  Unknown: 94.74%
```

**Usage:**
```python
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
scores = softmax(output[0])
gender = classes[scores.argmax()] if scores.max() > 0.6 else 'Unknown'
```

> ⚠️ Input source is DDR (featuremap) — must apply ImageNet normalization manually before inference.

---

### 10. Gender MobileNetV2 — Male / Female / Unknown (Recommended)

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
| **BPU March** | bayes-e |
| **Optimize Level** | O3 |
| **Compile Mode** | latency |
| **Input Source** | DDR (featuremap — not pyramid) |
| **Cosine Similarity** | **0.999759** 🔥 |
| **All layers Cosine Sim** | **> 0.983** (every single layer) 🔥 |
| **BPU FPS** | ~1,590 FPS |
| **BPU Latency** | ~628 μs |
| **File Size (.bin/.hbm)** | ~2.9 MB |
| **Calibration Dataset** | 154 UTKFace images (float32, ImageNet normalized) |
| **Calibration Method** | max (auto-selected) |
| **DataType** | int8 (BPU), float32 (CPU: final Reshape only) |

**3 Output Classes:**
```
0: Male    1: Female    2: Unknown (age < 5, blurry, side/back view)
```

**Why MobileNetV2 over MobileNetV3:**
```
MobileNetV3: Hard-Swish activation → difficult to quantize → some layers < 0.5 Cosine Sim
MobileNetV2: ReLU6 activation      → quantizes cleanly   → all layers > 0.983 Cosine Sim
```

**Model Architecture:**
```
MobileNetV2 (pretrained ImageNet)
→ GlobalAveragePool → [1, 1280]
→ Linear(1280→256) → ReLU → Dropout(0.3)
→ Linear(256→3)
```

**Training Details:**
```
Dataset:        UTKFace 23,708 images  |  Train/Val: 18,967 / 4,741
Epochs:         50 (best at epoch 40)  |  Optimizer: AdamW
Scheduler:      CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
Loss:           CrossEntropyLoss (label_smoothing=0.1, weight=[1,1,3])
Augmentation:   RandomCrop(224), HFlip, ColorJitter, RandomRotation(10), RandomErasing, Mixup(0.3)
Mixed Precision: AMP
```

**Validation Accuracy:**
```
Overall: 92.30%  |  Male: 94.27%  |  Female: 89.66%  |  Unknown: 94.07%
```

**Comparison vs MobileNetV3:**

| Metric | MobileNetV3 | MobileNetV2 | Winner |
|--------|------------|------------|--------|
| Output Cosine Sim | 0.997586 | **0.999759** | ✅ V2 |
| Min layer Cosine Sim | -0.006 | **0.983** | ✅ V2 |
| FPS | ~1,012 | **~1,590** | ✅ V2 |
| Latency | ~1.0 ms | **~0.63 ms** | ✅ V2 |
| File size | 5.5 MB | **2.9 MB** | ✅ V2 |
| Val Accuracy | 93.21% | 92.30% | ✅ V3 (slight) |

**Usage:**
```python
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
scores = softmax(output[0])
gender = classes[scores.argmax()] if scores.max() > 0.6 else 'Unknown'
```

> ⚠️ Input source is DDR (featuremap) — must apply ImageNet normalization manually before inference.  
> ⭐ **This is the recommended gender model for production deployment on RDK X5.**

---

## Calibration Details

| Parameter | Value |
|-----------|-------|
| **Total Images** | 54 (34 COCO val2017 + 20 random) |
| **Gender models** | 154 UTKFace face images (float32 normalized) |
| **Pixel Format** | uint8 RGB (or float32 for gender models) |
| **File Format** | Raw binary `.bin` (H × W × C) |
| **Method Selected** | max-percentile (percentile = 0.99995) |
| **Per-Channel Quantization** | False |
| **Batch Size** | 8 (auto-reset to 1 when needed) |

| Model | Calibration Size | Folder |
|-------|-----------------|--------|
| yolov8s, yolov8n-face, yolov8s-pose | 640×640 uint8 | `calibration_processed/` |
| genderage | 96×96 uint8 | `calibration_processed_96/` |
| headpose | 224×224 uint8 | `calibration_processed_224/` |
| emotion | 64×64 grayscale uint8 | `calibration_processed_64/` |
| osnet_reid | 256×128 uint8 | `calibration_processed_256x128/` |
| csrnet | 320×320 uint8 | `calibration_processed_320/` |
| gender_mobilenetv3/v2 | 224×224 float32 ImageNet-normalized | `calibration_gender_float/` |

---

## Final Model Summary

| # | Model | Task | Cosine Sim | FPS | Latency | Size |
|---|-------|------|-----------|-----|---------|------|
| 1 | yolov8s.bin | Person Detection | 0.999772 | ~88 | ~11.3 ms | 13 MB |
| 2 | yolov8n-face.bin | Face Detection | 0.999706 | ~100+ | <10 ms | 4.9 MB |
| 3 | yolov8s-pose.bin | Pose Estimation 17kp | 0.999903 | ~21 | ~47.3 ms | 15 MB |
| 4 | genderage.bin | Age + Gender (InsightFace) | 0.995687 | ~10,116 | ~0.1 ms | 579 KB |
| 5 | headpose.bin | Head Pose yaw/pitch/roll | 1.000000 | ~6,878 | ~0.145 ms | 454 KB |
| 6 | osnet_reid.bin | Person Re-ID 512-dim | 0.985353 | ~868 | ~1.152 ms | 2.7 MB |
| 7 | emotion.bin | Emotion 8 classes | 0.999995 | ~684 | ~1.46 ms | 8.7 MB |
| 8 | csrnet.bin | Crowd Density Map | 0.995526 | ~224 | ~4.4 ms | 514 KB |
| 9 | gender_mobilenetv3.bin/.hbm | Gender 3-class (MobileNetV3) | 0.997586 | ~1,012 | ~1.0 ms | 5.5 MB |
| 10 | **gender_mobilenetv2.bin/.hbm** ⭐ | **Gender 3-class (MobileNetV2)** | **0.999759** | **~1,590** | **~0.63 ms** | **2.9 MB** |

**Total pipeline size: ~54.2 MB**

> ⭐ `gender_mobilenetv2` is recommended over `gender_mobilenetv3` — significantly better quantization quality with all layers > 0.983 Cosine Similarity.

---

## How to Use on Board

### 1. Copy models to board
```bash
scp models/bin/*.bin supersensor@172.20.10.2:/home/supersensor/models/
```

### 2. SSH into board
```bash
ssh supersensor@172.20.10.2
```

### 3. Check model info
```bash
hrt_model_exec model_info --model_file=models/yolov8s.bin
```

### 4. Benchmark latency (single thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=1 \
  --frame_count=200
```

### 5. Benchmark FPS (multi thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=8 \
  --frame_count=200
```

### 6. Test all models at once
```bash
for model in /home/supersensor/models/*.bin; do
    echo "========== $(basename $model) =========="
    hrt_model_exec perf --model_file=$model --thread_num=1 --frame_count=100
    echo ""
done
```

### 7. Set CPU to performance mode
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

---

## License

Models are based on open-source weights:
- YOLOv8: [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- InsightFace: [MIT License](https://github.com/deepinsight/insightface/blob/master/LICENSE)
- OSNet: [MIT License](https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE)
- FERPlus: [MIT License](https://github.com/microsoft/FERPlus/blob/master/LICENSE)
- CSRNet: [MIT License](https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/LICENSE)