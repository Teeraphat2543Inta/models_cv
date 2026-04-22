# 01 - Setup Guide
> RDK X5 Vision Pipeline — Complete Setup from Scratch

---

## Prerequisites

| Tool | Version | Platform |
|------|---------|----------|
| Docker Desktop | Latest | macOS / Windows / Linux |
| Python | 3.10+ | macOS / Linux |
| Git | Latest | All |
| GitHub Codespaces | — | Browser |
| Google Colab | — | Browser |

---

## Step 1: Install Docker Desktop

1. Download from https://www.docker.com/products/docker-desktop
2. Install and launch the application
3. Verify Docker is running:

```bash
docker --version
# Docker version 27.x.x
```

---

## Step 2: Check Mac Chip Architecture

```bash
uname -m
# arm64  = Apple Silicon (M1/M2/M3) → must add --platform linux/amd64
# x86_64 = Intel Mac               → works normally
```

> ⚠️ **Apple Silicon users:** The D-Robotics toolchain is x86_64 only. Always use `--platform linux/amd64` flag to run via Rosetta emulation.

---

## Step 3: Clone Repository

```bash
git clone https://github.com/Teeraphat2543Inta/models_cv.git
cd models_cv
```

Repository structure:
```
models_cv/
├── models/
│   ├── bin/     ← compiled .bin models for RDK X5 deployment
│   ├── hbm/     ← .hbm models (identical content to .bin, renamed for compatibility)
│   ├── onnx/    ← ONNX source models (pre-conversion)
│   └── configs/ ← YAML configs for hb_mapper
├── data/
│   └── calibration/ ← calibration images (54 JPG files)
├── scripts/
└── docs/
```

---

## Step 4: Pull D-Robotics Toolchain Docker Image

### macOS Intel / Linux
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

### macOS Apple Silicon (M1/M2/M3)
```bash
docker pull --platform linux/amd64 openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

> 💡 The `cpu` image is sufficient for model conversion. The `gpu` image is only needed for GPU-accelerated training.

---

## Step 5: Run Docker Container

### macOS Intel / Linux
```bash
docker run -it --rm \
  -v /path/to/models_cv:/open_explorer \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

### macOS Apple Silicon (M1/M2/M3)
```bash
docker run -it --rm \
  --platform linux/amd64 \
  -v /path/to/models_cv:/open_explorer \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

### GitHub Codespaces (Recommended — no local setup required)
```bash
docker run -it --rm \
  -v /workspaces/models_cv:/open_explorer \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

Verify toolchain is working:
```bash
hb_mapper --version   # 1.24.3
hbdk-cc --version     # 3.49.15
```

---

## Step 6: Export ONNX from PyTorch

Can be done on macOS or Google Colab — no Docker required.

```bash
pip install ultralytics onnx onnxsim
```

### Export YOLOv8
```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', opset=11, simplify=True, imgsz=640)
"
```

### Export custom model (e.g. MobileNetV3 Gender)
See the [Fine-tuning on Google Colab](#fine-tuning-on-google-colab) section below.

---

## Step 7: Prepare Calibration Data

Calibration data must be raw binary uint8 RGB files.

```python
import cv2, numpy as np, os

src = "data/calibration"           # source JPG images
dst = "data/calibration_processed" # output binary files
os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):
    if not f.endswith('.jpg'): continue
    img = cv2.imread(os.path.join(src, f))
    img = cv2.resize(img, (640, 640))           # adjust size per model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
    img.astype(np.uint8).tofile(
        os.path.join(dst, f.replace('.jpg', '.bin'))
    )
```

Required calibration sizes per model:

| Model | Size | Folder |
|-------|------|--------|
| yolov8s, yolov8n-face, yolov8s-pose | 640×640 | `calibration_processed/` |
| genderage | 96×96 | `calibration_processed_96/` |
| headpose | 224×224 | `calibration_processed_224/` |
| emotion | 64×64 grayscale | `calibration_processed_64/` |
| osnet_reid | 256×128 | `calibration_processed_256x128/` |
| csrnet | 320×320 | `calibration_processed_320/` |
| gender_mobilenetv3 | 224×224 float32 ImageNet-normalized | `calibration_gender_float/` |

For float32 calibration (gender_mobilenetv3 only):
```python
mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
std  = np.array([58.395,  57.12,  57.375], dtype=np.float32)
img  = (img.astype(np.float32) - mean) / std
img.tofile(dst_path)  # save as float32 binary
```

---

## Step 8: Convert ONNX → .bin (inside Docker)

```bash
hb_mapper makertbin \
  --model-type onnx \
  --config models/configs/<model_name>.yaml \
  --model models/onnx/<model_name>.onnx
```

### Example YAML — NV12 pyramid input (camera direct)
```yaml
model_parameters:
  onnx_model: "/open_explorer/models/onnx/yolov8s.onnx"
  output_model_file_prefix: "yolov8s"
  march: "bayes-e"

input_parameters:
  input_name: "images"
  input_shape: "1x3x640x640"
  input_type_rt: "nv12"
  input_layout_rt: "NHWC"
  input_type_train: "rgb"
  input_layout_train: "NCHW"
  norm_type: "data_scale"
  scale_value: 0.003921568627

calibration_parameters:
  cal_data_dir: "/open_explorer/data/calibration_processed"
  calibration_type: "default"

compiler_parameters:
  compile_mode: "latency"
  debug: false
  optimize_level: "O3"
```

### Example YAML — featuremap DDR input (gender_mobilenetv3)
```yaml
model_parameters:
  onnx_model: "/open_explorer/models/onnx/gender_mobilenetv3.onnx"
  output_model_file_prefix: "gender_mobilenetv3"
  march: "bayes-e"

input_parameters:
  input_name: "input"
  input_shape: "1x3x224x224"
  input_type_rt: "featuremap"
  input_layout_rt: "NCHW"
  input_type_train: "featuremap"
  input_layout_train: "NCHW"
  norm_type: "no_preprocess"

calibration_parameters:
  cal_data_dir: "/open_explorer/data/calibration_gender_float"
  cal_data_type: "float32"
  calibration_type: "default"

compiler_parameters:
  compile_mode: "latency"
  debug: false
  optimize_level: "O3"
```

Move output to correct folder:
```bash
mv models/configs/model_output/<model_name>.bin models/bin/
cp models/bin/<model_name>.bin models/hbm/<model_name>.hbm
```

---

## Step 9: Copy Models to RDK X5 Board

```bash
# Copy all models at once
scp models/bin/*.bin supersensor@172.20.10.2:/home/supersensor/models/

# Or copy a single model
scp models/bin/yolov8s.bin supersensor@172.20.10.2:/home/supersensor/models/
```

SSH into the board:
```bash
ssh supersensor@172.20.10.2
```

---

## Step 10: Test Models on RDK X5 Board

### Check model info
```bash
hrt_model_exec model_info --model_file=models/yolov8s.bin
```

### Benchmark latency (single thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=1 \
  --frame_count=200
```

### Benchmark FPS (multi thread)
```bash
hrt_model_exec perf \
  --model_file=models/yolov8s.bin \
  --thread_num=4 \
  --frame_count=200
```

### Test all models at once
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

### Set CPU to performance mode (for best benchmark results)
```bash
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
```

---

## Fine-tuning on Google Colab

For fine-tuning the gender_mobilenetv3 model on the UTKFace dataset:

1. Open [Google Colab](https://colab.research.google.com)
2. Change runtime: **Runtime → Change runtime type → T4 GPU → Save**
3. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
4. Install dependencies:
```python
!pip install ultralytics timm onnx onnxsim -q
```
5. Download UTKFace dataset:
```python
import os
os.environ['KAGGLE_API_TOKEN'] = 'YOUR_KAGGLE_API_TOKEN'
!pip install kaggle -q
!kaggle datasets download -d jangedoo/utkface-new
!unzip -q utkface-new.zip -d /content/utkface
```
6. Fine-tune and export ONNX following the training script
7. Download the `.onnx` file and convert to `.bin` using the Docker toolchain

---

## Troubleshooting

### Docker: Out of Memory (process Terminated during calibration)
```bash
# Reduce input size or ensure calibration uses batch_size=1
# Check available RAM before conversion
free -h
# Recommended: at least 8GB free RAM for 640x640 models
```

### Low Cosine Similarity after conversion
The most common cause is a mismatch between calibration data and training data domain.
```bash
# Use calibration images from the same domain as training data
# e.g., gender model  → use real face crops, not generic COCO images
# e.g., csrnet        → use crowd scene images
# e.g., emotion model → use face images not random photos
```

### hbdk-cc crash: "calculate size exceed peak dim"
```bash
# Change optimize_level from O3 to O1
# Known issue with yolov8s-pose only
optimize_level: "O1"
```

### ONNX opset version mismatch (need opset 11)
```python
import onnx
from onnx import version_converter
model = onnx.load('model.onnx')
model = version_converter.convert_version(model, 11)
onnx.save(model, 'model_op11.onnx')
print('Done: opset', model.opset_import[0].version)
```

### Docker not found on Codespaces
```bash
sudo service docker start
sudo docker run ...
```

### NHWC vs NCHW layout conflict error
```bash
# PyTorch models   → input_layout_train: "NCHW"
# TensorFlow models (via tf2onnx) → input_layout_train: "NHWC"
# Check with: python3 -c "import onnx; m=onnx.load('model.onnx'); print(m.graph.input)"
```

---

## Toolchain Versions

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
| Python | 3.12 |
| Board OS | Ubuntu 22.04 aarch64 |
| Board IP | 172.20.10.2 |
| Board User | supersensor |