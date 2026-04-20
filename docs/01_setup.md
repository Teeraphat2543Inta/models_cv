# 01 - Setup Guide

## Step 1: ติดตั้ง Docker บน macOS
1. ดาวน์โหลด Docker Desktop จาก https://www.docker.com/products/docker-desktop
2. ติดตั้งและเปิดโปรแกรม
3. ตรวจสอบ:
```bash
docker --version
```

## Step 2: เช็ค chip Mac
```bash
uname -m
# arm64  = M1/M2/M3 → ต้องใช้ --platform linux/amd64
# x86_64 = Intel Mac → ใช้ได้ปกติ
```

## Step 3: Pull D-Robotics Docker Image
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.6
```

หากใช้ Apple Silicon (M1/M2/M3):
```bash
docker pull --platform linux/amd64 openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.6
```

## Step 4: รัน Docker Container
```bash
docker run -it --rm \
  --platform linux/amd64 \
  -v /Users/teeraphat/Desktop/my_models/New_models:/workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.6 /bin/bash
```

## Step 5: Export ONNX (ทำบน macOS ได้เลย ไม่ต้องใช้ Docker)
```bash
pip install ultralytics
cd /Users/teeraphat/Desktop/my_models/New_models/rdkx5_vision
python scripts/export_yolo.py
```

## Step 6: Convert ONNX → .bin (ทำใน Docker)
```bash
# ใน Docker container
cd /workspace/rdkx5_vision
bash scripts/convert_model.sh
```

## Step 7: ทดสอบบนบอร์ด (เมื่อมีบอร์ด)
```bash
# copy ไฟล์ไปบอร์ด
scp -r models/bin/ root@<BOARD_IP>:/userdata/models/

# รันบนบอร์ด
bash scripts/test_inference.sh
```
