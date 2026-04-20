# RDK X5 Vision Pipeline
AI Vision Pipeline สำหรับ RDK X5 + Pi Camera Module 2

## Features
- Person Detection (YOLOv8s)
- Face Detection (YOLOv8n-face)
- Pose Estimation (YOLOv8s-pose)
- Age & Gender Analysis (InsightFace)

## โครงสร้างโฟลเดอร์
```
rdkx5_vision/
├── models/
│   ├── onnx/       ← ไฟล์ .onnx หลัง export จาก PyTorch
│   ├── bin/        ← ไฟล์ .bin หลัง convert ด้วย D-Robotics toolchain
│   └── configs/    ← config YAML สำหรับ model conversion
├── scripts/
│   ├── export_yolo.py       ← Export PyTorch → ONNX
│   ├── convert_model.sh     ← Convert ONNX → .bin
│   └── test_inference.sh    ← ทดสอบบนบอร์ด
├── src/
│   └── pipeline/
│       ├── main.cc          ← Main pipeline
│       ├── detector.cc      ← Person & Face detection
│       ├── pose.cc          ← Pose estimation
│       └── face_attr.cc     ← Age & Gender
├── data/
│   ├── test_images/    ← รูปสำหรับทดสอบ
│   └── datasets/       ← datasets สำหรับ evaluate
├── output/
│   ├── logs/           ← inference logs
│   └── results/        ← ผลลัพธ์รูปภาพ
└── docs/               ← เอกสาร
```

## Getting Started
ดู docs/01_setup.md
