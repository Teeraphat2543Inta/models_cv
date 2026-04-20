#!/bin/bash
# Convert ONNX → .bin สำหรับ RDK X5
# รันใน D-Robotics Docker container

ONNX_DIR="../models/onnx"
BIN_DIR="../models/bin"
CONFIG_DIR="../models/configs"

mkdir -p "$BIN_DIR"

MODELS=("yolov8s" "yolov8n-face" "yolov8s-pose")

for MODEL in "${MODELS[@]}"; do
    echo "Converting: $MODEL"
    hb_mapper makertbin \
        --config "$CONFIG_DIR/${MODEL}.yaml" \
        --model-type onnx \
        --model "$ONNX_DIR/${MODEL}.onnx" \
        --output-dir "$BIN_DIR"
    echo "✅ Done: $MODEL"
done

echo "✅ All models converted! ไฟล์อยู่ที่ models/bin/"
