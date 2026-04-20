#!/bin/bash
# ทดสอบ inference บน RDK X5
# รันบนบอร์ดโดยตรง

BIN_DIR="../models/bin"
TEST_IMG="../data/test_images/test.jpg"

MODELS=(
    "yolov8s"
    "yolov8n-face"
    "yolov8s-pose"
)

echo "🔍 ตรวจสอบ model info..."
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- $MODEL ---"
    hrt_model_exec model_info \
        --model_file="$BIN_DIR/${MODEL}.bin"
done

echo ""
echo "⚡ ทดสอบ latency (single thread)..."
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- $MODEL ---"
    hrt_model_exec perf \
        --model_file="$BIN_DIR/${MODEL}.bin" \
        --thread_num=1 \
        --frame_count=100
done

echo ""
echo "🚀 ทดสอบ FPS (multi thread)..."
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "--- $MODEL ---"
    hrt_model_exec perf \
        --model_file="$BIN_DIR/${MODEL}.bin" \
        --thread_num=8 \
        --frame_count=200
done
