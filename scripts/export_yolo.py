"""
Export YOLOv8 models to ONNX format
รัน: python scripts/export_yolo.py
"""

from ultralytics import YOLO
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../models/onnx")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {"name": "yolov8s",       "weights": "yolov8s.pt",       "task": "person detection"},
    {"name": "yolov8n-face",  "weights": "yolov8n-face.pt",  "task": "face detection"},
    {"name": "yolov8s-pose",  "weights": "yolov8s-pose.pt",  "task": "pose estimation"},
]

def export_model(cfg):
    print(f"\n{'='*50}")
    print(f"Exporting: {cfg['name']} ({cfg['task']})")
    print(f"{'='*50}")
    try:
        model = YOLO(cfg["weights"])
        model.export(
            format="onnx",
            imgsz=640,
            opset=11,
            simplify=True,
        )
        src = cfg["weights"].replace(".pt", ".onnx")
        dst = os.path.join(OUTPUT_DIR, f"{cfg['name']}.onnx")
        os.rename(src, dst)
        print(f"✅ Saved to {dst}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    for m in MODELS:
        export_model(m)
    print("\n✅ Export complete! ไฟล์อยู่ที่ models/onnx/")
