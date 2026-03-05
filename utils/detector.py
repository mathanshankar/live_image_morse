# YOLOv8 detection module

from ultralytics import YOLO

# Load YOLOv8 Nano (CPU)
model = YOLO("models/yolov8n.pt")
model.fuse()


def detect_object(frame):
    results = model(
        frame,
        device="cpu",
        imgsz=416,
        conf=0.5,
        verbose=False
    )

    for r in results:
        if r.boxes is not None and len(r.boxes.cls) > 0:
            box = r.boxes[0]
            cls_id = int(box.cls[0])
            return model.names[cls_id], box.xyxy[0]

    return "None", None
