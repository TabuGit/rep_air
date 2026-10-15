from ultralytics import YOLO

model = YOLO("./models/yolov8n.pt")
results = model("video.mp4", save=True)

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"{model.names[cls_id]}: {conf:.2f}")