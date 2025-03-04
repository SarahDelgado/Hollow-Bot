from ultralytics import YOLO

# Cargar modelo YOLOv8
model = YOLO("yolov8n.pt")

# Entrenar modelo usando data.yaml
model.train(data="dataset/data.yaml", epochs=50, imgsz=640)
