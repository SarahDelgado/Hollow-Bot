from ultralytics import YOLO

# Cargar modelo YOLOv8
model = YOLO("yolov8n.pt")

# Entrenar modelo usando data_character.yaml
model.train(data="dataset/data_character.yaml", epochs=50, imgsz=640)
