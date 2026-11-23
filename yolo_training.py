from ultralytics import YOLO

# Modelo base
model = YOLO('yolov8n.pt')

# Entrenar
results = model.train(data='/home/jesus/Documents/Robótica/automodel_car/vision_codes/yolo_training/dataset_stop_sign/traffic_stop_sign/data.yaml', epochs=50, imgsz=640)