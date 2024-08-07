"""
Train Yolov8 model on any given dataset
"""

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Path of dataset configuration file
data_path = "./data/License-Plate-Recognition/data.yaml"

# Train the model
results = model.train(data=data_path, epochs=10, imgsz=640, batch=16, device=0)