# train_bee_model.py : Train a bee detection model from image and
#                      label dataset
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

import os
from ultralytics import YOLO

from train_yolo_model import train_yolo_model
from split_dataset import split_dataset
from make_background_img_labels import prepare_background_data
from inference_and_save_img import inference_and_save_img


dataset_dir = "datasets/bee_dataset_5"

# List of all classes this model should recognize
classes = ["Bee"]

# Split dataset into train, validate, and test sets
split_dataset(dataset_dir, classes, include_test = True)

data_yaml_path = f'{dataset_dir}/data.yaml'

print()
print()

model = YOLO('yolov8n.pt')
results = model.train(data=data_yaml_path, epochs=50, imgsz=1920, batch=4)

# results, model = train_yolo_model(data_yaml_path, epochs=50, imgsz=1920, batch=8, save_model=False)
model.info

print()
print()

# Get model metrics
metrics = model.val()
print(f"Mean Average Precision (mAP@50): {metrics.box.map50}")
print(f"Mean Average Precision (mAP@50-95): {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")  # Mean Precision
print(f"Recall: {metrics.box.mr}")     # Mean Recall


# Test detection results with an image with pollen
print()
print("The following are detection results for an image with pollen:")
pred_results = model.predict(source='RO3_1_frame_144.jpg')
for result in pred_results:
    boxes = result.boxes
    for box in boxes:
        print(f"Object Detected: Class {int(box.cls[0])}, Confidence: {box.conf[0]:.2f}, Box coords: {box.xyxy[0]}")


# Run inference on images then save them
inference_and_save_img(model, "test_image_1.jpg")            # Output saved to 'test_image_1_output.jpg'
inference_and_save_img(model, "test_image_2.jpg")            # Output saved to 'test_image_2_output.jpg'