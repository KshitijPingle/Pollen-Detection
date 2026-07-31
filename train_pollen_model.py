# train_pollen_model.py : Train a pollen detection model from image and
#                         label dataset
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

import os
from ultralytics import YOLO
from split_dataset import split_dataset
from make_background_img_labels import prepare_background_data
from metrics import evaluate_test_metrics

# This file is to train the Pollen Detection model
dataset_dir = "datasets/pollen_train_test_val"

# List of all classes this model should recognize
classes = ["Pollen"]

# Make background image labels with no pollen objects
bg_source_images = 'datasets/pollen_background_images'        # Source: Where your raw background photos are currently sitting

# Destination: Your actual training dataset folders
dataset_img_dir = f'{dataset_dir}/images'
dataset_lbl_dir = f'{dataset_dir}/labels'


prepare_background_data(bg_source_images, dataset_img_dir, dataset_lbl_dir)

# Split dataset into train, validate, and test sets
split_dataset(dataset_dir, classes, include_test = True)


data_yaml_path = f'{dataset_dir}/data.yaml'

print()
print()

model = YOLO('yolov8n.pt')
results = model.train(data=data_yaml_path, epochs=50, imgsz=1920, batch=4, box=10.0, cls=2.0)        # default box = 7.5, default cls = 1.0

# Using a higher "box" loss weight value
#   Increase box loss weight for smaller objects
#   Tells the model that an angular error on small pollen grains is a huge mistake
#   Forces weight to settle on much more precise numbers

# Using a higher Cls weight value
#   Helps with finding smaller "dots"



model.info

print()
print()

# Get model metrics
pollen_model_weight = "runs/detect/train34/weights/best.pt"
pollen_data_yaml = "/home/kshitij/Bees/label-studio-ml-backend/datasets/pollen_train_test_val/data.yaml"
evaluate_test_metrics(pollen_model_weight, data_yaml_path)
