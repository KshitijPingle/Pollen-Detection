# train_bee_crop_pollen.py : Train a pollen detection model from image and
#                           label dataset to detect pollen from cropped bee
#                           images
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

from ultralytics import YOLO

# Load your CURRENT weights
model = YOLO('runs/detect/train34/weights/best.pt') 

# Train on the NEW cropped dataset
model.train(
    data='datasets/bee_crops/bee_pollen.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    save=True,
    # project='pollen_research',
    # name='hierarchical_step2'
)