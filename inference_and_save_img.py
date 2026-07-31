# inference_and_save_img.py : Use computer vision models to detect objects, then save
#                             the image for comparison and evaluation
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

import os
from ultralytics import YOLO
import cv2


def inference_and_save_img(model, image):
    """ Run inference on an image with a model, then save that image """

    # Run inference on test image
    results = model(image)

    # Visualize
    annotated = results[0].plot()

    # # Display Image
    # cv2.imshow("YOLO Output", annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save Image
    output_img = f"{image[:-4]}_output.jpg"
    cv2.imwrite(output_img, annotated)

bee_and_pollen_model_weight = "runs/detect/train12/weights/best.pt"

pollen_model_weight = "runs/detect/train16/weights/best.pt"


model = YOLO(pollen_model_weight)

inference_and_save_img(model, "test_image_1.jpg")
inference_and_save_img(model, "test_image_2.jpg")
