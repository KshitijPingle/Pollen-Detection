# hierarchical_inference.py : File to run hierarchical inference to detect pollen. First detects bees,
#                             then crops around detected bees. Pollen detection occurs with the cropped
#                             bee images.
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

# Benefits of cropping first: Huge reduction of noise by eliminating background information not
#                             not contributing towards finding pollen. Leads to easier detection
#                             of small pollen baskets in proportionally huge 1080 X 1920 image

import cv2
import os
from ultralytics import YOLO

# Load existing models
bee_model = YOLO("runs/detect/train31/weights/best.pt")
pollen_model = YOLO("runs/detect/train34/weights/best.pt")
bee_cropped_model = YOLO('runs/detect/train36/weights/best.pt')

def run_hierarchical_inference(frame, count, output_dir):
    # Step 1: Find the bees (the "Containers" for pollen)
    bee_results = bee_model(frame, imgsz=1920)

    # Visualize
    annotated = bee_results[0].plot()

    # Save Image
    output_img = f"{output_dir}/bee_test_image_{count}_output.jpg"
    cv2.imwrite(output_img, annotated)

    pollen_results = pollen_model(frame, imgsz=1920)

    # Visualize
    annotated = pollen_results[0].plot()

    # Save Image
    output_img = f"{output_dir}/pollen_test_image_{count}_output.jpg"
    cv2.imwrite(output_img, annotated)
    
    pollen_detections = []

    labeler_count = 1
    for result in bee_results:
        for box in result.boxes:
            # Get bee coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Step 2: Crop the Bee from the original image
            # Add a small buffer (10px) to ensure we don't cut off the legs/pollen
            bee_crop = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]

            # Save the cropped images
            bee_crop_output_img = f"{output_dir}/bee_cropped_{labeler_count}_output.jpg"
            cv2.imwrite(bee_crop_output_img, bee_crop)
            
            # Step 3: Run Bee Cropped Pollen Model on the CROP
            # Since the bee is now the "whole world" to the model, 
            # those tiny pollen pixels become huge features.
            pollen_on_bee = bee_cropped_model(bee_crop, imgsz=640)
            

            # Visualize
            annotated = pollen_on_bee[0].plot()
            # Save Image
            output_img = f"{output_dir}/bee_cropped_pollen_{labeler_count}_output.jpg"
            cv2.imwrite(output_img, annotated)


            labeler_count = labeler_count + 1
            pollen_detections.append(pollen_on_bee)
            
    return pollen_detections


if (__name__ == '__main__'):
    test_images_dir = 'test_images'
    labeled_output_dir = 'test_output'
    pollen_detection = []

    count = 1
    for img_name in os.listdir(test_images_dir):
        full_path = os.path.join(test_images_dir, img_name)

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
        
        print("Image: ", img_name)

        frame = cv2.imread(full_path)

        results = run_hierarchical_inference(frame, count, labeled_output_dir)
        pollen_detection.append(results)
        count = count + 1
    
    # print(pollen_detection)