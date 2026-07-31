# metrics.py : Functions to output model metrics
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

import os
from ultralytics import YOLO
import pandas as pd
import cv2
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from inference_and_save_img import *


# Code from Gemini
def evaluate_test_metrics(model_path, data_config, results_csv_path):
    """
    Evaluates a trained YOLO OBB model on Test sets.
    
    Args:
        model_path (str): Path to the trained .pt weights (e.g., 'runs/obb/train/weights/best.pt').
        data_config (str): Path to the dataset .yaml file.
        results_csv_path (str): Path to the results.csv file. Contains information about the best performing metrics every epoch

    Importance:
        Results obtained from 'model.val()' might contain data leaked images, where the model is testing itself on "twins" of the training images.
        This function tests the trained models on truly unseen data.
    """

    # 1. Initialize the model
    model = YOLO(model_path)

    # 2. Load Training History from CSV
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Could not find {results_csv_path}")
    
    df_history = pd.read_csv(results_csv_path)
    # Strip whitespace from column names just in case
    df_history.columns = df_history.columns.str.strip()
    
    # Find the best epoch based on mAP50
    best_idx = df_history['metrics/mAP50(B)'].idxmax()
    best_row = df_history.iloc[best_idx]
    
    
    # 3. Run Validation
    # The 'val' split is typically used to tune hyperparameters
    print("--- Starting Validation Split Evaluation ---")
    val_results = model.val(
        data=data_config,
        split='val',
        batch=16,
        imgsz=640,
        save_json=True,  # Useful for COCO-style analysis
        name='val_results'
    )
    
    # 4. Run Testing
    # The 'test' split provides the final unbiased performance metrics for the paper
    print("\n--- Starting Test Split Evaluation ---")
    test_results = model.val(
        data=data_config,
        split='test',
        batch=16,
        imgsz=640,
        save_json=True,
        # project=project_name,
        name='test_results'
    )
    
    # 5. Consolidate Metrics
    # Use the .obb for YOLOv8n-OBB specific results
    metrics_summary = {
        "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Train (Best Epoch)": [
            best_row['metrics/precision(B)'],
            best_row['metrics/recall(B)'],
            best_row['metrics/mAP50(B)'],
            best_row['metrics/mAP50-95(B)']
        ],
        "Validation (Final)": [
            val_results.box.mp, 
            val_results.box.mr, 
            val_results.box.map50, 
            val_results.box.map
        ],
        "Test (Unseen)": [
            test_results.box.mp, 
            test_results.box.mr, 
            test_results.box.map50, 
            test_results.box.map
        ]
    }
    
    # df = pd.DataFrame(metrics_summary)
    # print("Test Results Summary:")
    # print(df.to_string(index=False))

    summary_df = pd.DataFrame(metrics_summary)
    
    # 4. Display results
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(summary_df.to_string(index=False))


    # Draw inference on test images
    inference_and_save_img(model, "test_image_1.jpg")            # Output saved to 'test_image_1_output.jpg'
    inference_and_save_img(model, "test_image_2.jpg")            # Output saved to 'test_image_2_output.jpg'
    inference_and_save_img(model, "08-59-47_March_27_2026_frame_233.png")
    
    # # Save summary to CSV for inclusion in research logs
    # df.to_csv(f"{project_name}/final_comparison.csv", index=False)
    # print(f"\nResults saved to {project_name}/")

bee_model_weight = "runs/detect/train31/weights/best.pt"
bee_model_results_csv = "runs/detect/train31/results.csv"

pollen_model_weight = "runs/detect/train34/weights/best.pt"
pollen_model_results_csv = "runs/detect/train34/results.csv"

bee_crop_pollen_weight = "runs/detect/train36/weights/best.pt"
bee_crop_pollen_results_csv = "runs/detect/train36/results.csv"

# # Evaluate val and test metrics
evaluate_test_metrics(bee_model_weight, "/home/kshitij/Bees/label-studio-ml-backend/datasets/bee_dataset_5/data.yaml", bee_model_results_csv)

# print()
# print()
evaluate_test_metrics(pollen_model_weight, "/home/kshitij/Bees/label-studio-ml-backend/datasets/pollen_train_test_val/data.yaml", pollen_model_results_csv)

# print()
# print()
evaluate_test_metrics(bee_crop_pollen_weight, "/home/kshitij/Bees/label-studio-ml-backend/datasets/bee_crops/bee_pollen.yaml", bee_crop_pollen_results_csv)



def evaluate_with_sahi_standard(model_path, image_dir):
    """
    Standard SAHI evaluation: No forced image_size, no cropping.
    Uses the model's internal defaults.
    """
    # 1. Initialize SAHI Model
    # Removing 'image_size' lets the model use its default (1920 from your training)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.55,
        device="cuda:0",
        image_size=640
    )

    # 2. Get Image Paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"--- Running Standard SAHI on {len(image_paths)} images ---")
    
    for img_path in tqdm(image_paths):
        # Load and convert to RGB (Crucial to avoid the discoloration)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Run Sliced Prediction on the FULL image
        result = get_sliced_prediction(
            image=img_rgb,
            detection_model=detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS",
            postprocess_match_threshold=0.5
        )
        
        # 4. Export results
        output_name = os.path.basename(img_path).split('.')[0]
        result.export_visuals(export_dir="sahi_reports/", file_name=output_name)

    print("\nInference complete. Visuals saved to 'sahi_reports/'")


# evaluate_with_sahi_standard(pollen_model_weight, "test_images")