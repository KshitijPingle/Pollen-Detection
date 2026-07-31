# bee_crop_dataset.py : Make new cropped bee dataset using bee detection computer
#                       vision model. Cropped bee dataset is used to train the 
#                       pollen detection model.
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
# Define your source and destination roots
base_source = 'datasets/pollen_train_test_val'
base_dest = 'datasets/bee_crops'

# List the splits you want to process
splits = ['train', 'val', 'test']

# Load your BEE model
bee_model = YOLO('runs/detect/train31/weights/best.pt')

def convert_obb_to_crop(obb_label, bee_box):
    """Translates 8-point OBB coordinates to a local crop."""
    cls = int(obb_label[0])
    coords = np.array(obb_label[1:]).reshape(4, 2)
    
    # 1. Convert to absolute pixels (1920x1080)
    coords[:, 0] *= 1920
    coords[:, 1] *= 1080
    
    # 2. Shift relative to crop top-left
    coords[:, 0] -= bee_box[0]
    coords[:, 1] -= bee_box[1]
    
    crop_w = bee_box[2] - bee_box[0]
    crop_h = bee_box[3] - bee_box[1]
    
    # 3. Check if center of OBB is inside the crop
    center = coords.mean(axis=0)
    if 0 <= center[0] <= crop_w and 0 <= center[1] <= crop_h:
        # Re-normalize to new crop size
        coords[:, 0] /= crop_w
        coords[:, 1] /= crop_h
        return [cls] + coords.flatten().tolist()
    return None

def process_split(split_name):
    img_dir = os.path.join(base_source, 'images', split_name)
    lbl_dir = os.path.join(base_source, 'labels', split_name)
    save_img_dir = os.path.join(base_dest, 'images', split_name)
    save_lbl_dir = os.path.join(base_dest, 'labels', split_name)
    
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    print(f"\n--- Processing {split_name} split ---")
    
    # Check if directory exists before processing
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} not found, skipping...")
        return

    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None: continue
        
        lab_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(lab_path): continue
        
        with open(lab_path, 'r') as f:
            global_labels = [list(map(float, line.split())) for line in f.readlines()]

        # Find the bees to define crop zones
        results = bee_model(img, conf=0.4, imgsz=1920, verbose=False)
        
        for i, box in enumerate(results[0].boxes):
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            
            # Padding (20%)
            pw, ph = (bx2-bx1)*0.2, (by2-by1)*0.2
            bx1, by1 = max(0, int(bx1-pw)), max(0, int(by1-ph))
            bx2, by2 = min(1920, int(bx2+pw)), min(1080, int(by2+ph))
            
            bee_crop = img[by1:by2, bx1:bx2]
            new_obb_labels = []
            for g_lab in global_labels:
                converted = convert_obb_to_crop(g_lab, [bx1, by1, bx2, by2])
                if converted: new_obb_labels.append(converted)
                
            if new_obb_labels:
                base_name = os.path.splitext(img_name)[0]
                cv2.imwrite(os.path.join(save_img_dir, f"{base_name}_bee_{i}.jpg"), bee_crop)
                with open(os.path.join(save_lbl_dir, f"{base_name}_bee_{i}.txt"), 'w') as f:
                    for nl in new_obb_labels:
                        f.write(f"{nl[0]} " + " ".join([f"{coord:.6f}" for coord in nl[1:]]) + "\n")

if __name__ == '__main__':
    for split in splits:
        process_split(split)