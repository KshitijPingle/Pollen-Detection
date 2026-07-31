# split_dataset.py : Split image and label datasets into training, 
#                    testing, and validation datasets
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026

import supervision as sv
import os
import yaml
from pathlib import Path
import random
import shutil

def delete_files_only(directory_path):
    """ Deletes all files in the specified directory but skips all subdirectories """

    # check if the directory actually exists
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        # Construct the full path of the item
        item_path = os.path.join(directory_path, item)

        # Check if it's a file (this automatically excludes directories)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            try:
                os.remove(item_path)
            except Exception as e:
                print(f"Failed to delete {item}: {e}")
        else:
            print(f"Skipping directory: {item}")

def split_dataset(dataset_dir, classes, split_ratio = 0.8, include_test = False, test_ratio = 0.1):
    """ Function to split a dataset into training, validation, and testing folders to be used by YOLO """

    # 'classes' is a list of Strings of all the classes the model recognizes

    # Define paths to your dataset
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "labels")
    test_dir = os.path.join(dataset_dir, "test")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")


    # Cleanup labels list
    labels = []
    for root, dirs, files in os.walk(annotations_dir):
        # 'root' is the current directory being visited
        # 'dirs' is a list of subdirectories in the current 'root'
        # 'files' is a list of files in the current 'root'

        for file_name in files:
            labels.append(file_name[:-4])

    # Remove unnecessary items from images folder (imgs without annotations)
    for _, _, files in os.walk(images_dir):
        for file_name in files:
            if (file_name[:-4] not in labels):
                os.remove(images_dir + '/' + file_name)

    # random_state=42 ensures that every time you run this, the "random" shuffle is the same.

    # Load your data into a supervision.DetectionDataset object
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=images_dir,
        annotations_directory_path=annotations_dir,
        data_yaml_path=data_yaml_path
    )

    if (include_test):
        # First split to train and temp, then split the temp into test and val
        train_ds, temp_ds = ds.split(split_ratio=split_ratio, random_state=42, shuffle=True)

        remaining_ratio = 1 - split_ratio
        internal_test_ratio = test_ratio / remaining_ratio
        
        # Second split: Split temp into val and test
        val_ds, test_ds = temp_ds.split(split_ratio= 1 - internal_test_ratio, random_state=42, shuffle=True)
        split_map = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    else:
        # Simply split into train and validation sets

        train_ds, val_ds = ds.split(split_ratio=split_ratio, random_state=42, shuffle=True)
        split_map = {'train': train_ds, 'val': val_ds}


    # Make subdirectories
    for set_name, dataset in split_map.items():
        img_path = os.path.join(dataset_dir, 'images', set_name)
        lbl_path = os.path.join(dataset_dir, 'labels', set_name)
        
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(lbl_path, exist_ok=True)
        
        dataset.as_yolo(
            images_directory_path=img_path,
            annotations_directory_path=lbl_path
        )
    
    # Delete extra files only from the img and annotations directories
    delete_files_only(images_dir)
    delete_files_only(annotations_dir)

    # Dynamically write to data.yaml file in dataset_dir folder
    data = {
        'train' : 'images/train',
        'val' : 'images/val',
        'nc' : len(classes),
        'names' : classes
    }

    if include_test:
        data['test'] = 'images/test'
        delete_files_only(test_dir)

    with open(f'{dataset_dir}/data.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)