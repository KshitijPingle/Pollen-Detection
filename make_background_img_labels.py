# make_background_img_labels.py : Helper functions to prepare bbackground data.
#                                 Background data is good for training computer vision
#                                 models to lower false positives
# By Kshitij Pingle
# pinglekshitij15@gmail.com
# 30 July 2026

# Last Modified : 30 July 2026



import os
import shutil

def prepare_background_data(bg_source_images, dataset_img_dir, dataset_lbl_dir):
    """ Function to copy over background imgs and make empty labels """

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Create destination folders if they don't exist
    os.makedirs(dataset_img_dir, exist_ok=True)
    os.makedirs(dataset_lbl_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(bg_source_images):
        if filename.lower().endswith(image_extensions):
            file_base = os.path.splitext(filename)[0]
            
            # Define paths
            src_path = os.path.join(bg_source_images, filename)
            dst_path = os.path.join(dataset_img_dir, filename)
            label_path = os.path.join(dataset_lbl_dir, f"{file_base}.txt")
            
            # Copy the image to the dataset folder
            shutil.copy2(src_path, dst_path)
            
            # Create the empty label file
            with open(label_path, 'w') as f:
                pass 
            
            count += 1
            # print(f"Processed: {filename}")

    print(f"\n--- Finished! ---")
    print(f"Total background images added: {count}")
    print(f"Images moved to: {dataset_img_dir}")
    print(f"Empty labels created in: {dataset_lbl_dir}")    

    print(f"Done! Created {count} empty label files for your background images.")