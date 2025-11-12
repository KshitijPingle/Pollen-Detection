import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2d, BatchNormalization, LeakyRelu, MaxPooling2d, Reshape
from tensorflow.keras.models import Model


# Input Shape
# Extracted image will be 1920x1080 pixels 
#   Standard YOLO input resolution is 416x416 pixels
#   Input image will be reshaped to 
input_shape = (1920, 1080, 3)       # 3 = three color channels (R, G, B)

# Input shape needs to be downsized
#   Resize to a more popular dimension
#   Maintain original resolution and aspect ratio
#   Also try with the original
#   Use tensorflow resize librabries
#       Try the 4k resolution as well (which operates at 30 fps)
#       HD vs 4K comparison

# 

# Fixed Dimenison Resizing OR Actual



# Output Shape
#   Output Shape dependeds on 
#       Grid Size: usually is 13x13, 26x26, 52x52 and depends on input resolution and downsampling
#       Number of Anchors per grid: Usually 3
#       Number of classes: 2 classes for me (pollen, and bee)
#       Each prediction vector: 
#           [tx, ty, tw, th, objectness, class_probs, etc.] => Total Length = 6 + num_classes = 5 + 2 = 8
#               tx, ty, tw, th are box coordinates

output_shape = (26, 26, 3, 8)
#   26x26 grid size
#   3 anchors
#   8 = 6 + 2
#       2 = two classes (pollen, bee)
#       6 = number of items in each prediction vector
#           [tx, ty, tw, th, objectness, class_probs, etc.]
# Input size determines the output grid size
#   Search for formula
#       Ex. Assume 128x128 input; 
#   3 anchors is okay

#       Last output can have many arguments, we just won't use all of them





# Training input
#   Array with image data
#       includes URL to image, annotations in image, image metadata, etc.
#   So, input shape should also include the num_of_samples as a dimension
#   Input Shape = (num_samples, height, width, channels)
#               = (num_of_images, 1080, 1920, 3)            1920 x 1080 pixel image with 3 colors (RGB)

# Example Code
import numpy as np
from PIL import Image # For loading images

# Assuming you have a list of image paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
target_height = 1080
target_width = 1920
images_list = []

for path in image_paths:
    img = Image.open(path).convert('RGB') # Load as RGB
    img = img.resize((target_width, target_height)) # Resize
    img_array = np.array(img) # Convert to NumPy array
    images_list.append(img_array)

# Stack the images to create the 4D array
X_train = np.array(images_list)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0

print(f"Shape of X_train: {X_train.shape}")