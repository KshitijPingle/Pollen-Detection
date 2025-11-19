import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization,
    LeakyReLU, MaxPooling2D, Reshape
)
from tensorflow.keras.models import Model
from pycocotools.coco import COCO
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

from helper_functions import *


def build_yolo_like_model(
    input_shape=(416, 416, 3),
    num_classes=20,
    anchors=3,
    grid_size=26
):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = MaxPooling2D((2, 2))(x)

    # Blocks 3–5 (128, 256, 512 filters)
    for filters in [128, 256, 512]:
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        if filters != 512:
            x = MaxPooling2D((2, 2))(x)

    # Block 6
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)

    # Detection head
    preds = Conv2D(
        anchors * (num_classes + 5),
        (1, 1),
        padding='same'
    )(x)

    outputs = Reshape(
        (grid_size, grid_size, anchors, num_classes + 5)
    )(preds)

    return Model(inputs, outputs)

# Instantiate
model = build_yolo_like_model(
    input_shape=(416, 416, 3),
    num_classes=80,   # e.g. COCO
    anchors=3,
    grid_size=26
)
model.summary()



# GPT CNN uses a repetition of BatchNormalization and LeakyRelu layers
#   Both the layers pair together to ensure stable and efficient training at every depth
#   BatchNormalization provides:
#       Reduces Internal Covariate shift, then learns optimal scale and shift parameters
#       Keep inputs in a stable range, speeding up convergence
#       Also acts like a small regularizer, reducing need for other forms of regularization like dropout
#   LeakyRelu provides:
#       Introduces small nonzero gradient for negative inputs
#       A negative slope behavior maintains gradient flow through all units, which is essential in deep architectures
#   Repeating both layers often provides:
#       Per block stability
#       Faster Training
#       Robust Deep Features


# Other advice
#   Consider using other normalization schemes (LayerNorm, GroupNorm)
#   Experiment with newer activations like Swich or Mish
#   For very deep models, investigate "pre-activation" ordering (BN -> Activation -> Conv), which is used in ResNet v2




# def yolo_loss(y_true, y_pred):
#     # Extract masks, box true/pred, class true/pred
#     # Compute each component
#     # return total_loss
#     pass

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError()
)


# # Initialize coco class by providing path to annotations
# coco = COCO('coco_datasets/image_info_test2017/annotations/image_info_test2017.json')

# Parse annotations
#   Build a list with each item being a 2-Tuple of the form (Image_path, annotations)
#   So, list = [(image_path_1, annotations_1), (image_path_2, annotations_2), (image_path_2, annotations_2), ... ]
#       Here, each annotation object should be of the form = [x_min, y_min, x_max, y_max, class_id]


# def build_dataset_index(images_dir, ann_file, category_map=None, skip_crowd=True):
#     coco = COCO(ann_file)
#     img_ids = coco.getImgIds()
#     dataset_index = []

#     for img_id in img_ids:
#         img_info = coco.loadImgs(img_id)[0]
#         file_name = img_info['file_name']
#         image_path = os.path.join(images_dir, file_name)

#         ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
#         anns = coco.loadAnns(ann_ids)

#         boxes = []
#         for a in anns:
#             if skip_crowd and a.get('iscrowd', 0) == 1:
#                 continue
#             x, y, w, h = a['bbox']                 # COCO bbox: x,y,w,h (pixels)
#             x1, y1, x2, y2 = x, y, x + w, y + h
#             coco_cat_id = a['category_id']
#             # Map COCO category ids to zero-based contiguous indices if provided
#             class_id = category_map[coco_cat_id] if category_map is not None else coco_cat_id
#             boxes.append([x1, y1, x2, y2, class_id])

#         # only keep images that exist and have at least one box (optional)
#         if os.path.exists(image_path) and len(boxes) > 0:
#             dataset_index.append((image_path, boxes))

#     return dataset_index

# training_folder = '../coco_datasets/train2017/train2017'
# training_annotations = '../coco_datasets/annotations_trainval2017/annotations/instances_train2017.json'

training_folder = '../CNN/coco_datasets/train2017/train2017'
training_annotations = '../CNN/coco_datasets/annotations_trainval2017/annotations/instances_train2017.json'
training_dataset_index = []

if os.path.isfile('training_index.txt'):
    with open('training_index.txt') as file:
        training_dataset_index = json.load(file)
else:
    training_dataset_index = build_dataset_index(training_folder, training_annotations) 
    # Write training dataset index to file
    with open('training_index.txt', 'w') as file:
        json.dump(training_dataset_index, file)

print("Length of training dataset index =", len(training_dataset_index))

# Visualize data
visualize_dataset_index(training_dataset_index)

subset_training_index = training_dataset_index[:10]
# Fit model to training data - Currently throwing a type error
#   ASK: If x input has to be images (list)
#        and y target label has to be annotations (list)
# model.fit(
#     subset_training_index,
#     epochs = 10,
#     # callbacks = [
#     # tf.keras.callbacks.ModelCheckpoint('checkpoints/yolo_epoch{epoch:02d}.h5', save_best_only=True, monitor='val_loss'),
#     # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
#     # tf.keras.callbacks.TensorBoard(log_dir='logs')
#     # ]
# )



# Tasks to complete:
#   ASK: Make a custom loss function to use when compiling - NO NEED
#   Download COCO train and test datasets - DONE
#   Make parser to load JSON data or use COCO's methods
#       I had chatgpt make me one in helper_function.py = build_dataset_index()
#   Optional: Visualize data
#       I had chatgpt make me functions for visualization and box drawing in helper_function.py
#   Make a dataloader using tensorflow or pytorch - LOOK into
