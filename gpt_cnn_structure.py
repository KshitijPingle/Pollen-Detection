import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization,
    LeakyReLU, MaxPooling2D, Reshape
)
from tensorflow.keras.models import Model

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




def yolo_loss(y_true, y_pred):
    # Extract masks, box true/pred, class true/pred
    # Compute each component
    # return total_loss
    pass

model.compile(
    optimizer='adam',
    loss=yolo_loss
)
