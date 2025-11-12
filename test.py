import tensorflow_datasets as tfds






# Testing GPT CNN model with CIFAR data


# Load the training split
ds_train, ds_info = tfds.load(name="coco/2017", split="train", shuffle_files=True, with_info=True)

# Load the validation split
ds_test = tfds.load(name="coco/2017", split="validation", shuffle_files=False, with_info=True)

# You can now use ds_train and ds_test for your model
