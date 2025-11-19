import tensorflow_datasets as tfds
from pycocotools.coco import COCO




# # Testing GPT CNN model with CIFAR data


# # Load the training split
# #   Load only the first 5 percent
# ds_train, ds_info = tfds.load(name="coco/2017", split="train[:5%]", shuffle_files=True, with_info=True)

# # Load the validation split
# #   Load only the first 5 percent
# ds_test = tfds.load(name="coco/2017", split="validation[:5%]", shuffle_files=False, with_info=True)

# # You can now use ds_train and ds_test for your model

coco = COCO('coco_datasets/image_info_test2017/annotations/image_info_test2017.json')
