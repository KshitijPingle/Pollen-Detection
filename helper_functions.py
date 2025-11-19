import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Optional: for building dataset_index from COCO
try:
    from pycocotools.coco import COCO
except Exception:
    COCO = None


# -------------------------
# Utility: draw boxes
# -------------------------
def draw_boxes_on_image_bgr(img_bgr, boxes, class_names=None, box_color=(0, 255, 0), thickness=2):
    """
    img_bgr: HxWx3 BGR image (as returned by cv2.imread)
    boxes: list of [x1, y1, x2, y2, class_id]
    class_names: list or dict mapping class_id->name (optional)
    returns: copy of img_bgr with drawings
    """
    img = img_bgr.copy()
    for box in boxes:
        if len(box) < 5:
            continue
        x1, y1, x2, y2, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        label = str(class_id) if class_names is None else class_names.get(class_id, str(class_id))
        # background rectangle for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# -------------------------
# Display helper
# -------------------------
def show_images_grid(images_bgr, titles=None, cols=2, figsize=(12, 8)):
    """
    images_bgr: list of images in BGR (cv2) format
    displays them with matplotlib after converting to RGB
    """
    n = len(images_bgr)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, img_bgr in enumerate(images_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img_rgb)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


# -------------------------
# Core function
# -------------------------
def visualize_dataset_index(dataset_index, n_examples=6, class_names=None, random_seed=42):
    """
    dataset_index: list of tuples (image_path, boxes),
                   boxes: list of [x1,y1,x2,y2,class_id]
    n_examples: number of images to display
    class_names: optional dict {class_id: name}
    """
    rng = np.random.default_rng(random_seed)
    if len(dataset_index) == 0:
        raise ValueError("dataset_index is empty")

    indices = rng.choice(len(dataset_index), size=min(n_examples, len(dataset_index)), replace=False)
    images_to_show = []
    titles = []
    for idx in indices:
        image_path, boxes = dataset_index[idx]
        if not os.path.exists(image_path):
            print(f"Image not found, skipping: {image_path}")
            continue
        img = cv2.imread(image_path)  # BGR format
        if img is None:
            print(f"Failed to read image, skipping: {image_path}")
            continue
        drawn = draw_boxes_on_image_bgr(img, boxes, class_names=class_names)
        images_to_show.append(drawn)
        titles.append(os.path.basename(image_path))
    if len(images_to_show) == 0:
        print("No valid images to show.")
        return
    show_images_grid(images_to_show, titles=titles, cols=2)


# -------------------------
# Optional: build dataset_index from COCO
# -------------------------
def build_dataset_index(images_dir, ann_file, category_map=None, skip_crowd=True, limit=None):
    """
    images_dir: directory with COCO images (train2017 or val2017)
    ann_file: path to COCO annotation JSON
    category_map: optional dict mapping coco_cat_id -> contiguous class_id
    Returns: list of (image_path, boxes) where boxes are [x1,y1,x2,y2,class_id]
    Requires pycocotools installed.
    """
    if COCO is None:
        raise RuntimeError("pycocotools is not available. Install with `pip install pycocotools`.")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    dataset_index = []
    for i, img_id in enumerate(img_ids):
        if limit is not None and len(dataset_index) >= limit:
            break
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        boxes = []
        for a in anns:
            if skip_crowd and a.get('iscrowd', 0) == 1:
                continue
            x, y, w, h = a['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            coco_cat_id = a['category_id']
            class_id = category_map[coco_cat_id] if category_map is not None else coco_cat_id
            boxes.append([x1, y1, x2, y2, class_id])
        if len(boxes) > 0:
            dataset_index.append((image_path, boxes))
    return dataset_index


# -------------------------
# Example usage
# -------------------------
# if __name__ == "__main__":
#     # Option A: If you already have dataset_index in memory, use it directly:
#     # Example dummy dataset_index entry for quick local test:
#     # dataset_index = [("path/to/image1.jpg", [[34,45,210,330,1], [120,120,200,200,2]]), ...]
#     dataset_index = []

#     # If you want to test using a small slice of COCO, uncomment and set paths:
#     # COCO_IMAGE_DIR = "/path/to/train2017"
#     # COCO_ANN_FILE  = "/path/to/instances_train2017.json"
#     # if COCO is not None:
#     #     dataset_index = build_dataset_index_from_coco(COCO_IMAGE_DIR, COCO_ANN_FILE, limit=20)

#     # Otherwise create a tiny test index (use your own images)
#     # Example: use local images in ./examples/ and dummy boxes for visualization testing
#     sample_dir = "./examples"
#     if os.path.isdir(sample_dir):
#         for fname in os.listdir(sample_dir)[:10]:
#             p = os.path.join(sample_dir, fname)
#             if not os.path.isfile(p):
#                 continue
#             # Dummy box for visibility: use center 50%-width/height
#             h, w = cv2.imread(p).shape[:2]
#             boxes = [[w*0.25, h*0.25, w*0.75, h*0.75, 0]]
#             dataset_index.append((p, boxes))

#     # If dataset_index remains empty, instruct user
#     if len(dataset_index) == 0:
#         print("dataset_index is empty. Populate dataset_index with (image_path, boxes) entries or "
#               "set sample_dir with images or enable the COCO helper in the script.")
#     else:
#         # Optional: provide human readable class names as dict {class_id: name}
#         class_names = {0: "object"}
#         visualize_dataset_index(dataset_index, n_examples=6, class_names=class_names)