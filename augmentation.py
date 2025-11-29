import os
import cv2
import glob
import albumentations as A
import numpy as np
import shutil

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=1.0),  # clockwise/counterclockwise 90°
    A.Rotate(limit=15, p=0.7),  # random -15° to +15°
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=0.7),  # -20% to +20% brightness
    A.Exposure(p=0.7, exposure_limit=(-0.1, 0.1)),  # -10% to +10% exposure
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_and_save(images_dir, labels_dir, save_images_dir, save_labels_dir, n_aug=3):
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    for img_path in image_files:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))

        # Read image and labels
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            lines = f.read().splitlines()
        bboxes, class_labels = [], []
        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.split())
            bboxes.append([xc, yc, bw, bh])
            class_labels.append(int(cls))

        # Save original
        shutil.copy(img_path, os.path.join(save_images_dir, filename))
        shutil.copy(label_path, os.path.join(save_labels_dir, filename.replace('.jpg', '.txt')))

        # Augment and save
        for i in range(n_aug):
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            out_img = transformed['image']
            out_boxes = transformed['bboxes']
            out_labels = transformed['class_labels']
            out_img_name = f"{filename[:-4]}_aug{i+1}.jpg"
            out_label_name = f"{filename[:-4]}_aug{i+1}.txt"

            cv2.imwrite(os.path.join(save_images_dir, out_img_name), out_img)
            with open(os.path.join(save_labels_dir, out_label_name), 'w') as f:
                for l, b in zip(out_labels, out_boxes):
                    f.write(f"{l} {' '.join(map(str, b))}\n")

# Usage example:
augment_and_save(
    images_dir="C:/Users/junha/Downloads/unpaired shape.v12-general-shape-aug.yolov8/train/images/",
    labels_dir="C:/Users/junha/Downloads/unpaired shape.v12-general-shape-aug.yolov8/train/labels/",
    save_images_dir="C:/Users/junha/Downloads/unpaired shape.v12-general-shape-aug.yolov8/train/images/aug_images/",
    save_labels_dir="C:/Users/junha/Downloads/unpaired shape.v12-general-shape-aug.yolov8/train/labels/aug_labels/",
    n_aug=3  # number of augmentations per original
)