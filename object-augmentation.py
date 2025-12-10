import os
import glob
import shutil

import cv2
import albumentations as A
import numpy as np


# ============ TRANSFORMS ============

# 1) Rotation only: -15 to +15 degrees
rotate_transform = A.Compose(
    [
        A.Rotate(limit=(-15, 15), p=1.0),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        clip=True,          # try to clip internally
    ),
)

# 2) Brightness only: about ±15% change
brightness_transform = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.15,  # ~ -15% to +15% brightness
            contrast_limit=0.0,     # no contrast change here
            p=1.0,
        )
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        clip=True,
    ),
)

# 3) Contrast only: about ±10% change
contrast_transform = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=0.0,   # no brightness change here
            contrast_limit=0.10,    # ~ -10% to +10% contrast
            p=1.0,
        )
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        clip=True,
    ),
)


# ============ HELPERS ============

def clip_yolo_box(box, eps=1e-6):
    """
    Clip a YOLO bbox [xc, yc, w, h] to [0,1] range with a tiny epsilon
    to avoid strict Albumentations checks failing on -1e-7, 1+1e-7, etc.
    """
    xc, yc, w, h = box
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    # Push extremely small negatives to 0 and extremely small >1 to 1
    if xc < 0.0:
        xc = 0.0
    if yc < 0.0:
        yc = 0.0
    if w < 0.0:
        w = 0.0
    if h < 0.0:
        h = 0.0

    if xc > 1.0:
        xc = 1.0
    if yc > 1.0:
        yc = 1.0
    if w > 1.0:
        w = 1.0
    if h > 1.0:
        h = 1.0

    # Optional: ensure width/height are not exactly zero if box exists
    return [xc, yc, w, h]


def save_augmented(image, bboxes, class_labels,
                   transform, base_name, aug_suffix,
                   save_images_dir, save_labels_dir):
    """
    Apply a single transform and save image + labels if any bbox remains.
    Also clips YOLO boxes to [0,1] to avoid float precision errors.
    """
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    out_img = transformed["image"]
    out_boxes = transformed["bboxes"]
    out_labels = transformed["class_labels"]

    if len(out_boxes) == 0:
        # All boxes disappeared or were filtered
        return

    # Clip boxes to [0,1]
    clipped_boxes = [clip_yolo_box(b) for b in out_boxes]
    out_boxes = clipped_boxes

    out_img_name = f"{base_name}_{aug_suffix}.jpg"
    out_label_name = f"{base_name}_{aug_suffix}.txt"

    cv2.imwrite(os.path.join(save_images_dir, out_img_name), out_img)
    with open(os.path.join(save_labels_dir, out_label_name), "w") as f:
        for l, b in zip(out_labels, out_boxes):
            cls_id = l
            xc, yc, bw, bh = b
            f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")


# ============ MAIN AUGMENTATION ============

def augment_and_save(images_dir, labels_dir, save_images_dir, save_labels_dir, n_aug=3):
    """
    For each image:
      1. Save original
      2. If n_aug >= 1: rotation-only augmentation
      3. If n_aug >= 2: brightness-only augmentation
      4. If n_aug >= 3: contrast-only augmentation
    """
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_labels_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Found {len(image_files)} images in {images_dir}")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt"))

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: cannot read image {img_path}, skipping.")
            continue

        if not os.path.exists(label_path):
            print(f"Warning: label file not found for {filename}, skipping.")
            continue

        # Read labels in YOLO format: cls xc yc w h (all normalized)
        with open(label_path, "r") as f:
            lines = f.read().splitlines()

        if len(lines) == 0:
            print(f"Warning: no boxes in {label_path}, skipping.")
            continue

        bboxes, class_labels = [], []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: bad label line '{line}' in {label_path}, skipping line.")
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            # Clip any slightly out-of-range original boxes as well
            xc, yc, bw, bh = clip_yolo_box([xc, yc, bw, bh])
            bboxes.append([xc, yc, bw, bh])
            class_labels.append(int(cls))

        # Save original image and label
        shutil.copy(img_path, os.path.join(save_images_dir, filename))
        shutil.copy(label_path, os.path.join(save_labels_dir, filename.replace(".jpg", ".txt")))

        base_name = filename[:-4]  # remove .jpg

        # Aug 1: rotation
        if n_aug >= 1:
            save_augmented(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
                transform=rotate_transform,
                base_name=base_name,
                aug_suffix="aug1_rot",
                save_images_dir=save_images_dir,
                save_labels_dir=save_labels_dir,
            )

        # Aug 2: brightness
        if n_aug >= 2:
            save_augmented(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
                transform=brightness_transform,
                base_name=base_name,
                aug_suffix="aug2_bright",
                save_images_dir=save_images_dir,
                save_labels_dir=save_labels_dir,
            )

        # Aug 3: contrast
        if n_aug >= 3:
            save_augmented(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
                transform=contrast_transform,
                base_name=base_name,
                aug_suffix="aug3_contrast",
                save_images_dir=save_images_dir,
                save_labels_dir=save_labels_dir,
            )

    print("Augmentation completed.")


# ============ ENTRY POINT ============

if __name__ == "__main__":
    # CHANGE these paths to your dataset locations
    augment_and_save(
        images_dir="C:/Users/junha/Downloads/cleaned_dataset/cleaned_dataset/images",
        labels_dir="C:/Users/junha/Downloads/cleaned_dataset/cleaned_dataset/labels",
        save_images_dir="C:/Users/junha/Downloads/cleaned_dataset/cleaned_dataset/aug_images",
        save_labels_dir="C:/Users/junha/Downloads/cleaned_dataset/cleaned_dataset/aug_labels",
        n_aug=3,  # 1: only rotation, 2: +brightness, 3: +contrast
    )
