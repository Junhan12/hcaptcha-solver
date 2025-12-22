

import os
import cv2
import glob
import shutil

import albumentations as A

# Define augmentation pipeline
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),  # random -15° to +15°
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # ~-20% to +20% brightness
            contrast_limit=0.2,    # ~-20% to +20% contrast
            p=0.7,
        ),
        A.RandomGamma(
            gamma_limit=(90, 110),  # mild exposure-like change
            p=0.7,
        ),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

def augment_and_save(images_dir, labels_dir, save_images_dir, save_labels_dir, n_aug=3):
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
                print(f"Warning: bad label line '{line}' in {label_path}, skipping.")
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            bboxes.append([xc, yc, bw, bh])
            class_labels.append(int(cls))

        # Save original image and label
        shutil.copy(img_path, os.path.join(save_images_dir, filename))
        shutil.copy(label_path, os.path.join(save_labels_dir, filename.replace(".jpg", ".txt")))

        # Generate augmented versions
        for i in range(n_aug):
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            out_img = transformed["image"]
            out_boxes = transformed["bboxes"]
            out_labels = transformed["class_labels"]

            # Skip if all boxes disappeared (e.g., cropped out)
            if len(out_boxes) == 0:
                continue

            out_img_name = f"{filename[:-4]}_aug{i+1}.jpg"
            out_label_name = f"{filename[:-4]}_aug{i+1}.txt"

            cv2.imwrite(os.path.join(save_images_dir, out_img_name), out_img)

            with open(os.path.join(save_labels_dir, out_label_name), "w") as f:
                for l, b in zip(out_labels, out_boxes):
                    cls_id = l
                    xc, yc, bw, bh = b
                    f.write(f"{cls_id} {xc} {yc} {bw} {bh}\n")

    print("Augmentation completed.")


if __name__ == "__main__":
    # CHANGE these paths to your dataset locations
    augment_and_save(
        images_dir="C:/Users/junha/OneDrive/Desktop/FYP/final-shape-dataset/preprocessed_all_images",
        labels_dir="C:/Users/junha/OneDrive/Desktop/FYP/final-shape-dataset/ori_all_labels",
        save_images_dir="C:/Users/junha/OneDrive/Desktop/FYP/final-shape-dataset/aug_images/",
        save_labels_dir="C:/Users/junha/OneDrive/Desktop/FYP/final-shape-dataset/aug_labels/",
        n_aug=3,                  # augmentations per image
    )
