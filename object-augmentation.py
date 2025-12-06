import cv2
import albumentations as A
import os
import shutil
from pathlib import Path
from tqdm import tqdm
# =========================
# USER CONFIGURATION
# =========================
DATASET_ROOT = Path(r"C:\Users\junha\Downloads\object-dataset\train")
OUTPUT_ROOT = Path(r"C:\Users\junha\Downloads\object-dataset\train")

# =========================
# HELPER FUNCTIONS
# =========================
def read_yolo_label(label_path):
    bboxes = []
    class_ids = []
    if not label_path.exists():
        return [], []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, class_ids

def save_yolo_label(save_path, bboxes, class_ids):
    with open(save_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_ids):
            # Safety Clip: Ensure YOLO format is strictly 0.0-1.0
            x_c, y_c, w, h = [max(0.0, min(1.0, val)) for val in bbox]
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

# =========================
# MAIN LOGIC
# =========================
def process_augmentation():
    # 1. Define Directories
    src_img_dir = DATASET_ROOT / "images"
    src_lbl_dir = DATASET_ROOT / "labels"
    out_img_dir = OUTPUT_ROOT / "aug_images"
    out_lbl_dir = OUTPUT_ROOT / "aug_labels"

    # 2. Create Output Folders
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 3. Define the 3 DISTINCT transformations (p=1.0 means ALWAYS apply)
    # We create a list of single operations.
    aug_operations = [
        # Augmentation 1: Rotation Only
        A.Compose([
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_ids'])),

        # Augmentation 2: Brightness Only
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_ids'])),

        # Augmentation 3: Contrast/Exposure Only
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_ids']))
    ]

    # 4. Get Images
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in src_img_dir.iterdir() if p.suffix.lower() in valid_ext]

    print(f"Found {len(image_files)} images.")
    print("Generating: 1 Original + 3 Augmented versions (Rotation, Brightness, Contrast)")

    count = 0

    for img_path in tqdm(image_files):
        stem = img_path.stem
        
        # Load Data
        image = cv2.imread(str(img_path))
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_path = src_lbl_dir / f"{stem}.txt"
        bboxes, class_ids = read_yolo_label(label_path)

        # --- SAVE 1: ORIGINAL (Copy) ---
        shutil.copy2(img_path, out_img_dir / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, out_lbl_dir / label_path.name)

        # --- SAVE 2, 3, 4: AUGMENTATIONS ---
        # We loop through our specific list of 3 operations
        for i, pipeline in enumerate(aug_operations):
            try:
                # Apply the specific pipeline
                augmented = pipeline(image=image, bboxes=bboxes, class_ids=class_ids)
                
                # Check if boxes still exist (Rotation might remove them)
                if not augmented['bboxes'] and bboxes:
                    # If rotation pushed object out of view, skip saving this specific version
                    continue

                # Save Image
                suffix = f"_aug{i+1}" # _aug1, _aug2, _aug3
                new_stem = f"{stem}{suffix}"
                
                aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_img_dir / f"{new_stem}{img_path.suffix}"), aug_img_bgr)

                # Save Label
                save_yolo_label(out_lbl_dir / f"{new_stem}.txt", augmented['bboxes'], augmented['class_ids'])
                
                count += 1
            except Exception as e:
                print(f"[WARN] Error on {stem} aug {i+1}: {e}")

    print(f"\n[DONE] Saved {count} augmented images to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_augmentation()
