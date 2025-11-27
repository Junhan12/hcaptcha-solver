import os
import json
from collections import Counter
from pathlib import Path


DATASET_ROOT = r"C:\Users\User\Downloads\FYP.v13i.yolov8"
label_ext = ".txt"

class_counts = Counter()
image_counts = Counter()

for split in ["train", "valid", "test"]:
    label_dir = Path(DATASET_ROOT) / split / "labels"
    if not label_dir.exists():
        continue

    for label_file in label_dir.glob(f"*{label_ext}"):
        with open(label_file, "r") as f:
            for line in f:
                if line.strip():
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
                    image_counts[split] += 1

summary = {
    "num_samples": dict(image_counts),
    "class_distribution": dict(class_counts),
}

with open("dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Saved to dataset_summary.json")


