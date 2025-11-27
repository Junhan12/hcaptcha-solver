from __future__ import annotations

import argparse
import random
import sys
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image, ImageEnhance

# =========================
# USER CONFIG (YOLO DATASET)
# =========================

DATASET_ROOT = Path(r"C:\Users\User\Downloads\FYP.v13i.yolov8")

# Only augment the training split by default
DEFAULT_SPLITS: Sequence[str] = ("train",)

DEFAULT_NUM_OUTPUTS = 3


def _random_small_rotation(image: Image.Image, rng: random.Random) -> Image.Image:
    """Rotate image by a random angle between -15° and +15°."""
    angle = rng.uniform(-15.0, 15.0)
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)


def _adjust_brightness(image: Image.Image, rng: random.Random) -> Image.Image:
    """Adjust brightness between -15% and +15%."""
    factor = 1.0 + rng.uniform(-0.15, 0.15)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _adjust_exposure(image: Image.Image, rng: random.Random) -> Image.Image:
    """Adjust exposure/contrast between -10% and +10%."""
    factor = 1.0 + rng.uniform(-0.10, 0.10)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def augment_image(
    image: Image.Image,
    num_outputs: int = 3,
    *,
    seed: int | None = None,
) -> List[Image.Image]:
    """
    Generate multiple augmented versions of the provided image.

    Args:
        image: Pillow image to augment.
        num_outputs: Number of augmented samples to return (default: 3).
        seed: Optional random seed for reproducibility.

    Returns:
        List of augmented Pillow images.
    """
    rng = random.Random(seed)
    outputs: List[Image.Image] = []

    for _ in range(num_outputs):
        aug = image.copy()
        aug = _random_small_rotation(aug, rng)
        aug = _adjust_brightness(aug, rng)
        aug = _adjust_exposure(aug, rng)
        outputs.append(aug)

    return outputs


def _iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield image file paths from the provided iterable."""
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    for path in paths:
        if path.is_dir():
            for nested in sorted(path.rglob("*")):
                if nested.suffix.lower() in supported:
                    yield nested
        elif path.suffix.lower() in supported:
            yield path


# ========== YOLO DATASET MODE (images + labels in train folder) ==========

def augment_yolo_split(
    dataset_root: Path,
    split: str = "train",
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    seed: int | None = None,
) -> None:
    """
    Augment a YOLO split (e.g., 'train') by creating new images and
    copying the corresponding label files with new names.

    Reads from:
        <root>/<split>/images
        <root>/<split>/labels

    Writes augmented images and labels back into those SAME folders.
    """
    rng = random.Random(seed)

    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[WARN] Split '{split}' is missing images/labels. Skipping.")
        return

    print(f"[INFO] Augmenting YOLO split '{split}'")
    print(f"       Images dir: {images_dir}")
    print(f"       Labels dir: {labels_dir}")

    aug_count = 0

    for image_path in _iter_images([images_dir]):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            print(f"[WARN] No label for image {image_path.name} ({label_path.name}), skipping.")
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # one random seed per original image to vary augmentations
            base_seed = rng.randint(0, 10_000_000)

            for idx in range(1, num_outputs + 1):
                sample_seed = base_seed + idx
                aug_list = augment_image(img, num_outputs=1, seed=sample_seed)
                aug = aug_list[0]

                new_stem = f"{stem}_aug{idx}"
                new_img_path = images_dir / f"{new_stem}{image_path.suffix}"
                new_label_path = labels_dir / f"{new_stem}.txt"

                # Save augmented image
                aug.save(new_img_path)

                # Copy original label file (we keep same bboxes; small rotation is tolerated)
                shutil.copy2(label_path, new_label_path)

                aug_count += 1
                print(f"[OK] Saved {new_img_path.name} and {new_label_path.name}")

    print(f"[INFO] Done augmenting '{split}'. Generated {aug_count} augmented images + labels.")


def augment_yolo_dataset(
    dataset_root: Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
    num_outputs: int = DEFAULT_NUM_OUTPUTS,
    seed: int | None = None,
) -> None:
    """Augment all specified YOLO splits (usually just 'train')."""
    for split in splits:
        augment_yolo_split(dataset_root, split=split, num_outputs=num_outputs, seed=seed)


# ========== ORIGINAL CLI MODE (images only, NO labels) ==========

def process_images(
    inputs: Sequence[str],
    output_dir: str,
    num_outputs: int = 3,
    seed: int | None = None,
) -> None:
    """Augment every image resolved from `inputs` and save results into `output_dir` (no labels)."""
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in _iter_images([Path(p) for p in inputs]):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            sample_seed = rng.randint(0, 10_000_000)
            augmented = augment_image(img, num_outputs=num_outputs, seed=sample_seed)

            for idx, aug in enumerate(augmented, start=1):
                save_name = f"{image_path.stem}_aug{idx}.png"
                aug.save(output_path / save_name, format="PNG")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate data augmentations matching the specified recipe."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files or directories containing images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Directory where augmented images will be written.",
    )
    parser.add_argument(
        "--num-outputs",
        "-n",
        type=int,
        default=3,
        help="Number of augmented samples per original image (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base random seed for reproducibility.",
    )
    return parser


def main() -> None:
    # If no CLI args provided (other than script name), use YOLO DATASET MODE
    if len(sys.argv) == 1:
        if not DATASET_ROOT.exists():
            raise FileNotFoundError(
                f"DATASET_ROOT does not exist. Please update DATASET_ROOT in {__file__}"
            )

        augment_yolo_dataset(
            dataset_root=DATASET_ROOT,
            splits=DEFAULT_SPLITS,
            num_outputs=DEFAULT_NUM_OUTPUTS,
            seed=None,
        )

        print("[INFO] YOLO augmentation complete using configured DATASET_ROOT.")
        return

    # CLI mode: image-only augmentation (no labels)
    parser = _build_parser()
    args = parser.parse_args()

    process_images(
        args.inputs,
        args.output,
        num_outputs=args.num_outputs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


