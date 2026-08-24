import os
import shutil
import random
from pathlib import Path

random.seed(42)  # reproducibility

# CONFIG
SOURCE_DIR = "Medicinal Plant Leaf Health Original Dataset"   # contains 16 class subfolders
OUTPUT_DIR = "Medicinal Plant Leaf Health Split Dataset"      # will contain train/val/test

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def split_dataset(source_dir, output_dir, splits):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    class_folders = sorted([f for f in source_dir.iterdir() if f.is_dir()])
    print(f"Found {len(class_folders)} classes\n")

    for split_name in splits:
        for cls in class_folders:
            (output_dir / split_name / cls.name).mkdir(parents=True, exist_ok=True)

    grand_total = 0
    summary = []

    for cls in class_folders:
        # case-insensitive extension match, catches every image regardless of naming
        images = [f for f in cls.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS]
        random.shuffle(images)

        n = len(images)
        if n == 0:
            print(f"WARNING: {cls.name} has 0 images — skipping")
            continue

        n_train = int(round(n * splits["train"]))
        n_val = int(round(n * splits["val"]))
        # test gets the remainder, so all images are accounted for even with rounding
        n_test = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            shutil.copy2(img, output_dir / "train" / cls.name / img.name)
        for img in val_imgs:
            shutil.copy2(img, output_dir / "val" / cls.name / img.name)
        for img in test_imgs:
            shutil.copy2(img, output_dir / "test" / cls.name / img.name)

        grand_total += n
        summary.append((cls.name, n, len(train_imgs), len(val_imgs), len(test_imgs)))
        print(f"{cls.name}: total={n} train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    print(f"\nGrand total images processed: {grand_total}")
    print(f"Expected (from your table): 1323")
    if grand_total != 1323:
        print("MISMATCH — check for extra files, hidden files, or missed images in subfolders")

    return summary

split_dataset(SOURCE_DIR, OUTPUT_DIR, SPLITS)
