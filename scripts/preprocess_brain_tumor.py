#!/usr/bin/env python
"""Preprocess Brain Tumor MRI dataset (Kaggle) for MEDUSA.
Usage:
  python scripts/preprocess_brain_tumor.py --zip_path "data/raw/mri/brain_tumor/brain-tumor-mri-dataset.zip"
"""

import os, sys, zipfile, argparse, shutil
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

CLASS_MAP = {
    "glioma_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor": 2,
    "no_tumor": 3,
    "glioma": 0,        # some folder variants
    "meningioma": 1,
    "pituitary": 2,
    "no": 3,
    "no tumor": 3,
    "no_tumor": 3,
}

IMG_SIZE = 224

def extract_and_preprocess(zip_path, output_root_2d, annotations_dir):
    extract_to = os.path.join(os.path.dirname(zip_path), "extracted")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    # Locate Training/Testing folders (sometimes nested inside another folder)
    data_root = None
    for root, dirs, files in os.walk(extract_to):
        if "Training" in dirs and "Testing" in dirs:
            data_root = root
            break
        # Sometimes the folder names are 'training' and 'testing' (lowercase)
        if "training" in dirs and "testing" in dirs:
            data_root = root
            break
    if data_root is None:
        # try to find a folder that contains subfolders with class names
        for root, dirs, files in os.walk(extract_to):
            for d in dirs:
                if d.lower() in [c.lower() for c in CLASS_MAP.keys()]:
                    data_root = root
                    break
            if data_root:
                break

    if data_root is None:
        print("Could not locate Training/Testing folders inside the zip.")
        sys.exit(1)

    # Process Training and Testing splits
    for split in ["Training", "Testing"]:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            split_path = os.path.join(data_root, split.lower())
        if not os.path.exists(split_path):
            print(f"Warning: {split} folder not found.")
            continue

        # Create output folder
        out_folder = os.path.join(output_root_2d, f"mri_train" if split == "Training" else "mri_valid")
        os.makedirs(out_folder, exist_ok=True)

        rows = []
        for class_name, class_id in CLASS_MAP.items():
            class_dir = os.path.join(split_path, class_name)
            if not os.path.exists(class_dir):
                # try without '_tumor' suffix, or with space
                alt_names = [class_name.replace("_tumor", ""), class_name.replace("_", " ")]
                for alt in alt_names:
                    alt_dir = os.path.join(split_path, alt)
                    if os.path.exists(alt_dir):
                        class_dir = alt_dir
                        break
            if not os.path.exists(class_dir):
                continue

            for img_file in os.listdir(class_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # normalize to [0,255]
                img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255
                img = img.astype(np.uint8)

                out_name = f"{class_name}_{split.lower()}_{img_file}"
                cv2.imwrite(os.path.join(out_folder, out_name), img)
                rows.append({"image_path": out_name, "label": class_id})

        if rows:
            label_df = pd.DataFrame(rows)
            csv_name = f"brain_tumor_{split.lower()}_labels.csv"
            label_csv = os.path.join(annotations_dir, csv_name)
            label_df.to_csv(label_csv, index=False)
            print(f"Saved {len(rows)} images for {split} -> {label_csv}")

    print("Brain tumor preprocessing complete.")
    print("Train folder: data/processed/2d_1024px/mri_train")
    print("Valid folder: data/processed/2d_1024px/mri_valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True, help="Path to brain-tumor-mri-dataset.zip")
    args = parser.parse_args()

    extract_and_preprocess(
        zip_path=args.zip_path,
        output_root_2d="data/processed/2d_1024px",
        annotations_dir="data/annotations"
    )