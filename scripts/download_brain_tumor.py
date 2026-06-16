#!/usr/bin/env python
"""
Download Brain Tumor MRI dataset from Kaggle and preprocess for MEDUSA.
Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
"""

import os
import sys
import zipfile
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
RAW_MRI_DIR = "data/raw/mri/brain_tumor"
PROCESSED_2D_DIR = "data/processed/2d_1024px"
ANNOTATIONS_DIR = "data/annotations"
IMG_SIZE = 224

# Class mapping from folder names
CLASS_MAP = {
    "glioma_tumor": 0,
    "meningioma_tumor": 1,
    "pituitary_tumor": 2,
    "no_tumor": 3
}

def download_dataset(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "brain-tumor-mri.zip")
    if not os.path.exists(zip_path):
        print("Downloading Brain Tumor MRI dataset via Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", output_dir],
            check=True
        )
        # The downloaded file is 'brain-tumor-mri-dataset.zip'
        downloaded = os.path.join(output_dir, "brain-tumor-mri-dataset.zip")
        if os.path.exists(downloaded):
            os.rename(downloaded, zip_path)
    else:
        print("Zip already exists.")
    return zip_path

def extract_and_preprocess(zip_path, output_root):
    # Extract
    extract_to = os.path.join(output_root, "extracted")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    # Find the actual data folder (likely 'Brain Tumor Data Set' or 'brain_tumor_dataset')
    # The Kaggle dataset structure: inside zip, there are two folders: 'Training' and 'Testing'
    # Each has subfolders by class.
    data_root = None
    for root, dirs, files in os.walk(extract_to):
        if "Training" in dirs and "Testing" in dirs:
            data_root = root
            break
    if data_root is None:
        # fallback: maybe it's just one folder named 'brain_tumor_dataset'
        for root, dirs, files in os.walk(extract_to):
            if any(c in root for c in CLASS_MAP.keys()):
                data_root = root
                break
    if data_root is None:
        print("Could not locate dataset folders. Expected 'Training' and 'Testing'.")
        sys.exit(1)

    # Process Training and Testing splits
    output_train_dir = os.path.join(output_root, "mri_train")
    output_valid_dir = os.path.join(output_root, "mri_valid")
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_valid_dir, exist_ok=True)

    for split, out_split_dir in [("Training", output_train_dir), ("Testing", output_valid_dir)]:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            # Some versions have lowercase 'training'
            split_path = os.path.join(data_root, split.lower())
        if not os.path.exists(split_path):
            print(f"Could not find {split} folder; skipping.")
            continue

        rows = []
        for class_name, class_id in CLASS_MAP.items():
            class_dir = os.path.join(split_path, class_name)
            if not os.path.exists(class_dir):
                # Try without '_tumor' suffix
                alt_name = class_name.replace("_tumor", "")
                class_dir = os.path.join(split_path, alt_name)
            if not os.path.exists(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    # Normalize and save as PNG
                    img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255
                    img = img.astype(np.uint8)
                    out_name = f"{class_name}_{split.lower()}_{img_file}"
                    out_path = os.path.join(out_split_dir, out_name)
                    cv2.imwrite(out_path, img)
                    rows.append({"image_path": out_name, "label": class_id})

        if rows:
            label_df = pd.DataFrame(rows)
            label_csv = os.path.join(ANNOTATIONS_DIR, f"brain_tumor_{split.lower()}_labels.csv")
            label_df.to_csv(label_csv, index=False)
            print(f"Saved {len(rows)} images for {split} -> {label_csv}")

def main():
    zip_path = download_dataset(RAW_MRI_DIR)
    extract_and_preprocess(zip_path, PROCESSED_2D_DIR)
    print("Brain Tumor MRI preprocessing complete.")

if __name__ == "__main__":
    main()