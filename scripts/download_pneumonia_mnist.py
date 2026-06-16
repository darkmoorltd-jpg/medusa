#!/usr/bin/env python
"""
Download PneumoniaMNIST 224×224 from official MedMNIST+ Zenodo and convert to MEDUSA format.
URL: https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1
Usage:
  python scripts/download_pneumonia_mnist.py
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image

ZENODO_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1"
RAW_DIR = "data/raw/xray/pneumonia_mnist"
PROCESSED_2D_DIR = "data/processed/2d_1024px"
ANNOTATIONS_DIR = "data/annotations"

def main():
    # 1. Download the .npz file
    os.makedirs(RAW_DIR, exist_ok=True)
    npz_path = os.path.join(RAW_DIR, "pneumoniamnist_224.npz")

    if not os.path.exists(npz_path):
        print(f"Downloading PneumoniaMNIST 224×224 (20.6 MB) from Zenodo...")
        urllib.request.urlretrieve(ZENODO_URL, npz_path)
        print("Download complete.")
    else:
        print(f"File already exists at {npz_path}, skipping download.")

    # 2. Load and extract
    data = np.load(npz_path)
    # Standard MedMNIST keys: train_images, train_labels, val_images, val_labels, test_images, test_labels
    train_imgs = data["train_images"]
    train_labels = data["train_labels"].flatten()
    val_imgs = data["val_images"]
    val_labels = data["val_labels"].flatten()
    test_imgs = data["test_images"]
    test_labels = data["test_labels"].flatten()

    # 3. Create output directories
    train_dir = os.path.join(PROCESSED_2D_DIR, "xray_train")
    val_dir = os.path.join(PROCESSED_2D_DIR, "xray_valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    # 4. Convert splits to PNGs
    splits = [
        ("train", train_imgs, train_labels, train_dir),
        ("val",   val_imgs,   val_labels,   train_dir),   # merge into train
        ("test",  test_imgs,  test_labels,  val_dir),     # test becomes validation for MEDUSA
    ]

    for split_name, images, labels, out_dir in splits:
        rows = []
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i])  # uint8, 224×224
            fname = f"pneumonia_{split_name}_{i:05d}.png"
            img.save(os.path.join(out_dir, fname))
            rows.append({"image_path": fname, "label": int(labels[i])})

        if rows:
            label_df = pd.DataFrame(rows)
            label_csv = os.path.join(ANNOTATIONS_DIR, f"pneumonia_mnist_{split_name}_labels.csv")
            label_df.to_csv(label_csv, index=False)
            print(f"Saved {len(rows)} images for {split_name} -> {label_csv}")

    print("Done. PneumoniaMNIST 224×224 ready for MEDUSA.")

if __name__ == "__main__":
    main()