#!/usr/bin/env python
"""
Download and preprocess CheXpert-small for MEDUSA.
Downloads via Kaggle API, or skips if zip already present.
Usage:
    python scripts/download_chexpert.py
    python scripts/download_chexpert.py --skip_download --chexpert_zip path/to/CheXpert-v1.0-small.zip
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import subprocess

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CHEXPERT_SMALL_FILENAME = "CheXpert-v1.0-small.zip"
RAW_XRAY_DIR = "data/raw/xray/chexpert"
PROCESSED_2D_DIR = "data/processed/2d_1024px"
ANNOTATIONS_DIR = "data/annotations"
IMG_SIZE = 224

# Frontal projection codes (AP/PA)
FRONTAL_CODES = ["AP", "PA"]

# CheXpert observations → unified label names
OBSERVATIONS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def download_chexpert(output_dir):
    """
    Download CheXpert-small using Kaggle API.
    Falls back to manual instruction if Kaggle is not configured.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, CHEXPERT_SMALL_FILENAME)

    if os.path.exists(zip_path):
        print(f"Zip already exists at {zip_path}, skipping download.")
        return zip_path

    print("Attempting download via Kaggle API...")
    try:
        # Try to download from Kaggle dataset 'ashery/chexpert'
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "ashery/chexpert",
                "-p", output_dir,
                "-f", CHEXPERT_SMALL_FILENAME
            ],
            check=True
        )
        # The file will be saved as CHEXPERT_SMALL_FILENAME in output_dir
        if os.path.exists(zip_path):
            print("Kaggle download successful.")
            return zip_path
        else:
            raise FileNotFoundError(f"Download completed but {zip_path} not found.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Kaggle download failed or not configured.")
        print("Please manually download CheXpert-small from:")
        print("  https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x")
        print("or")
        print("  https://www.kaggle.com/datasets/ashery/chexpert")
        print(f"Place the file '{CHEXPERT_SMALL_FILENAME}' (11 GB) into:")
        print(f"  {os.path.abspath(output_dir)}")
        print("Then run this script again with --skip_download.")
        sys.exit(1)

def extract_zip(zip_path, extract_to):
    """Extract zip file, preserving folder structure."""
    print(f"Extracting {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction finished.")

def preprocess_chexpert(chexpert_root, processed_root, annotations_root):
    """
    Convert frontal CXRs to PNG and create labels CSV.
    chexpert_root: path to the extracted CheXpert folder.
    """
    # Locate the actual dataset folder (may be nested)
    chexpert_dataset = Path(chexpert_root)
    if (chexpert_dataset / "CheXpert-v1.0-small").exists():
        chexpert_dataset = chexpert_dataset / "CheXpert-v1.0-small"

    # Check for train/valid folders
    train_folder = chexpert_dataset / "train"
    valid_folder = chexpert_dataset / "valid"
    train_csv = chexpert_dataset / "train.csv"
    valid_csv = chexpert_dataset / "valid.csv"
    if not train_folder.exists() or not train_csv.exists():
        print("Error: Could not find CheXpert train folder or CSV.")
        print(f"Contents of {chexpert_dataset}: {list(chexpert_dataset.iterdir())}")
        sys.exit(1)

    # Read CSVs
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)

    # Filter frontal images only
    df_train = df_train[df_train["Frontal/Lateral"].isin(FRONTAL_CODES)]
    df_valid = df_valid[df_valid["Frontal/Lateral"].isin(FRONTAL_CODES)]

    # Create output directories and process images
    for split, folder, df in [("train", train_folder, df_train), ("valid", valid_folder, df_valid)]:
        out_dir = os.path.join(processed_root, f"xray_{split}")
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for idx, row in df.iterrows():
            img_rel_path = row["Path"]
            img_path = os.path.join(chexpert_dataset, img_rel_path)
            if not os.path.exists(img_path):
                print(f"Missing: {img_path}")
                continue

            # Read, resize, normalize, save
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = (img - img.min()) / max(img.max() - img.min(), 1e-6)  # [0,1]
            img_uint8 = (img * 255).astype(np.uint8)

            # Safe filename
            new_name = row["Path"].replace("/", "_").replace("\\", "_") + ".png"
            out_path = os.path.join(out_dir, new_name)
            cv2.imwrite(out_path, img_uint8)

            # Collect labels (map -1 uncertain to 0)
            label_dict = {"image_path": new_name}
            for obs in OBSERVATIONS:
                val = row[obs] if obs in row else -1
                label_dict[obs] = 0 if val == -1 else int(val)
            rows.append(label_dict)

        # Save labels CSV for this split
        if rows:
            label_df = pd.DataFrame(rows)
            label_csv = os.path.join(annotations_root, f"chexpert_{split}_labels.csv")
            label_df.to_csv(label_csv, index=False)
            print(f"Saved {len(rows)} labels to {label_csv}")
        else:
            print(f"No images processed for {split}!")

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess CheXpert-small for MEDUSA")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, use existing zip.")
    parser.add_argument("--chexpert_zip", type=str, default=None,
                        help="Path to existing CheXpert zip (if not in default location).")
    parser.add_argument("--output_dir", type=str, default=RAW_XRAY_DIR,
                        help="Directory to store raw CheXpert data.")
    args = parser.parse_args()

    # Paths
    raw_dir = args.output_dir
    zip_path = args.chexpert_zip or os.path.join(raw_dir, CHEXPERT_SMALL_FILENAME)

    # Download if not skipping
    if not args.skip_download and not args.chexpert_zip:
        zip_path = download_chexpert(raw_dir)
    elif args.skip_download and not os.path.exists(zip_path):
        print(f"Error: Zip not found at {zip_path}. Provide --chexpert_zip or allow download.")
        sys.exit(1)

    # Extract
    extract_zip(zip_path, raw_dir)

    # Preprocess
    preprocess_chexpert(
        chexpert_root=raw_dir,
        processed_root=PROCESSED_2D_DIR,
        annotations_root=ANNOTATIONS_DIR
    )

    print("\nDone. MEDUSA data ready for training.")
    print("  python -m src.pretrain.mae_pretrain --data_2d data/processed/2d_1024px")

if __name__ == "__main__":
    main()