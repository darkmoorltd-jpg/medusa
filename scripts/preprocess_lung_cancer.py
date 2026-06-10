#!/usr/bin/env python
"""Preprocess IQ‑OTH/NCCD lung cancer CT dataset for MEDUSA.
Expected structure inside data/raw/ct/:
    archive/The IQ-OTHNCCD lung cancer dataset/
        Bengin cases/
        Malignant cases/
        Normal cases/
(or any nesting depth – the script searches recursively.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = 'data/raw/ct'
OUT_DIR = 'data/processed/2d_1024px'
ANNO_DIR = 'data/annotations'
IMG_SIZE = 224
TEST_SIZE = 0.2          # 20% for validation
RANDOM_SEED = 42

# Class mapping – folder names must match exactly
CLASS_MAP = {
    'Bengin cases': (0, 'benign'),
    'Malignant cases': (1, 'malignant'),
    'Normal cases': (2, 'normal')
}

def find_class_folders():
    """Recursively find the three class folders under RAW_DIR."""
    raw = Path(RAW_DIR)
    for root, dirs, files in os.walk(raw):
        # If the current directory contains the three expected subdirs
        dir_names = set(dirs)
        if all(c in dir_names for c in CLASS_MAP.keys()):
            return Path(root)
    raise FileNotFoundError("Could not find all three class folders anywhere under data/raw/ct")

def main():
    class_root = find_class_folders()
    print(f"Found class folders at: {class_root}")

    all_rows = []
    for class_folder, (label_id, label_name) in CLASS_MAP.items():
        class_dir = class_root / class_folder
        for img_file in class_dir.glob('*.jpg'):
            all_rows.append({'path': str(img_file), 'label': label_id, 'class_name': label_name})

    df = pd.DataFrame(all_rows)
    print(f"Total images found: {len(df)}")

    # Stratified split into train/valid
    train_df, valid_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_SEED
    )

    for split, df_split, out_subfolder in [
        ('train', train_df, 'lung_train'),
        ('valid', valid_df, 'lung_valid')
    ]:
        out_dir = os.path.join(OUT_DIR, out_subfolder)
        os.makedirs(out_dir, exist_ok=True)

        records = []
        for _, row in df_split.iterrows():
            img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255
            img = img.astype(np.uint8)

            # Sanitise filename
            safe_class = row['class_name'].replace(' ', '_')
            fname = f"{safe_class}_{Path(row['path']).stem}.png"
            cv2.imwrite(os.path.join(out_dir, fname), img)
            records.append({'image_path': fname, 'label': row['label']})

        label_csv = os.path.join(ANNO_DIR, f'lung_cancer_{split}_labels.csv')
        pd.DataFrame(records).to_csv(label_csv, index=False)
        print(f"{split}: {len(records)} images -> {label_csv}")

    print("Lung cancer preprocessing complete.")
    print(f"Train samples: {len(train_df)}  Valid samples: {len(valid_df)}")

if __name__ == '__main__':
    main()