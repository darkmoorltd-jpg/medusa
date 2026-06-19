#!/usr/bin/env python
"""Preprocess tuberculosis chest X‑ray dataset for MEDUSA."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = 'data/raw/xray/tuberculosis'
OUT_DIR = 'data/processed/2d_1024px/xray_tb'
ANNO_DIR = 'data/annotations'
IMG_SIZE = 224

def main():
    records = []
    raw = Path(RAW_DIR)

    # Recursively find all images
    for img_path in raw.rglob('*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        # Determine label from parent folder name
        folder_name = img_path.parent.name.lower()
        if 'normal' in folder_name:
            label = 0
        elif 'tb' in folder_name or 'tuber' in folder_name:
            label = 1
        else:
            continue   # skip unknown folders

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Normalize to [0,255] for PNG saving
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255
        img = img.astype(np.uint8)

        # Unique filename
        fname = f"{folder_name}_{img_path.stem}.png"
        out_path = os.path.join(OUT_DIR, fname)
        os.makedirs(OUT_DIR, exist_ok=True)
        cv2.imwrite(out_path, img)

        records.append({'image_path': fname, 'label': label})

    df = pd.DataFrame(records)
    print(f"Total images found: {len(df)}")
    print(f"Normal: {sum(df['label']==0)}  TB: {sum(df['label']==1)}")

    # Stratified split 80/20
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    os.makedirs(ANNO_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(ANNO_DIR, 'tb_train.csv'), index=False)
    val_df.to_csv(os.path.join(ANNO_DIR, 'tb_val.csv'), index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print("TB preprocessing complete.")

if __name__ == '__main__':
    main()