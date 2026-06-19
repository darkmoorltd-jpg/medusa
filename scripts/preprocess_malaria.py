#!/usr/bin/env python
"""Preprocess NIH Malaria cell images for MEDUSA (handles nested folders)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = 'data/raw/microscopy/malaria/cell_images'
OUT_DIR = 'data/processed/2d_1024px/microscopy_malaria'
ANNO_DIR = 'data/annotations'
IMG_SIZE = 224

def main():
    records = []
    raw = Path(RAW_DIR)

    # Recursively find all images; determine label from parent folder name
    for img_path in raw.rglob('*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        # The parent folder name tells us the class
        parent_name = img_path.parent.name.lower()
        if 'parasitized' in parent_name:
            label = 1
        elif 'uninfected' in parent_name:
            label = 0
        else:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Save to output directory with unique name
        out_name = f"{img_path.parent.name}_{img_path.name}"
        out_path = os.path.join(OUT_DIR, out_name)
        os.makedirs(OUT_DIR, exist_ok=True)
        cv2.imwrite(out_path, img)

        records.append({'image_path': out_name, 'label': label})

    df = pd.DataFrame(records)
    print(f"Total images: {len(df)}")
    print(f"Parasitized: {sum(df['label']==1)}  Uninfected: {sum(df['label']==0)}")

    # Stratified 80/20 split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    os.makedirs(ANNO_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(ANNO_DIR, 'malaria_train.csv'), index=False)
    val_df.to_csv(os.path.join(ANNO_DIR, 'malaria_val.csv'), index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print("Malaria preprocessing complete.")

if __name__ == '__main__':
    main()