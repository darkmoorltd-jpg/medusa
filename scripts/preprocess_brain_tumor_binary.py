#!/usr/bin/env python
"""Preprocess binary brain-tumour dataset (Negative/Positive) with patient‑level split."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

RAW_DIR = 'data/raw/mri/Brain_Tumor'
OUT_DIR = 'data/processed/2d_1024px/mri_binary'
ANNO_DIR = 'data/annotations'
IMG_SIZE = 224

def extract_patient_id(fname):
    """Extract patient ID from filename (e.g., Te-noTr_0000.jpg → Te-noTr)."""
    return fname.split('_')[0]

def main():
    records = []
    raw = Path(RAW_DIR)
    for class_name, label in [('Negative', 0), ('Positive', 1)]:
        class_dir = raw / class_name
        if not class_dir.exists():
            continue
        for img_file in class_dir.glob('*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = (img - img.min()) / max(img.max() - img.min(), 1e-6) * 255
            img = img.astype(np.uint8)

            fname = f"{class_name}_{img_file.name}"
            out_path = os.path.join(OUT_DIR, fname)
            os.makedirs(OUT_DIR, exist_ok=True)
            cv2.imwrite(out_path, img)

            patient_id = extract_patient_id(img_file.name)
            records.append({'image_path': fname, 'label': label, 'patient_id': patient_id})

    df = pd.DataFrame(records)
    print(f"Total images: {len(df)}")
    print(f"Negative: {sum(df['label']==0)}  Positive: {sum(df['label']==1)}")

    # Patient‑level split 80/20
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df['patient_id']))
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]

    os.makedirs(ANNO_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(ANNO_DIR, 'brain_tumor_binary_train.csv'), index=False)
    val_df.to_csv(os.path.join(ANNO_DIR, 'brain_tumor_binary_val.csv'), index=False)

    print(f"Train patients: {train_df['patient_id'].nunique()} | Val patients: {val_df['patient_id'].nunique()}")
    print(f"Train images: {len(train_df)} | Val images: {len(val_df)}")
    print("Binary brain‑tumour preprocessing complete.")

if __name__ == '__main__':
    main()