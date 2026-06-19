#!/usr/bin/env python
"""Preprocess C-NMC leukemia dataset for MEDUSA (BMP → grayscale PNG, CSV‑based validation labels)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import cv2
import numpy as np
from pathlib import Path

RAW_ROOT = Path('data/raw/microscopy/leukemia/C-NMC_Leukemia')
OUT_DIR  = 'data/processed/2d_1024px/microscopy_leukemia'
ANNO_DIR = 'data/annotations'
IMG_SIZE = 224

def main():
    records = []

    # ---- TRAINING DATA (folder-based labels) ----
    train_dir = RAW_ROOT / 'training_data'
    if train_dir.exists():
        for fold_dir in train_dir.iterdir():
            if not fold_dir.is_dir():
                continue
            for class_dir in fold_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name.lower()
                if class_name == 'all':
                    label = 1
                elif class_name == 'hem':
                    label = 0
                else:
                    continue
                for img_file in class_dir.glob('*.bmp'):
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is None or img.size == 0:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    out_name = f"train_{class_name}_{img_file.name.replace('.bmp', '.png')}"
                    out_path = os.path.join(OUT_DIR, out_name)
                    os.makedirs(OUT_DIR, exist_ok=True)
                    cv2.imwrite(out_path, img)
                    records.append({'image_path': out_name, 'label': label, 'split': 'train'})

    # ---- VALIDATION DATA (CSV-based labels) ----
    val_dir = RAW_ROOT / 'validation_data'
    csv_files = list(val_dir.glob('*.csv'))
    label_map = {}
    if csv_files:
        df = pd.read_csv(csv_files[0])
        # Columns: new_names (filename), labels (0/1)
        for _, row in df.iterrows():
            fname = str(row['new_names']).strip()
            label = int(row['labels'])
            label_map[fname] = label

    if val_dir.exists():
        for img_file in val_dir.glob('*.bmp'):
            fname = img_file.name
            label = label_map.get(fname, -1)
            if label == -1:
                continue   # skip if not in CSV
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            out_name = f"val_{fname.replace('.bmp', '.png')}"
            out_path = os.path.join(OUT_DIR, out_name)
            os.makedirs(OUT_DIR, exist_ok=True)
            cv2.imwrite(out_path, img)
            records.append({'image_path': out_name, 'label': label, 'split': 'val'})

    # ---- TESTING DATA (CSV-based, treated as extra validation) ----
    test_dir = RAW_ROOT / 'testing_data'
    test_csv_files = list(test_dir.glob('*.csv')) if test_dir.exists() else []
    if test_csv_files:
        df_test = pd.read_csv(test_csv_files[0])
        test_label_map = {}
        for _, row in df_test.iterrows():
            test_label_map[str(row['new_names']).strip()] = int(row['labels'])
        if test_dir.exists():
            for img_file in test_dir.glob('*.bmp'):
                fname = img_file.name
                label = test_label_map.get(fname, -1)
                if label == -1:
                    continue
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                out_name = f"test_{fname.replace('.bmp', '.png')}"
                out_path = os.path.join(OUT_DIR, out_name)
                os.makedirs(OUT_DIR, exist_ok=True)
                cv2.imwrite(out_path, img)
                records.append({'image_path': out_name, 'label': label, 'split': 'val'})

    # ---- Save CSVs ----
    df_all = pd.DataFrame(records)
    print(f"Total images: {len(df_all)}")
    print(f"ALL (cancer): {sum(df_all['label']==1)}  HEM (normal): {sum(df_all['label']==0)}")

    train_df = df_all[df_all['split'] == 'train']
    val_df   = df_all[df_all['split'] == 'val']
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    os.makedirs(ANNO_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(ANNO_DIR, 'leukemia_train.csv'), index=False)
    val_df.to_csv(os.path.join(ANNO_DIR, 'leukemia_val.csv'), index=False)
    print("Leukemia preprocessing complete.")

if __name__ == '__main__':
    main()