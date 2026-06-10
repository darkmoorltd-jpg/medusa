#!/usr/bin/env python
"""Evaluate MEDUSA Tiny pneumonia classifier on the whole validation set."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import glob
import numpy as np
from scripts.finetune_tiny_pneumonia import TinyClassifier
from sklearn.metrics import accuracy_score, classification_report

VALID_DIR = 'data/processed/2d_1024px/xray_valid'
LABEL_CSV = 'data/annotations/pneumonia_mnist_test_labels.csv'
MODEL_PATH = 'medusa_tiny_pneumonia.pt'

def load_labels(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df['image_path'], df['label']))

if __name__ == '__main__':
    label_map = load_labels(LABEL_CSV)
    model = TinyClassifier(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=False))
    model.eval()

    y_true, y_pred = [], []
    files = list(glob.glob(os.path.join(VALID_DIR, '*.png')))
    for f in files:
        fname = os.path.basename(f)
        if fname not in label_map:
            continue
        y_true.append(label_map[fname])
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img_t = torch.from_numpy(cv2.resize(img, (224,224))).unsqueeze(0).unsqueeze(0).float()/255
        with torch.no_grad():
            logits = model(img_t)
            pred = logits.argmax(dim=1).item()
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"Total images: {len(y_true)}")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))