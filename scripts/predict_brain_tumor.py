#!/usr/bin/env python
"""Quick inference with MEDUSA Tiny Brain Tumor classifier."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import argparse
from scripts.finetune_tiny_brain_tumor import TinyBrainTumorClassifier

CLASSES = ['glioma', 'meningioma', 'pituitary', 'healthy']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to a brain MRI image (JPG/PNG)')
    parser.add_argument('--model', default='medusa_tiny_brain_tumor.pt', help='Model file')
    args = parser.parse_args()

    model = TinyBrainTumorClassifier(num_classes=4)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=False))
    model.eval()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img_t = torch.from_numpy(cv2.resize(img, (224, 224))).unsqueeze(0).unsqueeze(0).float() / 255

    with torch.no_grad():
        prob = torch.softmax(model(img_t), 1).squeeze()

    print("Tumour type probabilities:")
    for c, p in zip(CLASSES, prob):
        print(f"  {c:12s}: {p:.3f}")
    print(f"\nPrediction: {CLASSES[prob.argmax().item()]}")

if __name__ == '__main__':
    main()