#!/usr/bin/env python
"""Quick inference with MEDUSA Tiny Pneumonia classifier."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import argparse
from scripts.finetune_tiny_pneumonia import TinyClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to a chest X-ray image (PNG/JPG)')
    parser.add_argument('--model', default='medusa_tiny_pneumonia.pt', help='Model file')
    args = parser.parse_args()

    # Load model
    model = TinyClassifier()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    # Preprocess image
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0

    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        pred = probs.argmax().item()
        label = 'Pneumonia' if pred == 1 else 'Normal'

    print(f"Prediction: {label}")
    print(f"Confidence: Normal={probs[0]:.3f}, Pneumonia={probs[1]:.3f}")

if __name__ == '__main__':
    main()