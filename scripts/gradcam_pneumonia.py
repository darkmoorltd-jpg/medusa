#!/usr/bin/env python
"""Grad‑CAM visualisation for MEDUSA Tiny Pneumonia classifier."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scripts.finetune_tiny_pneumonia import TinyClassifier, TinyViT

def gradcam(model, img_tensor, target_layer):
    """Simple Grad‑CAM: gradients of the target class w.r.t. the feature map."""
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    logits = model(img_tensor)
    class_idx = logits.argmax(dim=1).item()
    logits[0, class_idx].backward()

    hook_f.remove()
    hook_b.remove()

    # Weighted combination
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])  # (C,)
    activations = features[0].squeeze(0)                        # (C, H, W)
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=0).detach().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap, class_idx

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to chest X‑ray')
    parser.add_argument('--model', default='medusa_tiny_pneumonia.pt')
    args = parser.parse_args()

    # Load model
    model = TinyClassifier(num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=False))
    model.eval()

    # Preprocess image
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (224, 224))
    img_t = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255

    # Pick a conv layer early in the encoder
    target_layer = model.encoder.patch_embed

    heatmap, pred_class = gradcam(model, img_t, target_layer)

    # Upsample heatmap to original size
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))

    # Overlay
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original X‑ray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    label = 'Pneumonia' if pred_class == 1 else 'Normal'
    plt.title(f'MEDUSA Focus – {label}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_output.png')
    print(f"Prediction: {label}  |  Heatmap saved as gradcam_output.png")