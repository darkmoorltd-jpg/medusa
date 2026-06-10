#!/usr/bin/env python
"""MEDUSA Multi‑Head API – pneumonia, brain tumour, lung cancer from a single endpoint."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image

pneumonia_model = None
brain_model = None
lung_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pneumonia_model, brain_model, lung_model

    from scripts.finetune_tiny_pneumonia import TinyClassifier
    pneumonia_model = TinyClassifier(num_classes=2)
    pneumonia_model.load_state_dict(
        torch.load('medusa_tiny_pneumonia.pt', map_location='cpu', weights_only=False),
        strict=False
    )
    pneumonia_model.eval()

    from scripts.finetune_tiny_brain_tumor_mae import TinyBrainTumorClassifier
    brain_model = TinyBrainTumorClassifier(num_classes=4)
    brain_model.load_state_dict(
        torch.load('medusa_tiny_brain_tumor_v2.pt', map_location='cpu', weights_only=False),
        strict=False
    )
    brain_model.eval()

    if os.path.exists('medusa_tiny_lung_cancer_v2.pt'):
        from scripts.finetune_tiny_lung_cancer import TinyLungCancerClassifier
        lung_model = TinyLungCancerClassifier(num_classes=3)
        lung_model.load_state_dict(
            torch.load('medusa_tiny_lung_cancer_v2.pt', map_location='cpu', weights_only=False),
            strict=False
        )
        lung_model.eval()
        print("✅ Lung cancer model loaded.")
    else:
        print("⚠️ Lung cancer model not found – lung CT endpoint will be unavailable.")

    print("✅ All available models loaded. Ready for inference.")
    yield

app = FastAPI(title="MEDUSA Multi‑Head API", lifespan=lifespan)

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((224, 224))
    img_t = torch.from_numpy(np.array(img)).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img_t

def detect_modality(image_bytes, filename=""):
    name_lower = filename.lower()
    if any(w in name_lower for w in ["xray", "x‑ray", "chest"]):
        return 'xray'
    if any(w in name_lower for w in ["mri", "brain", "tumor", "tumour"]):
        return 'mri'
    if any(w in name_lower for w in ["ct", "lung", "malignant", "benign"]):
        return 'ct'
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    arr = np.array(img)
    if arr.mean() > 100:
        return 'xray'
    elif 40 <= arr.mean() <= 120:
        return 'ct'
    else:
        return 'mri'

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_t = preprocess(contents)
    modality = detect_modality(contents, file.filename or "")

    with torch.no_grad():
        if modality == 'xray':
            logits = pneumonia_model(img_t)
            prob = torch.softmax(logits, 1).squeeze()
            return {
                "modality": "xray",
                "normal": round(prob[0].item(), 4),
                "pneumonia": round(prob[1].item(), 4),
                "prediction": "Pneumonia" if prob[1] > 0.85 else "Normal"
            }
        elif modality == 'mri':
            logits = brain_model(img_t)
            prob = torch.softmax(logits, 1).squeeze()
            classes = ['glioma', 'meningioma', 'pituitary', 'healthy']
            result = {"modality": "mri", "prediction": classes[prob.argmax().item()]}
            for c, p in zip(classes, prob):
                result[c] = round(p.item(), 4)
            return result
        else:  # ct
            if lung_model is not None:
                logits = lung_model(img_t)
                prob = torch.softmax(logits, 1).squeeze()
                classes = ['benign', 'malignant', 'normal']
                result = {"modality": "ct", "prediction": classes[prob.argmax().item()]}
                for c, p in zip(classes, prob):
                    result[c] = round(p.item(), 4)
                return result
            else:
                return {"error": "Lung cancer model not loaded. Train it first."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "pneumonia": pneumonia_model is not None,
            "brain_tumour": brain_model is not None,
            "lung_cancer": lung_model is not None
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)