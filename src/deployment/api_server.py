from fastapi import FastAPI, File, UploadFile
import torch
import io
import pydicom
import numpy as np
from PIL import Image
from src.models.medusa_foundation import MedusaFoundation

app = FastAPI(title="MEDUSA Inference API")

# Load model once at startup
model = None
@app.on_event("startup")
def load_model():
    global model
    model = MedusaFoundation(task='classification', num_classes=14)
    # model.load_encoder_weights('pretrained_mae.ckpt')
    model.eval()

@app.post("/predict/xray")
async def predict_xray(file: UploadFile = File(...)):
    """Receive a chest X-ray (PNG) and return disease probabilities."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')
    img = img.resize((224, 224))
    img_tensor = torch.from_numpy(np.array(img)).unsqueeze(0).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        mod_id = torch.tensor([0])  # 0 = X-ray
        logits = model(img_tensor, mod_id, is_3d=False)
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    return {"probabilities": probs}

@app.post("/predict/dicom")
async def predict_dicom(file: UploadFile = File(...)):
    """Receive a DICOM file, extract pixel array, return prediction."""
    contents = await file.read()
    dicom = pydicom.dcmread(io.BytesIO(contents))
    img = dicom.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # Assume 2D for simplicity; handle 3D later
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    # Determine modality from DICOM tag (simplified)
    mod_str = dicom.Modality if 'Modality' in dicom else 'CT'
    mod_map = {'CR':0, 'DX':0, 'CT':1, 'MR':2}
    mod_id = torch.tensor([mod_map.get(mod_str, 0)])

    with torch.no_grad():
        logits = model(img_tensor, mod_id, is_3d=False)
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    return {"modality": mod_str, "probabilities": probs}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)