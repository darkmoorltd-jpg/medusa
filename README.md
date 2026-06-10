# 🧠 MEDUSA — Multi‑Modal Medical AI

**One model. Three modalities. Three diseases. Runs on a CPU.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-pending-brightgreen.svg)](https://arxiv.org/)

MEDUSA is a single **851K‑parameter Vision Transformer** that diagnoses **pneumonia** (chest X‑ray), **brain tumours** (MRI), and **lung cancer** (CT).  
All training was done on a consumer laptop — **no GPU required**.  
The model is 3.4 MB, infers in 43 ms, and includes **Grad‑CAM explainability**.

---

## 📊 Performance

| Disease | Modality | Classes | Accuracy | Sensitivity (key class) |
|---------|----------|---------|----------|--------------------------|
| **Pneumonia** | Chest X‑ray | Normal / Pneumonia | **87.0%** | Pneumonia 88%, Normal 87% |
| **Brain Tumour** | Brain MRI | Glioma / Meningioma / Pituitary | **80.9%** | Pituitary 93% |
| **Lung Cancer** | Lung CT | Benign / Malignant / Normal | **82.9%** | Malignant 89% |

*All results on unseen test sets. Full per‑class metrics in the paper.*

---

## 🎥 Live Demo

A dark‑mode Streamlit app lets you upload images and see diagnoses with Grad‑CAM heatmaps.  
**→ [Launch demo](#)** (run locally: `streamlit run scripts/medusa_demo.py`)

![MEDUSA demo screenshot](docs/demo_screenshot.png)

---

## ⚙️ Why MEDUSA is different

- **One backbone for all modalities** — add a new disease in hours, not months.
- **Runs on a CPU** — model size < 4 MB, inference 43 ms on a laptop.
- **Self‑supervised pre‑training** — Masked Autoencoder learns from unlabelled data.
- **Explainable** — Grad‑CAM heatmaps show exactly what the model sees.
- **Production‑ready** — Multi‑head API (FastAPI) and interactive web demo included.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/medusa.git
cd medusa
pip install -r requirements.txt

# Test pneumonia on a chest X‑ray
python scripts/predict_pneumonia.py path/to/chest_xray.jpg

# Test brain tumour on an MRI
python scripts/predict_brain_tumor.py path/to/mri.jpg

# Test lung cancer on a CT slice
python scripts/predict_lung_cancer.py path/to/ct_slice.jpg

# Launch the web demo
streamlit run scripts/medusa_demo.py

# Start the multi‑head API
python scripts/multi_head_api.py