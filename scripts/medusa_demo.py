import streamlit as st
import torch, cv2, sys, numpy as np
from PIL import Image
sys.path.insert(0, '.')
from scripts.finetune_tiny_pneumonia import TinyClassifier
from scripts.finetune_tiny_brain_tumor_mae import TinyBrainTumorClassifier
from scripts.finetune_tiny_lung_cancer import TinyLungCancerClassifier

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="MEDUSA – Medical AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------------
# Custom CSS – dark mode, glassmorphism cards, animations
# -------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }
    .diagnosis-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .diagnosis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }
    .stFileUploader {
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s;
    }
    .stFileUploader:hover {
        border-color: #00d2ff !important;
        box-shadow: 0 0 20px rgba(0,210,255,0.2);
    }
    .metric-card {
        background: rgba(0, 210, 255, 0.1);
        border-radius: 12px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-left: 4px solid #00d2ff;
    }
    .stButton > button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.6rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0,210,255,0.5);
    }
    .stProgress > div > div {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5) !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Grad‑CAM function (target = patch_embed)
# -------------------------------------------------------------------
def gradcam(model, img_tensor, target_layer, class_idx=None):
    """Return heatmap (H, W) for the given class."""
    features = []
    gradients = []

    def forward_hook(module, inp, outp):
        features.append(outp)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)

    logits = model(img_tensor)
    if class_idx is None:
        class_idx = logits.argmax(dim=1).item()
    logits[0, class_idx].backward()

    hook_f.remove()
    hook_b.remove()

    pooled_grad = torch.mean(gradients[0], dim=[0, 2, 3])  # (C,)
    activations = features[0].squeeze(0)                    # (C, H, W)
    for i in range(activations.shape[0]):
        activations[i] *= pooled_grad[i]
    heatmap = torch.mean(activations, dim=0).detach().cpu()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)
    return heatmap.numpy()

# -------------------------------------------------------------------
# Load models once and cache
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    p = TinyClassifier(num_classes=2)
    p.load_state_dict(
        torch.load('medusa_tiny_pneumonia.pt', map_location='cpu', weights_only=False),
        strict=False
    )
    p.eval()

    b = TinyBrainTumorClassifier(num_classes=4)
    b.load_state_dict(
        torch.load('medusa_tiny_brain_tumor_v2.pt', map_location='cpu', weights_only=False),
        strict=False
    )
    b.eval()

    try:
        l = TinyLungCancerClassifier(num_classes=3)
        l.load_state_dict(
            torch.load('medusa_tiny_lung_cancer_v2.pt', map_location='cpu', weights_only=False),
            strict=False
        )
        l.eval()
    except:
        l = None

    return p, b, l

p_model, b_model, l_model = load_models()

# -------------------------------------------------------------------
# Helper: guess modality
# -------------------------------------------------------------------
def guess_modality(img_array, filename):
    avg = img_array.mean()
    name_lower = filename.lower()
    if any(w in name_lower for w in ["xray", "x‑ray", "chest"]):
        return "xray"
    if any(w in name_lower for w in ["mri", "brain", "tumor", "tumour"]):
        return "mri"
    if any(w in name_lower for w in ["ct", "lung", "malignant", "benign"]):
        return "ct"
    if avg > 100:
        return "xray"
    elif 40 <= avg <= 120:
        return "ct"
    else:
        return "mri"

# -------------------------------------------------------------------
# UI Header
# -------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>🧠 MEDUSA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #b0b0d0;'>Multi‑modal Diagnostic AI</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #8888aa;'>Pneumonia · Brain Tumour · Lung Cancer</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #3a3a5a;'>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------
st.markdown("<div style='text-align: center; margin: 2rem 0;'>"
             "<p style='font-size: 1.1rem; color: #ccc;'>Drop chest X‑rays, brain MRIs, or lung CT slices below</p>"
             "</div>", unsafe_allow_html=True)

uploads = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# -------------------------------------------------------------------
# Process and display results
# -------------------------------------------------------------------
if uploads:
    st.markdown("---")
    for upload in uploads:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error(f"Could not read {upload.name}")
            continue

        modality = guess_modality(img, upload.name)
        modality_names = {"xray": "Chest X‑Ray", "mri": "Brain MRI", "ct": "Lung CT"}
        mod_display = modality_names.get(modality, "Unknown")

        # Preprocess
        img_resized = cv2.resize(img, (224, 224))
        img_t = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255

        with torch.no_grad():
            if modality == "xray":
                prob = torch.softmax(p_model(img_t), 1).squeeze()
                pred_class = "Pneumonia" if prob[1] > 0.85 else "Normal"
                details = [("Normal", prob[0].item()), ("Pneumonia", prob[1].item())]
                model_used = p_model
                class_idx = 1 if prob[1] > 0.85 else 0
                icon = "🫁"
            elif modality == "mri":
                prob = torch.softmax(b_model(img_t), 1).squeeze()
                classes = ['Glioma', 'Meningioma', 'Pituitary', 'Healthy']
                pred_class = classes[prob.argmax().item()]
                details = [(c, p.item()) for c, p in zip(classes, prob)]
                model_used = b_model
                class_idx = prob.argmax().item()
                icon = "🧠"
            else:  # ct
                if l_model is not None:
                    prob = torch.softmax(l_model(img_t), 1).squeeze()
                    classes = ['Benign', 'Malignant', 'Normal']
                    pred_class = classes[prob.argmax().item()]
                    details = [(c, p.item()) for c, p in zip(classes, prob)]
                    model_used = l_model
                    class_idx = prob.argmax().item()
                else:
                    pred_class = "Model not trained yet"
                    details = []
                    model_used = None
                    class_idx = 0
                icon = "🫁"

        # Generate Grad‑CAM heatmap
        if model_used is not None:
            target_layer = model_used.encoder.patch_embed
            hm = gradcam(model_used, img_t, target_layer, class_idx)
            hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6,
                                           heatmap_colored, 0.4, 0)
        else:
            superimposed = None

        # --- Layout: original image (left) + Grad‑CAM overlay (right) + results (below) ---
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img, caption="Original Image", use_container_width=True, clamp=True)
        with col_img2:
            if superimposed is not None:
                st.image(superimposed, caption="MEDUSA Focus (Grad‑CAM)", use_container_width=True, clamp=True)
            else:
                st.image(img, caption="Original (heatmap unavailable)", use_container_width=True, clamp=True)

        # Results
        st.markdown(f"<div class='diagnosis-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #f0f0f0;'>{icon} {mod_display} Analysis</h3>", unsafe_allow_html=True)
        if "Normal" in pred_class or "Healthy" in pred_class:
            st.success(f"## {pred_class}")
        elif "Model not trained" in pred_class:
            st.warning(f"## {pred_class}")
        else:
            st.error(f"## {pred_class}")
        for label, prob_val in details:
            st.markdown(f"<div class='metric-card'>"
                         f"<span style='color: #ccc;'>{label}: <b>{prob_val:.1%}</b></span>"
                         f"</div>", unsafe_allow_html=True)
            st.progress(float(prob_val))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align: center; margin-top: 4rem; color: #888;'>"
                 "<p style='font-size: 2rem;'>📤</p>"
                 "<p>Upload one or more medical images to begin</p>"
                 "<p style='font-size: 0.8rem;'>Supported: Chest X‑rays, Brain MRI, Lung CT slices</p>"
                 "</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("<hr style='border: 1px solid #3a3a5a; margin-top: 3rem;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 0.8rem;'>"
             "MEDUSA · 850K params · CPU‑only · v1.0<br>"
             "<a href='https://github.com/yourusername/medusa' style='color: #888;'>GitHub</a>"
             "</p>", unsafe_allow_html=True)