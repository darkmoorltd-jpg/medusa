import streamlit as st
import torch, cv2, sys, numpy as np
from PIL import Image
from io import BytesIO
import base64, time, datetime, tempfile
import torch.nn as nn
from fpdf import FPDF

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(page_title="MEDUSA", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# -------------------------------------------------------------------
# Dark mode CSS
# -------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #f0f0f0; }
    .diagnosis-card { background: rgba(255,255,255,0.08); backdrop-filter: blur(16px); border-radius: 20px; padding: 2rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.15); }
    .stFileUploader { border: 2px dashed rgba(255,255,255,0.3)!important; border-radius:16px!important; padding:2rem!important; }
    .metric-card { background: rgba(0,210,255,0.1); border-radius:12px; padding:0.8rem; border-left:4px solid #00d2ff; }
    .stProgress > div > div { background: linear-gradient(45deg, #00d2ff, #3a7bd5)!important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# TinyViT + classifiers (all in one file)
# -------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TinyViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)

class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyViT()
        self.head = nn.Linear(128, 2)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats[:, 0, :])

class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyViT()
        self.head = nn.Linear(128, 4)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats[:, 0, :])

class LungCancerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyViT()
        self.head = nn.Linear(128, 3)
    def forward(self, x):
        feats = self.encoder(x)
        return self.head(feats[:, 0, :])

# -------------------------------------------------------------------
# Model loading (cached)
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    p = PneumoniaClassifier()
    p.load_state_dict(torch.load('medusa_tiny_pneumonia.pt', map_location='cpu', weights_only=False), strict=False)
    p.eval()
    b = BrainTumorClassifier()
    b.load_state_dict(torch.load('medusa_tiny_brain_tumor_v2.pt', map_location='cpu', weights_only=False), strict=False)
    b.eval()
    l = None
    try:
        l = LungCancerClassifier()
        l.load_state_dict(torch.load('medusa_tiny_lung_cancer_v2.pt', map_location='cpu', weights_only=False), strict=False)
        l.eval()
    except: pass
    return p, b, l

p_model, b_model, l_model = load_models()

# -------------------------------------------------------------------
# Grad‑CAM
# -------------------------------------------------------------------
def gradcam(model, img_tensor, target_layer, class_idx=None):
    features, gradients = [], []
    def fhook(m,i,o): features.append(o)
    def bhook(m,gi,go): gradients.append(go[0])
    h1 = target_layer.register_forward_hook(fhook)
    h2 = target_layer.register_full_backward_hook(bhook)
    logits = model(img_tensor)
    if class_idx is None: class_idx = logits.argmax(dim=1).item()
    logits[0, class_idx].backward()
    h1.remove(); h2.remove()
    pooled = torch.mean(gradients[0], dim=[0,2,3])
    acts = features[0].squeeze(0)
    for i in range(acts.shape[0]): acts[i] *= pooled[i]
    hm = torch.mean(acts, dim=0).detach().cpu().numpy()
    hm = np.maximum(hm, 0); hm /= (hm.max()+1e-8)
    return hm

# -------------------------------------------------------------------
# Modality guess
# -------------------------------------------------------------------
def guess_modality(img, fname):
    avg = img.mean(); name = fname.lower()
    if any(w in name for w in ["xray","chest"]): return "xray"
    if any(w in name for w in ["mri","brain","tumor","tumour"]): return "mri"
    if any(w in name for w in ["ct","lung"]): return "ct"
    if avg > 100: return "xray"
    elif 40 <= avg <= 120: return "ct"
    else: return "mri"

# -------------------------------------------------------------------
# PDF generation – all strings use ASCII hyphens only
# -------------------------------------------------------------------
def generate_pdf(patient_id, modality, predictions, original_b64, gradcam_b64, hospital_name="DARKMOOR LTD"):
    # Replace any non-ASCII hyphen that might slip through
    modality = modality.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    patient_id = patient_id.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    hospital_name = hospital_name.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Header
    pdf.set_fill_color(30, 30, 50)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,hospital_name,ln=1,align='L')
    pdf.set_font("Helvetica","",8)
    pdf.cell(0,5,"MEDUSA AI Diagnostic Report",ln=1,align='L')
    pdf.set_y(35)
    pdf.set_text_color(0,0,0)
    pdf.set_font("Helvetica","B",14)
    pdf.cell(0,8,"RADIOLOGY REPORT",ln=1,align='C')
    pdf.ln(4)
    pdf.set_font("Helvetica","B",10)
    w=70
    pdf.cell(w,6,"Patient ID:",border=0)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,6,patient_id,ln=1)
    pdf.set_font("Helvetica","B",10)
    pdf.cell(w,6,"Modality:",border=0)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,6,modality,ln=1)
    pdf.set_font("Helvetica","B",10)
    pdf.cell(w,6,"Date:",border=0)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,6,datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),ln=1)
    pdf.ln(6)
    # Images
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"IMAGES",ln=1)
    pdf.ln(2)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_orig:
        orig_img = Image.open(BytesIO(base64.b64decode(original_b64)))
        orig_img.save(tmp_orig.name)
        orig_path = tmp_orig.name
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_grad:
        grad_img = Image.open(BytesIO(base64.b64decode(gradcam_b64)))
        grad_img.save(tmp_grad.name)
        grad_path = tmp_grad.name
    x_left,x_right = 10,110
    y_img = pdf.get_y()
    pdf.image(orig_path, x=x_left, y=y_img, w=85)
    pdf.set_font("Helvetica","",8)
    pdf.text(x_left, y_img+80, "Original Image")
    pdf.image(grad_path, x=x_right, y=y_img, w=85)
    pdf.text(x_right, y_img+80, "AI Focus (Grad-CAM)")
    pdf.set_y(y_img+85)
    pdf.ln(8)
    # Findings
    pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"FINDINGS",ln=1)
    pdf.ln(2)
    pdf.set_fill_color(240,240,240)
    pdf.set_font("Helvetica","B",10)
    pdf.cell(80,7,"Finding",border=1,fill=True)
    pdf.cell(40,7,"Probability",border=1,fill=True,align='C')
    pdf.ln()
    pdf.set_font("Helvetica","",10)
    for label, prob in predictions:
        label = label.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        pdf.cell(80,7,label,border=1)
        pdf.cell(40,7,f"{prob:.1%}",border=1,align='C')
        pdf.ln()
    pdf.ln(5)
    # Disclaimer
    pdf.set_font("Helvetica","I",7)
    pdf.set_text_color(100,100,100)
    pdf.multi_cell(0,4,"Disclaimer: MEDUSA is an AI screening tool. This report is intended to assist qualified radiologists; it does not constitute a final medical diagnosis. Always correlate with clinical findings.")
    # Footer
    pdf.set_y(-20)
    pdf.set_font("Helvetica","",7)
    pdf.set_text_color(150,150,150)
    pdf.cell(0,10,f"Generated by DARKMOOR LTD | MEDUSA v1.0 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",align='C')
    pdf_bytes = pdf.output(dest="S")
    return base64.b64encode(pdf_bytes).decode()

# -------------------------------------------------------------------
# Session state for patient history
# -------------------------------------------------------------------
if "patient_records" not in st.session_state:
    st.session_state.patient_records = {}

# -------------------------------------------------------------------
# UI Header
# -------------------------------------------------------------------
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.markdown("<h1 style='text-align:center;font-size:3rem;'>🧠 MEDUSA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#b0b0d0;'>Multi-modal Diagnostic AI</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #3a3a5a;'>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Patient ID & history
# -------------------------------------------------------------------
patient_id = st.text_input("Patient ID (optional - leave blank to skip history)", value="")
if patient_id and patient_id in st.session_state.patient_records:
    with st.expander(f"📋 History for Patient {patient_id}"):
        for scan in st.session_state.patient_records[patient_id][-5:]:
            st.write(f"{scan['time']} — {scan['modality']}: {scan['prediction']}")

# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------
uploads = st.file_uploader("Upload chest X-rays, brain MRIs, or lung CT slices", type=["jpg","jpeg","png"], accept_multiple_files=True, label_visibility="collapsed")

if uploads:
    override = st.radio("Modality (override if auto-detection fails):", ["Auto-Detect", "Chest X-Ray", "Brain MRI", "Lung CT"], horizontal=True)

    for upload in uploads:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        if override == "Chest X-Ray": modality = "xray"
        elif override == "Brain MRI": modality = "mri"
        elif override == "Lung CT": modality = "ct"
        else: modality = guess_modality(img, upload.name)

        mod_names = {"xray":"Chest X-Ray","mri":"Brain MRI","ct":"Lung CT"}
        mod_disp = mod_names.get(modality, "Unknown")

        img_resized = cv2.resize(img, (224,224))
        img_t = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()/255

        with torch.no_grad():
            if modality == "xray":
                logits = p_model(img_t)
                prob = torch.softmax(logits, 1).squeeze()
                pred = "Pneumonia" if prob[1]>0.85 else "Normal"
                details = [("Normal", prob[0].item()), ("Pneumonia", prob[1].item())]
                model_used = p_model; cls = 1 if prob[1]>0.85 else 0; icon="🫁"
            elif modality == "mri":
                logits = b_model(img_t)
                prob = torch.softmax(logits, 1).squeeze()
                classes = ['Glioma','Meningioma','Pituitary','Healthy']
                pred = classes[prob.argmax().item()]
                details = [(c, p.item()) for c,p in zip(classes, prob)]
                model_used = b_model; cls = prob.argmax().item(); icon="🧠"
            else:
                if l_model:
                    logits = l_model(img_t)
                    prob = torch.softmax(logits, 1).squeeze()
                    classes = ['Benign','Malignant','Normal']
                    pred = classes[prob.argmax().item()]
                    details = [(c, p.item()) for c,p in zip(classes, prob)]
                    model_used = l_model; cls = prob.argmax().item()
                else: pred = "Model not trained yet"; details=[]; model_used=None; cls=0
                icon="🫁"

        superimposed = None
        orig_b64 = None
        if model_used:
            hm = gradcam(model_used, img_t, model_used.encoder.patch_embed, cls)
            hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
            heat_colored = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heat_colored, 0.4, 0)
            _, orig_buffer = cv2.imencode('.png', img)
            orig_b64 = base64.b64encode(orig_buffer).decode()
            _, grad_buffer = cv2.imencode('.png', superimposed)
            grad_b64 = base64.b64encode(grad_buffer).decode()

        if patient_id and model_used:
            if patient_id not in st.session_state.patient_records:
                st.session_state.patient_records[patient_id] = []
            st.session_state.patient_records[patient_id].append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "modality": mod_disp,
                "prediction": pred,
                "details": details
            })

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img, caption="Original Image", width='stretch', clamp=True)
        with col_img2:
            if superimposed is not None:
                st.image(superimposed, caption="MEDUSA Focus (Grad-CAM)", width='stretch', clamp=True)

        st.markdown(f"<div class='diagnosis-card'><h3>{icon} {mod_disp} Analysis</h3>", unsafe_allow_html=True)
        if "Normal" in pred or "Healthy" in pred: st.success(f"## {pred}")
        elif "Pneumonia" in pred: st.error(f"## {pred}")
        else: st.warning(f"## {pred}")
        for label, pval in details:
            st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
            st.progress(float(pval))
        st.markdown("</div>", unsafe_allow_html=True)

        feedback = st.radio("Radiologist confirmation:", ["Agree", "Disagree"], key=f"fb_{upload.name}_{time.time()}")
        if feedback == "Disagree":
            correct = st.text_input("Correct diagnosis:", key=f"corr_{upload.name}_{time.time()}")
            if correct:
                st.info(f"Override saved: {correct} (AI predicted: {pred})")

        if orig_b64 and grad_b64:
            pdf_b64 = generate_pdf(patient_id or "Unknown", mod_disp, details, orig_b64, grad_b64)
            href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="medusa_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

        if modality == "xray" and prob[1] > 0.95:
            st.error("⚠️ CRITICAL FINDING — PNEUMONIA (high confidence)")

        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #3a3a5a;margin-top:3rem;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;'>🥬 Powered by DARKMOOR LTD · Darkmoorltd@gmail.com<br>MEDUSA · 850K params · CPU-only · v1.0</p>", unsafe_allow_html=True)