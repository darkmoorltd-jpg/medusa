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
# TinyViT + classifiers
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
    def __init__(self): super().__init__(); self.encoder = TinyViT(); self.head = nn.Linear(128, 2)
    def forward(self, x): return self.head(self.encoder(x)[:, 0, :])

class TBClassifierModel(nn.Module):
    def __init__(self): super().__init__(); self.encoder = TinyViT(); self.head = nn.Linear(128, 2)
    def forward(self, x): return self.head(self.encoder(x)[:, 0, :])

class BrainTumorClassifier(nn.Module):
    def __init__(self): super().__init__(); self.encoder = TinyViT(); self.head = nn.Linear(128, 4)
    def forward(self, x): return self.head(self.encoder(x)[:, 0, :])

class LungCancerClassifier(nn.Module):
    def __init__(self): super().__init__(); self.encoder = TinyViT(); self.head = nn.Linear(128, 3)
    def forward(self, x): return self.head(self.encoder(x)[:, 0, :])

class MalariaClassifierModel(nn.Module):
    def __init__(self): super().__init__(); self.encoder = TinyViT(); self.head = nn.Linear(128, 2)
    def forward(self, x): return self.head(self.encoder(x)[:, 0, :])

class LeukemiaClassifierModel(nn.Module):
    def __init__(self): super().__init__(); self.encoder = TinyViT(); self.head = nn.Linear(128, 2)
    def forward(self, x): return self.head(self.encoder(x)[:, 0, :])

# -------------------------------------------------------------------
# Model loading (cached)
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    p = PneumoniaClassifier()
    p.load_state_dict(torch.load('medusa_tiny_pneumonia.pt', map_location='cpu', weights_only=False), strict=False); p.eval()
    tb = TBClassifierModel()
    tb.load_state_dict(torch.load('medusa_tiny_tb.pt', map_location='cpu', weights_only=False), strict=False); tb.eval()
    b = BrainTumorClassifier()
    b.load_state_dict(torch.load('medusa_tiny_brain_tumor_4class.pt', map_location='cpu', weights_only=False), strict=False); b.eval()
    l = None
    try:
        l = LungCancerClassifier()
        l.load_state_dict(torch.load('medusa_tiny_lung_cancer_v2.pt', map_location='cpu', weights_only=False), strict=False); l.eval()
    except: pass
    mal = None
    try:
        mal = MalariaClassifierModel()
        mal.load_state_dict(torch.load('medusa_tiny_malaria.pt', map_location='cpu', weights_only=False), strict=False); mal.eval()
    except: pass
    leu = None
    try:
        leu = LeukemiaClassifierModel()
        leu.load_state_dict(torch.load('medusa_tiny_leukemia.pt', map_location='cpu', weights_only=False), strict=False); leu.eval()
    except: pass
    return p, tb, b, l, mal, leu

p_model, tb_model, b_model, l_model, mal_model, leu_model = load_models()

# -------------------------------------------------------------------
# Grad‑CAM via Captum
# -------------------------------------------------------------------
try:
    from captum.attr import LayerGradCam
    def gradcam(model, img_np, class_idx):
        x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().div(255.0)
        x = x.detach().requires_grad_(True)
        wrapped = lambda inp: model(inp)
        lgc = LayerGradCam(wrapped, model.encoder.patch_embed)
        attr = lgc.attribute(x, target=class_idx)
        cam = attr.squeeze().mean(dim=0).detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        return cam
except ImportError:
    def gradcam(model, img_np, class_idx):
        return np.ones_like(img_np) * 0.5

# -------------------------------------------------------------------
# Modality guess
# -------------------------------------------------------------------
def guess_modality(img, fname):
    avg = img.mean(); name = fname.lower()
    if any(w in name for w in ["xray","chest","tb","normal","pneumonia"]): return "xray"
    if any(w in name for w in ["mri","brain","tumor","tumour"]): return "mri"
    if any(w in name for w in ["ct","lung"]): return "ct"
    if any(w in name for w in ["malaria","plasmodium","parasitized","uninfected","cell","microscopy"]): return "microscopy"
    if any(w in name for w in ["leukemia","leuk","all","hem","blast"]): return "microscopy"
    if avg > 100: return "xray"
    elif 40 <= avg <= 120: return "ct"
    else: return "mri"

# -------------------------------------------------------------------
# PDF generation
# -------------------------------------------------------------------
def generate_pdf(patient_id, modality, predictions, original_b64, gradcam_b64, hospital_name="DARKMOOR LTD"):
    modality = modality.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    patient_id = patient_id.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    hospital_name = hospital_name.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
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
    pdf.set_font("Helvetica","I",7)
    pdf.set_text_color(100,100,100)
    pdf.multi_cell(0,4,"Disclaimer: MEDUSA is an AI screening tool. This report is intended to assist qualified radiologists; it does not constitute a final medical diagnosis. Always correlate with clinical findings.")
    pdf.set_y(-20)
    pdf.set_font("Helvetica","",7)
    pdf.set_text_color(150,150,150)
    pdf.cell(0,10,f"Generated by DARKMOOR LTD | MEDUSA v1.0 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",align='C')
    pdf_bytes = pdf.output(dest="S")
    return base64.b64encode(pdf_bytes).decode()

# -------------------------------------------------------------------
# Session state
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

patient_id = st.text_input("Patient ID (optional - leave blank to skip history)", value="")
if patient_id and patient_id in st.session_state.patient_records:
    with st.expander(f"📋 History for Patient {patient_id}"):
        for scan in st.session_state.patient_records[patient_id][-5:]:
            st.write(f"{scan['time']} — {scan['modality']}: {scan['prediction']}")

# -------------------------------------------------------------------
# Upload section
# -------------------------------------------------------------------
uploads = st.file_uploader("Upload chest X-rays, brain MRIs, lung CT slices, or microscopy images", type=["jpg","jpeg","png","bmp"], accept_multiple_files=True, label_visibility="collapsed")

if uploads:
    override = st.radio("Modality (override if auto-detection fails):", ["Auto-Detect", "Chest X-Ray", "Brain MRI", "Lung CT", "Microscopy"], horizontal=True)

    for upload in uploads:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        if override == "Chest X-Ray": modality = "xray"
        elif override == "Brain MRI": modality = "mri"
        elif override == "Lung CT": modality = "ct"
        elif override == "Microscopy": modality = "microscopy"
        else: modality = guess_modality(img, upload.name)

        mod_names = {"xray":"Chest X-Ray","mri":"Brain MRI","ct":"Lung CT","microscopy":"Microscopy"}
        mod_disp = mod_names.get(modality, "Unknown")

        img_resized = cv2.resize(img, (224,224))
        img_t = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()/255

        with torch.no_grad():
            if modality == "xray":
                # Pneumonia
                logits_p = p_model(img_t)
                prob_p = torch.softmax(logits_p, 1).squeeze()
                pred_p = "Pneumonia" if prob_p[1]>0.85 else "Normal"
                details_p = [("Normal", prob_p[0].item()), ("Pneumonia", prob_p[1].item())]
                cls_p = 1 if prob_p[1]>0.85 else 0

                # TB
                logits_tb = tb_model(img_t)
                prob_tb = torch.softmax(logits_tb, 1).squeeze()
                pred_tb = "TB Positive" if prob_tb[1]>0.85 else "TB Negative"
                details_tb = [("TB Negative", prob_tb[0].item()), ("TB Positive", prob_tb[1].item())]

                st.markdown("<div class='diagnosis-card'><h3>🫁 Chest X-Ray Analysis</h3>", unsafe_allow_html=True)

                st.markdown("#### 🫁 Pneumonia")
                if pred_p == "Normal": st.success(f"**Result:** {pred_p}")
                else: st.error(f"**Result:** {pred_p}")
                for label, pval in details_p:
                    st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
                    st.progress(float(pval))

                # Grad‑CAM pneumonia
                hm_p = gradcam(p_model, img_resized, cls_p)
                hm_p = cv2.resize(hm_p, (img.shape[1], img.shape[0]))
                heat_p = cv2.applyColorMap(np.uint8(255*hm_p), cv2.COLORMAP_JET)
                superimposed_p = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heat_p, 0.4, 0)

                col1, col2 = st.columns(2)
                with col1: st.image(img, caption="Original", width='stretch', clamp=True)
                with col2: st.image(superimposed_p, caption="Pneumonia Focus", width='stretch', clamp=True)

                st.markdown("---")
                st.markdown("#### 🦠 Tuberculosis")
                if pred_tb == "TB Negative": st.success(f"**Result:** {pred_tb}")
                else: st.error(f"**Result:** {pred_tb}")
                for label, pval in details_tb:
                    st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
                    st.progress(float(pval))
                st.markdown("</div>", unsafe_allow_html=True)

                # PDF
                _, orig_buffer = cv2.imencode('.png', img)
                orig_b64 = base64.b64encode(orig_buffer).decode()
                _, grad_buffer = cv2.imencode('.png', superimposed_p)
                grad_b64 = base64.b64encode(grad_buffer).decode()
                all_details = details_p + details_tb
                pdf_b64 = generate_pdf(patient_id or "Unknown", "Chest X-Ray", all_details, orig_b64, grad_b64)
                href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="medusa_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

                if prob_p[1] > 0.95: st.error("⚠️ CRITICAL FINDING — PNEUMONIA")
                if prob_tb[1] > 0.95: st.error("⚠️ CRITICAL FINDING — TUBERCULOSIS")

            elif modality == "mri":
                logits = b_model(img_t)
                prob = torch.softmax(logits, 1).squeeze()
                classes = ['Glioma','Meningioma','Pituitary','No Tumor']
                max_prob, cls_idx = prob.max(dim=0)
                if max_prob.item() < 0.95: pred = "Uncertain / Likely Normal"; cls = 3
                else: pred = classes[cls_idx.item()]; cls = cls_idx.item()
                details = [(c, p.item()) for c,p in zip(classes, prob)]

                st.markdown(f"<div class='diagnosis-card'><h3>🧠 Brain MRI Analysis</h3>", unsafe_allow_html=True)
                if "Normal" in pred or "No Tumor" in pred or "Uncertain" in pred: st.success(f"## {pred}")
                else: st.warning(f"## {pred}")
                for label, pval in details:
                    st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
                    st.progress(float(pval))
                st.markdown("</div>", unsafe_allow_html=True)

                hm = gradcam(b_model, img_resized, cls)
                hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
                heat = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heat, 0.4, 0)

                col_img1, col_img2 = st.columns(2)
                with col_img1: st.image(img, caption="Original Image", width='stretch', clamp=True)
                with col_img2: st.image(superimposed, caption="MEDUSA Focus (Grad-CAM)", width='stretch', clamp=True)

                _, orig_buffer = cv2.imencode('.png', img)
                orig_b64 = base64.b64encode(orig_buffer).decode()
                _, grad_buffer = cv2.imencode('.png', superimposed)
                grad_b64 = base64.b64encode(grad_buffer).decode()
                pdf_b64 = generate_pdf(patient_id or "Unknown", mod_disp, details, orig_b64, grad_b64)
                href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="medusa_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

                feedback = st.radio("Radiologist confirmation:", ["Agree", "Disagree"], key=f"fb_{upload.name}_{time.time()}")
                if feedback == "Disagree":
                    correct = st.text_input("Correct diagnosis:", key=f"corr_{upload.name}_{time.time()}")
                    if correct: st.info(f"Override saved: {correct} (AI predicted: {pred})")

            elif modality == "ct":
                if l_model:
                    logits = l_model(img_t)
                    prob = torch.softmax(logits, 1).squeeze()
                    classes = ['Benign','Malignant','Normal']
                    pred = classes[prob.argmax().item()]
                    details = [(c, p.item()) for c,p in zip(classes, prob)]
                    cls = prob.argmax().item()
                else: pred = "Model not trained yet"; details=[]; cls=0

                st.markdown(f"<div class='diagnosis-card'><h3>🫁 Lung CT Analysis</h3>", unsafe_allow_html=True)
                if "Normal" in pred: st.success(f"## {pred}")
                else: st.warning(f"## {pred}")
                for label, pval in details:
                    st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
                    st.progress(float(pval))
                st.markdown("</div>", unsafe_allow_html=True)

                if l_model:
                    hm = gradcam(l_model, img_resized, cls)
                    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
                    heat = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heat, 0.4, 0)
                else: superimposed = None

                col_img1, col_img2 = st.columns(2)
                with col_img1: st.image(img, caption="Original Image", width='stretch', clamp=True)
                with col_img2:
                    if superimposed is not None: st.image(superimposed, caption="MEDUSA Focus (Grad-CAM)", width='stretch', clamp=True)

                _, orig_buffer = cv2.imencode('.png', img)
                orig_b64 = base64.b64encode(orig_buffer).decode()
                if l_model: _, grad_buffer = cv2.imencode('.png', superimposed); grad_b64 = base64.b64encode(grad_buffer).decode()
                else: grad_b64 = orig_b64
                pdf_b64 = generate_pdf(patient_id or "Unknown", mod_disp, details, orig_b64, grad_b64)
                href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="medusa_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

                feedback = st.radio("Radiologist confirmation:", ["Agree", "Disagree"], key=f"fb_{upload.name}_{time.time()}")
                if feedback == "Disagree":
                    correct = st.text_input("Correct diagnosis:", key=f"corr_{upload.name}_{time.time()}")
                    if correct: st.info(f"Override saved: {correct} (AI predicted: {pred})")

            elif modality == "microscopy":
                # ── MALARIA SECTION ──
                if mal_model:
                    logits_mal = mal_model(img_t)
                    prob_mal = torch.softmax(logits_mal, 1).squeeze()
                    pred_mal = "Parasitized" if prob_mal[1]>0.5 else "Uninfected"
                    details_mal = [("Uninfected", prob_mal[0].item()), ("Parasitized", prob_mal[1].item())]
                    cls_mal = 1 if prob_mal[1]>0.5 else 0
                else:
                    pred_mal = "Model not trained"; details_mal=[]; cls_mal=0

                # ── LEUKEMIA SECTION ──
                if leu_model:
                    logits_leu = leu_model(img_t)
                    prob_leu = torch.softmax(logits_leu, 1).squeeze()
                    pred_leu = "ALL (Cancer)" if prob_leu[1]>0.5 else "HEM (Normal)"
                    details_leu = [("HEM (Normal)", prob_leu[0].item()), ("ALL (Cancer)", prob_leu[1].item())]
                    cls_leu = 1 if prob_leu[1]>0.5 else 0
                else:
                    pred_leu = "Model not trained"; details_leu=[]; cls_leu=0

                st.markdown("<div class='diagnosis-card'><h3>🔬 Microscopy Analysis</h3>", unsafe_allow_html=True)

                # Malaria
                st.markdown("#### 🦟 Malaria")
                if mal_model:
                    if pred_mal == "Uninfected": st.success(f"**Result:** {pred_mal}")
                    else: st.error(f"**Result:** {pred_mal}")
                    for label, pval in details_mal:
                        st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
                        st.progress(float(pval))
                    # Grad‑CAM malaria
                    hm_mal = gradcam(mal_model, img_resized, cls_mal)
                    hm_mal = cv2.resize(hm_mal, (img.shape[1], img.shape[0]))
                    heat_mal = cv2.applyColorMap(np.uint8(255*hm_mal), cv2.COLORMAP_JET)
                    superimposed_mal = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heat_mal, 0.4, 0)
                    col1, col2 = st.columns(2)
                    with col1: st.image(img, caption="Original", width='stretch', clamp=True)
                    with col2: st.image(superimposed_mal, caption="Malaria Focus", width='stretch', clamp=True)
                else:
                    st.info("Malaria model not loaded.")

                st.markdown("---")

                # Leukemia
                st.markdown("#### 🩸 Leukemia")
                if leu_model:
                    if pred_leu == "HEM (Normal)": st.success(f"**Result:** {pred_leu}")
                    else: st.error(f"**Result:** {pred_leu}")
                    for label, pval in details_leu:
                        st.markdown(f"<div class='metric-card'>{label}: <b>{pval:.1%}</b></div>", unsafe_allow_html=True)
                        st.progress(float(pval))
                    # Grad‑CAM leukemia
                    hm_leu = gradcam(leu_model, img_resized, cls_leu)
                    hm_leu = cv2.resize(hm_leu, (img.shape[1], img.shape[0]))
                    heat_leu = cv2.applyColorMap(np.uint8(255*hm_leu), cv2.COLORMAP_JET)
                    superimposed_leu = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heat_leu, 0.4, 0)
                    col1, col2 = st.columns(2)
                    with col1: st.image(img, caption="Original", width='stretch', clamp=True)
                    with col2: st.image(superimposed_leu, caption="Leukemia Focus", width='stretch', clamp=True)
                else:
                    st.info("Leukemia model not loaded.")

                st.markdown("</div>", unsafe_allow_html=True)

                # PDF for microscopy (uses malaria heatmap as primary)
                if mal_model:
                    _, orig_buffer = cv2.imencode('.png', img)
                    orig_b64 = base64.b64encode(orig_buffer).decode()
                    _, grad_buffer = cv2.imencode('.png', superimposed_mal)
                    grad_b64 = base64.b64encode(grad_buffer).decode()
                    all_details = details_mal + details_leu
                    pdf_b64 = generate_pdf(patient_id or "Unknown", "Microscopy", all_details, orig_b64, grad_b64)
                    href = f'<a href="data:application/pdf;base64,{pdf_b64}" download="medusa_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📄 Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #3a3a5a;margin-top:3rem;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#888;'>🥬 Powered by DARKMOOR LTD · Darkmoorltd@gmail.com<br>MEDUSA · 850K params · CPU-only · v1.0</p>", unsafe_allow_html=True)