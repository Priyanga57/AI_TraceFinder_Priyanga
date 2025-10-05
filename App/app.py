# app.py

import os, pickle, math, json, re, glob
import numpy as np
import streamlit as st
import cv2, pywt
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
ART_SCN = os.path.join(BASE_DIR, "models")           # Scanner & feature models
ART_TP  = os.path.join(BASE_DIR, "models", "artifacts_tamper_patch")
ART_PAIR = os.path.join(BASE_DIR, "models", "artifacts_tamper_pair")
TAMP_ROOT = os.path.join(ART_SCN, "Tampered images")

APP_TITLE = "🖨️ TraceFinder 2.0 - Forensic Scanner & Tamper Dashboard"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16
TOPK = 0.30
HIT_THR = 0.85
MIN_HITS = 2

# ---------------- PDF backends ----------------
PDF_BACKEND = None
try:
    from pdf2image import convert_from_bytes as pdf2img_convert
    PDF_BACKEND = "pdf2image"
except Exception:
    try:
        import fitz
        PDF_BACKEND = "pymupdf"
    except Exception:
        PDF_BACKEND = None

# ----------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"""
    <style>
        body {{
            background: linear-gradient(135deg, #1f1f2e, #2d2d44);
            color: #ffffff;
        }}
        .card {{
            padding: 16px;
            border-radius: 12px;
            background: #1f1f2e;
            border: 1px solid #444;
            margin-bottom: 12px;
        }}
        .confidence-bar {{
            background: #444;
            border-radius: 8px;
            overflow: hidden;
            height: 12px;
            width: 100%;
            margin-bottom:10px;
        }}
        .confidence-fill {{
            height: 12px;
            background: #ffda79;
        }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("🔍 Upload a scanned page to identify the **scanner source** & check for **tampering**.")

# ----------------- UTILITIES ----------------
def decode_upload_to_bgr(uploaded):
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]

    if ext == ".pdf":
        if PDF_BACKEND == "pdf2image":
            pages = pdf2img_convert(raw, dpi=300, fmt="png")
            pil = pages[0].convert("RGB")
            rgb = np.array(pil)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), name
        elif PDF_BACKEND == "pymupdf":
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), name
        else:
            raise ImportError("PDF support unavailable. Install pdf2image or pymupdf")
    
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None: raise ValueError("Could not decode file")
    return bgr, name

def load_to_residual_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (gray - den).astype(np.float32)

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats = [float(mag[(r>=bins[i]) & (r<bins[i+1])].mean() if ((r>=bins[i]) & (r<bins[i+1])).any() else 0.0) for i in range(K)]
    return np.asarray(feats, dtype=np.float32)

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def make_scanner_feats(res):
    v_corr = [corr2d(res, fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(res, K=6)
    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
    v = np.array(v_corr + v_fft.tolist() + v_lbp.tolist(), dtype=np.float32).reshape(1, -1)
    return sc_sc.transform(v)

# ----------------- LOAD MODELS ----------------
import tensorflow as tf

hyb_model = None
cand = [os.path.join(ART_SCN, "scanner_hybrid.keras"),
        os.path.join(ART_SCN, "scanner_hybrid.h5")]
found = next((p for p in cand if os.path.exists(p)), None)
if found: hyb_model = tf.keras.models.load_model(found)

# Load artifacts
with open(os.path.join(ART_SCN, "hybrid_label_encoder.pkl"), "rb") as f: le_sc = pickle.load(f)
with open(os.path.join(ART_SCN, "hybrid_feat_scaler.pkl"), "rb") as f: sc_sc = pickle.load(f)
with open(os.path.join(ART_SCN, "scanner_fingerprints.pkl"), "rb") as f: fps = pickle.load(f)
fp_keys = np.load(os.path.join(ART_SCN, "fp_keys.npy"), allow_pickle=True).tolist()

# ----------------- UI -----------------
uploaded = st.file_uploader(
    "📤 Upload scanned page", 
    type=["tif","tiff","png","jpg","jpeg","pdf"], 
    label_visibility="collapsed"
)

if uploaded:
    try:
        bgr, fname = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # ---- Scanner Prediction ----
        s_label, s_conf = "Unknown", 0.0
        if hyb_model:
            x_img = np.expand_dims(residual, axis=(0,-1))
            x_feat = make_scanner_feats(residual)
            ps = hyb_model.predict([x_img, x_feat], verbose=0).ravel()
            s_idx = int(np.argmax(ps)); s_label = le_sc.classes_[s_idx]; s_conf = float(ps[s_idx]*100.0)

        # ---- Tamper placeholder (replace with your logic) ----
        verdict = "✅ Clean"  # You can integrate your tamper inference here

        # ---- Display ----
        col1, col2 = st.columns([1.5, 2], gap="large")
        with col1:
            st.markdown(f"""
                <div class='card'>
                    <h3>🖨️ Scanner Identification</h3>
                    <h2>{s_label} 🖨️</h2>
                    <div class='confidence-bar'>
                        <div class='confidence-fill' style='width:{s_conf}%;'></div>
                    </div>
                    <p>Confidence: <b>{s_conf:.1f}%</b> 📊</p>
                </div>
                <div class='card'>
                    <h3>🕵️ Tamper Verdict</h3>
                    <h2>{verdict} 🔍</h2>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    except Exception as e:
        import traceback
        st.error("⚠️ Inference error!")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag and drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze.")
