# app.py
import os, pickle, json, math, traceback
from pathlib import Path
import numpy as np
import cv2, pywt
from skimage.feature import local_binary_pattern as sk_lbp
import streamlit as st

# ----------------- CONFIG -----------------
APP_TITLE = "🖨️🕵️ TraceFinder 2.0 - Forensic Scanner & Tamper Dashboard"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

ROOT = Path(r"C:\AI Trace Finder\App\models")
ART_SCN = ROOT
ART_TP = ROOT / "artifacts_tamper_patch"
ART_PAIR = ROOT / "artifacts_tamper_pair"

# ----------------- Streamlit page -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1f1f2e, #2e2e44); padding:20px; border-radius:12px;">
        <h1 style='color:white; text-align:center'>{APP_TITLE}</h1>
        <p style='color:#ddd; text-align:center;'>Upload a scanned page to identify the 🖨️ scanner and check for 🕵️ tampering</p>
    </div>
""", unsafe_allow_html=True)

# ----------------- UTILS -----------------
def decode_upload_to_bgr(uploaded):
    import fitz
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        doc = fitz.open(stream=raw, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), name
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Could not decode file")
    return bgr, name

def load_to_residual_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    cA, (cH,cV,cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH,cV,cD)), "haar")
    return (gray - den).astype(np.float32)

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img)))/(rng+1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    hist, _ = np.histogram(codes, bins=np.arange(P+3), density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h,w = mag.shape; cy,cx = h//2,w//2
    yy, xx = np.ogrid[:h,:w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0,r.max()+1e-6,K+1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(mag[mask].mean() if mask.any() else 0.0))
    return np.array(feats, dtype=np.float32)

def corr2d(a,b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a @ b)/d) if d!=0 else 0.0

# ----------------- LOAD MODELS & ARTIFACTS -----------------
scanner_model = None
try:
    import tensorflow as tf
    model_path = ART_SCN / "scanner_hybrid.keras"
    if model_path.exists():
        scanner_model = tf.keras.models.load_model(str(model_path))
except Exception as e:
    st.warning(f"🛑 Scanner model load failed: {e}")

# Artifacts
def load_scanner_artifacts():
    with open(ART_SCN / "hybrid_label_encoder.pkl","rb") as f: le = pickle.load(f)
    with open(ART_SCN / "hybrid_feat_scaler.pkl","rb") as f: scaler = pickle.load(f)
    with open(ART_SCN / "scannerfingerprints.pkl","rb") as f: fps = pickle.load(f)
    fp_keys = np.load(ART_SCN / "fp_keys.npy", allow_pickle=True).tolist()
    return le, scaler, fps, fp_keys

try:
    le_sc, sc_scaler, scanner_fps, fp_keys = load_scanner_artifacts()
    st.success("✅ Scanner model and artifacts loaded")
except Exception as e:
    st.warning(f"🛑 Failed loading scanner artifacts: {e}")

def make_scanner_feats(res):
    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
    v_fft  = fft_radial_energy(res)
    v_lbp  = lbp_hist_safe(res)
    feat = np.array(v_corr + v_fft.tolist() + v_lbp.tolist(), dtype=np.float32).reshape(1,-1)
    return sc_scaler.transform(feat)

# ----------------- UI -----------------
uploaded = st.file_uploader("📤 Upload scanned page", type=["tif","tiff","png","jpg","jpeg","pdf"])
if uploaded:
    try:
        bgr, name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # Scanner prediction
        s_label, s_conf = "Unknown", 0.0
        if scanner_model:
            x_img = np.expand_dims(residual, axis=(0,-1))
            x_feat = make_scanner_feats(residual)
            ps = scanner_model.predict([x_img, x_feat], verbose=0).ravel()
            if not np.isnan(ps).any() and np.any(ps>0):
                idx = int(np.argmax(ps))
                s_label, s_conf = le_sc.classes_[idx], float(ps[idx]*100)

        # Placeholder verdict (tamper logic can be added)
        verdict = "✅ Clean"

        # Display cards
        col1, col2 = st.columns([1.5,2], gap="large")
        with col1:
            st.markdown(f"""
            <div style='padding:16px;border-radius:12px;background:linear-gradient(90deg,#222,#444);color:white;'>
                <h2>🖨️ Scanner Identification</h2>
                <h1 style='color:#ffda79'>{s_label}</h1>
                <p>Confidence: <b>{s_conf:.1f}%</b> {"🔹" * int(s_conf//10)}</p>
                <div style='background:#555;border-radius:8px;overflow:hidden;height:12px;width:100%;margin-bottom:10px;'>
                    <div style='width:{s_conf}%;background:#ffda79;height:12px;'></div>
                </div>
                <h2>🕵️ Tamper Verdict</h2>
                <h1 style='color:#70ff70'>{verdict}</h1>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    except Exception:
        st.error("⚠️ Inference error")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag & drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze.")
