import os, re, glob, math, json, pickle
from pathlib import Path
import numpy as np
import streamlit as st
import cv2, pywt

# PDF support
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

from skimage.feature import local_binary_pattern as sk_lbp

# --- CONFIG ---
APP_TITLE = "🔍 TraceFinder — Scanner Identification & Tamper Detection 🕵️‍♂️"
IMG_SIZE = (256, 256)
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title=APP_TITLE, page_icon="🔍", layout="wide")

st.markdown(
    """
    <div style='text-align:center; padding-top:8px;'>
        <h1>🔍 TraceFinder</h1>
        <h4 style='color:#6a7ff7;'>Forensic Scanner Identification <span style='font-size:36px;'>🖨️</span></h4>
        <p style='color:#aaaaff; font-size:20px;'>Instantly analyze any scanned image or PDF.<br><span style='font-size:28px;'>✨</span> See source & confidence results below! <span style='font-size:28px;'>📊</span></p>
    </div>
    """, unsafe_allow_html=True
)

def pdf_bytes_to_bgr(file_bytes: bytes):
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PDF support not available. Add 'pymupdf' to requirements.txt and redeploy.")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def decode_upload_to_bgr(uploaded):
    try:
        uploaded.seek(0)
    except Exception:
        pass
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        bgr = pdf_bytes_to_bgr(raw)
        return bgr, name
    buf = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("❌ Could not decode file")
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
    yy, xx = np.ogrid[:h, :w]; r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, r.max() + 1e-6, K + 1)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i + 1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return np.asarray(feats, dtype=np.float32)

def corr2d(a, b):
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

import joblib, tensorflow as tf
MODEL_PATH  = BASE_DIR / "models" / "scanner_hybrid.keras"
LE_PATH     = BASE_DIR / "models" / "hybrid_label_encoder.pkl"
SCALER_PATH = BASE_DIR / "models" / "hybrid_feat_scaler.pkl"
FPS_PATH    = BASE_DIR / "models" / "scannerfingerprints.pkl"
FP_KEYS     = BASE_DIR / "models" / "fp_keys.npy"

hyb_model  = tf.keras.models.load_model(str(MODEL_PATH))
le_inf     = joblib.load(LE_PATH)
scaler_inf = joblib.load(SCALER_PATH)
with open(FPS_PATH, "rb") as f:
    scanner_fps_inf = pickle.load(f)
fp_keys_inf = np.load(FP_KEYS, allow_pickle=True).tolist()

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
    v_fft  = fft_radial_energy(res, K=6)
    v_lbp  = lbp_hist_safe(res, P=8, R=1.0)
    v = np.array(v_corr + list(v_fft) + list(v_lbp), dtype=np.float32).reshape(1, -1)
    return scaler_inf.transform(v)

def predict_scanner(residual):
    x_img = np.expand_dims(residual, axis=(0, -1))
    x_ft  = make_feats_from_res(residual)
    ps = hyb_model.predict([x_img, x_ft], verbose=0).ravel()
    idx = int(np.argmax(ps))
    return str(le_inf.classes_[idx]), float(ps[idx] * 100.0)

st.write("")
uploaded = st.file_uploader("📁 Upload your scanned page (TIF/TIFF/JPG/PNG/PDF)", type=["tif", "tiff", "jpg", "jpeg", "png", "pdf"], label_visibility="visible")
def safe_show_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_column_width=True, caption="🖼️ Uploaded Image")

if uploaded:
    try:
        bgr, display_name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)
        label, conf = predict_scanner(residual)
        col1, col2 = st.columns([1.1, 1.9], gap="large")
        with col2:
            safe_show_image(bgr)
        with col1:
            st.markdown(
                f"""
                <div style='padding:22px;border-radius:12px;background:#1d2337;border:2px solid #6a7ff7;'>
                    <div style='font-size:23px;color:#7B98EE;'>🖨️ Scanner</div>
                    <div style='font-size:32px;margin-top:10px;font-weight:bold;'>{label}</div>
                    <div style='font-size:16px;color:#fddcff;margin-top:12px;'>🎯 Confidence: <b>{conf:.1f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        st.error("🚨 Inference error")
        st.code(str(e))
else:
    st.info("🧭 Drag-and-drop or select a scanned image/PDF above to analyze.")
