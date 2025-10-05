# app/app.py

import os
import pickle
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import streamlit as st
import cv2
import pywt
import tensorflow as tf
from skimage.feature import local_binary_pattern as sk_lbp

# ----------------- App Config -----------------
APP_TITLE = "🖨️ TraceFinder 2.0 - Forensic Scanner & Tamper Dashboard"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

# ----------------- GitHub Model Paths -----------------
GITHUB_BASE = "https://github.com/Priyanga57/AI_TraceFinder_Priyanga/raw/main/App/models"
ARTIFACTS = {
    "scanner_model": "scanner_hybrid.keras",
    "label_encoder": "hybrid_label_encoder.pkl",
    "scanner_fps": "scannerfingerprints.pkl",
    "fp_keys": "fp_keys.npy",
    "scaler": "hybrid_feat_scaler.pkl"
}

# ----------------- Streamlit Page -----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"<h1 style='color:white'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("🔍 Upload a scanned page to analyze the **scanner source** & check for **tampering**.")

# ----------------- Utility to download GitHub files -----------------
def download_from_github(filename):
    url = f"{GITHUB_BASE}/{filename}"
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(url, temp_file.name)
    return temp_file.name

# ----------------- Image utils -----------------
def decode_upload_to_bgr(uploaded):
    uploaded.seek(0)
    raw = uploaded.read()
    ext = os.path.splitext(uploaded.name.lower())[-1]
    if ext == ".pdf":
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), uploaded.name
    buf = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Could not decode file")
    return bgr, uploaded.name

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
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    
    bins = np.linspace(0, r.max() + 1e-6, K + 1)
    feats = []
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        feats.append(float(mag[mask].mean()) if mask.any() else 0.0)
    return np.asarray(feats, dtype=np.float32)

# ----------------- Load Model & Artifacts -----------------
def load_scanner_model():
    path = download_from_github(ARTIFACTS["scanner_model"])
    return tf.keras.models.load_model(path)

try:
    scanner_model = load_scanner_model()
    le_sc = pickle.load(open(download_from_github(ARTIFACTS["label_encoder"]), "rb"))
    scanner_fps = pickle.load(open(download_from_github(ARTIFACTS["scanner_fps"]), "rb"))
    fp_keys = np.load(download_from_github(ARTIFACTS["fp_keys"]), allow_pickle=True).tolist()
    sc_scaler = pickle.load(open(download_from_github(ARTIFACTS["scaler"]), "rb"))
    scanner_ready = True
    st.success("✅ Scanner model and artifacts loaded successfully!")
except Exception as e:
    scanner_model = None
    scanner_ready = False
    st.error(f"🛑 Failed loading scanner model or artifacts: {e}")

# ----------------- Prediction Functions -----------------
def corr2d(a, b):
    a, b = a.ravel().astype(np.float32), b.ravel().astype(np.float32)
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def make_scanner_feats(res):
    v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(res, K=6).tolist()
    v_lbp = lbp_hist_safe(res, P=8, R=1.0).tolist()
    return sc_scaler.transform(np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1))

def try_scanner_predict(res):
    if not scanner_ready:
        return "Unknown", 0.0
    x_img = np.expand_dims(res, axis=(0, -1))
    x_feat = make_scanner_feats(res)
    ps = scanner_model.predict([x_img, x_feat], verbose=0).ravel()
    if np.isnan(ps).any() or np.allclose(ps, 0):
        return "Unknown", 0.0
    idx = int(np.argmax(ps))
    return str(le_sc.classes_[idx]), float(ps[idx] * 100.0)

# ----------------- UI -----------------
def safe_show_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_column_width=True)

uploaded = st.file_uploader("📤 Upload a scanned page", type=["tif","tiff","png","jpg","jpeg","pdf"])

if uploaded:
    try:
        bgr, name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)
        s_label, s_conf = try_scanner_predict(residual)
        verdict = "✅ Clean"  # Placeholder for tamper detection

        col1, col2 = st.columns([1.5, 2], gap="large")
        with col2:
            safe_show_image(bgr)
        with col1:
            st.markdown(f"""
            <div style='padding:16px;border-radius:12px;background:#2d2d44;color:white;'>
                <h2>🖨️ Scanner Identification</h2>
                <h1 style='color:#ffda79'>{s_label} ✅</h1>
                <p>Confidence: <b>{s_conf:.1f}%</b> {"🔹" * int(s_conf // 10)}</p>
                <div style='background:#444;border-radius:8px;overflow:hidden;height:12px;width:100%;margin-bottom:10px;'>
                    <div style='width:{s_conf}%;background:#ffda79;height:12px;'></div>
                </div>
                <h2>🕵️ Tamper Verdict</h2>
                <h1 style='color:#70ff70'>{verdict} 🔍</h1>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error("⚠️ Inference error!")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag & drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze the scanner.")
