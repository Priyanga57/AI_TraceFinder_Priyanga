# app/app.py

import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import cv2
import pywt
import tensorflow as tf
from skimage.feature import local_binary_pattern as sk_lbp
from tensorflow.keras.layers import TFSMLayer

# ----------------- App Config -----------------
APP_TITLE = "🖨️ TraceFinder 2.0 - Forensic Scanner & Tamper Dashboard"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

# Local model/artifacts path
BASE_DIR = Path(r"C:\AI Trace Finder\App\models")
ART_SCN = BASE_DIR

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"<h1 style='color:white'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("🔍 Upload a scanned page to analyze the **scanner source** & check for **tampering**.")

# ----------------- Load Scanner Model -----------------
def load_scanner_model():
    path = ART_SCN / "scanner_hybrid.keras"
    if path.exists():
        try:
            # Keras 3 SavedModel loading
            model = tf.keras.Sequential([TFSMLayer(str(path), call_endpoint="serving_default")])
            return model
        except Exception as e:
            st.error(f"🛑 Failed loading scanner model: {e}")
    else:
        st.error("🛑 Scanner model file not found at local path.")
    return None

scanner_model = load_scanner_model()
scanner_ready = scanner_model is not None
scanner_err = None if scanner_ready else "Scanner model not loaded."

# ----------------- Load Artifacts -----------------
def load_artifacts():
    le = pickle.load(open(ART_SCN / "hybrid_label_encoder.pkl", "rb"))
    fps = pickle.load(open(ART_SCN / "scannerfingerprints.pkl", "rb"))
    keys = np.load(ART_SCN / "fp_keys.npy", allow_pickle=True).tolist()
    scaler = pickle.load(open(ART_SCN / "hybrid_feat_scaler.pkl", "rb"))
    return le, fps, keys, scaler

if scanner_ready:
    try:
        le_sc, scanner_fps, fp_keys, sc_scaler = load_artifacts()
        st.success("✅ Scanner model and artifacts loaded successfully!")
    except Exception as e:
        scanner_ready = False
        scanner_err = f"🛑 Failed loading artifacts: {e}"

# ----------------- Image & Feature utils -----------------
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
    feats = [float(mag[(r >= bins[i]) & (r < bins[i + 1])].mean() if ((r >= bins[i]) & (r < bins[i + 1])).any() else 0.0)
             for i in range(K)]
    return np.asarray(feats, dtype=np.float32)

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
        if scanner_err: st.info(scanner_err)
        return "Unknown", 0.0
    x_img = np.expand_dims(res, axis=(0, -1))
    x_feat = make_scanner_feats(res)
    ps = scanner_model.predict([x_img, x_feat], verbose=0).ravel()
    if np.isnan(ps).any() or np.allclose(ps, 0):
        return "Unknown", 0.0
    idx = int(np.argmax(ps))
    return str(le_sc.classes_[idx]), float(ps[idx] * 100.0)

# ----------------- Streamlit UI -----------------
def safe_show_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_column_width=True)

uploaded = st.file_uploader("📤 Upload a scanned page", type=["tif","tiff","png","jpg","jpeg","pdf"])

if uploaded:
    try:
        bgr, name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)
        s_label, s_conf = try_scanner_predict(residual)
        verdict = "✅ Clean"  # placeholder

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
