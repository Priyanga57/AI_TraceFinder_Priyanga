# app.py
import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from inference import load_models, predict_scanner, infer_tamper_single

# ---------------- CONFIG ----------------
APP_TITLE = "🎯 TraceFinder - Scanner & Tamper Detector"
IMG_SIZE = (256, 256)

# ---------------- STREAMLIT SETTINGS ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <div style='background: linear-gradient(90deg, #6a11cb, #2575fc); padding: 20px; border-radius: 10px'>
        <h1 style='color:white; margin:0;'>🎨 TraceFinder - Scanner & Tamper Detector</h1>
    </div>
    """, unsafe_allow_html=True
)

# ---------------- LOAD MODELS ONCE ----------------
@st.cache_resource
def load_all_models():
    return load_models()

models = load_all_models()
hyb_model, le_sc, sc_sc, fps, fp_keys = models["scanner"]
sc_tp, clf_tp, THRS_TP = models["tamper_patch"]

# ---------------- IMAGE UTILS ----------------
def decode_upload_to_bgr(uploaded):
    uploaded.seek(0)
    buf = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Cannot decode image")
    return bgr

def load_to_residual_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim==3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    import pywt
    cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA,(cH,cV,cD)),"haar")
    return (gray - den).astype(np.float32)

def safe_show_image(img_bgr):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

# ---------------- FILE UPLOADER ----------------
uploaded = st.file_uploader("Upload scanned page 🖨️", type=["tif","tiff","png","jpg","jpeg"])

if uploaded:
    try:
        bgr = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # SCANNER
        scanner_name, scanner_conf = predict_scanner(residual, hyb_model, le_sc, sc_sc, fps, fp_keys)

        # TAMPER
        tampered, p_img, thr, hits = infer_tamper_single(residual, sc_tp, clf_tp, THRS_TP)
        verdict = "🛑 Tampered" if tampered else "✅ Clean"

        colL, colR = st.columns([1.5,2])
        with colR: safe_show_image(bgr)
        with colL:
            st.markdown(f"""
            <div style='padding:16px;border-radius:12px;background:#1c1f26;color:white;'>
                <h3>🕵️ Scanner Identification</h3>
                <p style='font-size:18px;'>{scanner_name} ({scanner_conf:.1f}% confidence)</p>
                <hr style='border:1px solid #555'>
                <h3>📊 Tamper Detection</h3>
                <p style='font-size:18px;'>{verdict}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error("Error during inference")
        st.code(str(e))
else:
    st.info("Drag and drop a scanned TIF/TIFF/PNG/JPG to detect scanner & tamper 🖨️🕵️")
