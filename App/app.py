# app.py

import os, re, glob, math, json, pickle
from pathlib import Path
import numpy as np
import streamlit as st
import cv2, pywt, tensorflow as tf
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp

# ----------------- CONFIG -----------------
APP_TITLE = "🕵️‍♂️ TraceFinder - Forensic Scanner ID"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

# Your models directory
ROOT = Path(r"C:\AI Trace Finder\App\models")
ART_SCN = ROOT
ART_IMG = ROOT
ART_PAIR = ROOT / "artifacts_tamper_pair"
TAMP_ROOT = ROOT / "Tampered images"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    f"""
    <div style='
        padding:20px;
        border-radius:12px;
        background:linear-gradient(135deg,#1f1f2e,#2a2f3a);
        color:white;
        text-align:center;
    '>
        <h1 style='margin:0'>{APP_TITLE}</h1>
        <p style='color:#9aa4b2;margin-top:5px;'>Upload a scanned page to detect scanner and tampering</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- IMAGE UTILITIES -----------------
def decode_upload_to_bgr(uploaded):
    try: uploaded.seek(0)
    except Exception: pass
    raw = uploaded.read(); name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            page = doc.load_page(0); pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), name
        except Exception:
            raise ImportError("PDF support requires pymupdf.")
    buf = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None: raise ValueError("Could not decode file")
    return bgr, name

def load_to_residual_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    cA,(cH,cV,cD)=pywt.dwt2(gray,"haar"); cH.fill(0); cV.fill(0); cD.fill(0)
    den=pywt.idwt2((cA,(cH,cV,cD)),"haar")
    return (gray - den).astype(np.float32)

def extract_patches(res, patch=PATCH, stride=STRIDE, limit=MAX_PATCHES, seed=42):
    H,W=res.shape
    ys=list(range(0,H-patch+1,stride)); xs=list(range(0,W-patch+1,stride))
    coords=[(y,x) for y in ys for x in xs]
    rng=np.random.RandomState(seed); rng.shuffle(coords)
    coords=coords[:min(limit,len(coords))]
    return [res[y:y+patch, x:x+patch] for y,x in coords]

def lbp_hist_safe(img, P=8, R=1.0):
    rng=float(np.ptp(img))
    g=np.zeros_like(img,dtype=np.float32) if rng<1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8=(g*255.0).astype(np.uint8)
    codes=sk_lbp(g8,P=P,R=R,method="uniform")
    n_bins=P+2
    hist,_=np.histogram(codes,bins=np.arange(n_bins+1),density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f=np.fft.fftshift(np.fft.fft2(img)); mag=np.abs(f)
    h,w=mag.shape; cy,cx=h//2,w//2
    yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy - cy)**2 + (xx - cx)**2)
    bins=np.linspace(0, r.max()+1e-6, K+1)
    feats=[]
    for i in range(K):
        m=(r>=bins[i])&(r<bins[i+1]); feats.append(float(mag[m].mean() if m.any() else 0.0))
    return np.asarray(feats, dtype=np.float32)

# ----------------- SCANNER MODEL -----------------
def load_any_hybrid():
    for p in [ART_SCN/"scanner_hybrid_14.keras", ART_SCN/"scanner_hybrid.keras", ART_SCN/"scanner_hybrid.h5", ART_SCN/"scanner_hybrid"]:
        if p.exists(): return tf.keras.models.load_model(str(p)), p.name
    return None, None

hyb_model, model_file = load_any_hybrid()
scanner_ready = hyb_model is not None
scanner_err = None if scanner_ready else "❌ No scanner_hybrid model found."

required_tab_feats = None
if scanner_ready:
    try:
        required_tab_feats = int(hyb_model.inputs[1].shape[-1])
    except Exception:
        scanner_ready = False
        scanner_err = "Hybrid model missing second input; need [image, features] inputs."

def must_pick_label_encoder():
    for lep in [ART_SCN/"hybrid_label_encoder.pkl", ART_SCN/"hybrid_label_encoder (1).pkl"]:
        if lep.exists():
            with open(lep,"rb") as f: return pickle.load(f)
    raise FileNotFoundError("❌ hybrid_label_encoder.pkl not found")

# ----------------- ADDITIONAL FUNCTIONS OMITTED FOR BREVITY -----------------
# (Include all the functions from your original code: lock_scanner_artifacts_by_required, corr2d, make_scanner_feats,
# try_scanner_predict, image-level tamper, paired inference, infer_domain_and_type_from_path_or_name, etc.)

# ----------------- STREAMLIT UPLOAD & DISPLAY -----------------
uploaded = st.file_uploader("📤 Upload scanned page", type=["tif","tiff","png","jpg","jpeg","pdf"], label_visibility="collapsed")

if uploaded:
    try:
        bgr, display_name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # Scanner prediction
        s_lab, s_conf = try_scanner_predict(residual)

        # Domain/type detection
        domain, typ_hint = infer_domain_and_type_from_path_or_name(display_name)
        pid = re.search(r"(s\d+_\d+)", display_name); pid = pid.group(1) if pid else None

        # Paired inference if original exists
        if pid and (pid in (orig_map or {})):
            domain="orig_pdf_tif"; typ_hint=None
            is_t, p_img, thr_used, hits = paired_infer_type_aware(orig_map[pid], residual, typ_hint)
        else:
            is_t, p_img, thr_used = infer_tamper_image_from_residual(residual, domain)
            hits = 0

        verdict = "⚠️ Tampered" if is_t else "✅ Clean"

        # Display
        colL, colR = st.columns([1.2, 1.8], gap="large")
        with colR: st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width="stretch")
        with colL:
            st.markdown(
                f"""
                <div style='padding:16px;border-radius:12px;background:linear-gradient(135deg,#1f1f2e,#2a2f3a);color:white;'>
                    <h3>🖨️ Scanner:</h3> <b>{s_lab}</b> ({s_conf:.1f}% confidence)<br><br>
                    <h3>🛡️ Tamper verdict:</h3> <b>{verdict}</b><br>
                    <p style='font-size:12px;color:#9aa4b2;'>p={p_img:.3f} · thr={thr_used:.3f} · domain={domain} · hits={hits}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        import traceback
        st.error("❌ Inference error")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag-and-drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze.")
