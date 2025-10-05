# app/app.py

import os, re, glob, math, json, pickle
from pathlib import Path
import numpy as np
import streamlit as st
import cv2, pywt, tensorflow as tf
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp

# ----------------- App Config -----------------
APP_TITLE = "🕵️‍♂️ TraceFinder - Forensic Scanner Identification"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

# Update this to your model folder
BASE_DIR = Path(r"C:\AI Trace Finder\App\models")  # Your model path
ART_SCN = BASE_DIR
ART_IMG = ART_SCN
ART_PAIR = ART_SCN / "artifacts_tamper_pair"
TAMP_ROOT = ART_SCN / "Tampered images"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"<h2 style='margin-top:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)

# ----------------- Image utils -----------------
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
            raise ImportError("PDF support requires pymupdf in requirements.")
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

# ----------------- Scanner-ID: model-driven lock -----------------
def corr2d(a,b):
    a=a.astype(np.float32).ravel(); b=b.astype(np.float32).ravel()
    a-=a.mean(); b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/d) if d!=0 else 0.0

def make_scanner_feats(res):
    v_corr=[corr2d(res, scanner_fps[k]) for k in fp_keys]
    v_fft =fft_radial_energy(res,K=6).tolist()
    v_lbp =lbp_hist_safe(res,P=8,R=1.0).tolist()
    v=np.array(v_corr+v_fft+v_lbp, dtype=np.float32).reshape(1,-1)
    return sc_sc.transform(v)

def try_scanner_predict(residual):
    if not scanner_ready:
        if scanner_err: st.info(f"Scanner-ID disabled: {scanner_err}")
        return "Unknown", 0.0
    x_img=np.expand_dims(residual,axis=(0,-1))
    x_ft =make_scanner_feats(residual)
    ps=hyb_model.predict([x_img,x_ft],verbose=0).ravel()
    idx=int(np.argmax(ps))
    return str(le_sc.classes_[idx]), float(ps[idx]*100.0)

def must_pick_label_encoder():
    for lep in [ART_SCN/"hybrid_label_encoder.pkl", ART_SCN/"hybrid_label_encoder (1).pkl"]:
        if lep.exists():
            with open(lep,"rb") as f: return pickle.load(f)
    raise FileNotFoundError("hybrid_label_encoder.pkl not found")

def lock_scanner_artifacts_by_required(required_feats: int):
    sc = pickle.load(open(ART_SCN/"hybrid_feat_scaler.pkl","rb"))
    candidates = [
        (ART_SCN/"scanner_fingerprints_14.pkl", ART_SCN/"fp_keys_14.npy", "14"),
        (ART_SCN/"scanner_fingerprints.pkl",    ART_SCN/"fp_keys.npy",    "legacy"),
    ]
    for fps_path, keys_path, tag in candidates:
        if not fps_path.exists() or not keys_path.exists(): continue
        with open(fps_path,"rb") as f: fps = pickle.load(f)
        keys = np.load(keys_path, allow_pickle=True).tolist()
        if len(keys) + 6 + 10 == required_feats and getattr(sc,"n_features_in_",None) == required_feats:
            return dict(fps=fps, keys=keys, scaler=sc, tag=tag)
    raise RuntimeError(f"No fingerprint/keys set matches required feature size {required_feats} and scaler.")

# Load hybrid model
def load_any_hybrid():
    for p in [ART_SCN/"scanner_hybrid_14.keras", ART_SCN/"scanner_hybrid.keras", ART_SCN/"scanner_hybrid.h5", ART_SCN/"scanner_hybrid"]:
        if p.exists(): return tf.keras.models.load_model(str(p)), p.name
    return None, None

hyb_model, model_file = load_any_hybrid()
scanner_ready = hyb_model is not None
scanner_err = None if scanner_ready else "No scanner_hybrid model found"

required_tab_feats = None
if scanner_ready:
    try: required_tab_feats = int(hyb_model.inputs[1].shape[-1])
    except Exception:
        scanner_ready=False
        scanner_err="Hybrid model missing second input; need [image, features] inputs."

if scanner_ready:
    try:
        le_sc = must_pick_label_encoder()
        locked = lock_scanner_artifacts_by_required(required_tab_feats)
        scanner_fps, fp_keys, sc_sc, stack_tag = locked["fps"], locked["keys"], locked["scaler"], locked["tag"]
        st.caption(f"🔒 Scanner stack locked: model={model_file}, tag={stack_tag}, feats={required_tab_feats}")
    except Exception as e:
        scanner_ready=False
        scanner_err=str(e)

# ----------------- Tamper Image -----------------
tamper_image_ok=True
try:
    sc_img=pickle.load(open(ART_IMG/"image_scaler.pkl","rb"))
    clf_img=pickle.load(open(ART_IMG/"image_svm_sig.pkl","rb"))
    thrp = ART_IMG/"image_thresholds.json"
    if not thrp.exists(): thrp = ART_IMG/"image_thresholds"
    THR_IMG=json.load(open(thrp,"r"))
except Exception as e:
    tamper_image_ok=False
    st.info(f"Tamper (image-level) disabled: {e}")

def image_feat_mean(res):
    patches = extract_patches(res, limit=MAX_PATCHES, seed=111)
    feats=[]
    for p in patches:
        lbp=lbp_hist_safe(p,8,1.0); fft6=fft_radial_energy(p,6)
        c1=float(np.std(p)); c2=float(np.mean(np.abs(p - np.mean(p))))
        feats.append(np.concatenate([lbp, fft6, np.array([c1,c2], np.float32)], 0))
    if len(feats)==0: feats=[np.zeros(18, np.float32)]
    return np.mean(np.stack(feats,0), axis=0).reshape(1,-1)

def infer_tamper_image_from_residual(residual, domain):
    if not tamper_image_ok: return 0,0.0,0.5
    x = sc_img.transform(image_feat_mean(residual))
    p = float(clf_img.predict_proba(x)[0,1])
    thr = THR_IMG.get("by_domain",{}).get(domain, THR_IMG.get("global",0.5))
    return int(p>=thr), p, thr

# ----------------- UI -----------------
def safe_show_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try: st.image(rgb, width="stretch")
    except TypeError: st.image(rgb)

uploaded = st.file_uploader("📤 Upload scanned page", type=["tif","tiff","png","jpg","jpeg","pdf"], label_visibility="collapsed")

if uploaded:
    try:
        bgr, display_name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        s_lab, s_conf = try_scanner_predict(residual)

        verdict = "✅ Clean" if s_conf<50 else "⚠️ Tampered"

        colL, colR = st.columns([1.2, 1.8], gap="large")
        with colR: safe_show_image(bgr)
        with colL:
            st.markdown(
                f"""
                <div style='padding:20px;border-radius:12px;background:#111317;border:1px solid #2a2f3a;'>
                    <div style='font-size:16px;color:#9aa4b2;'>🖨️ Scanner</div>
                    <div style='font-size:22px;margin-top:4px;color:#ffffff;'>{s_lab}</div>
                    <div style='font-size:13px;color:#9aa4b2;margin-top:2px;'>{s_conf:.1f}% confidence</div>
                    <hr style='border:none;border-top:1px solid #2a2f3a;margin:12px 0;'>
                    <div style='font-size:16px;color:#9aa4b2;'>🕵️ Tamper verdict</div>
                    <div style='font-size:22px;margin-top:4px;color:#ffffff;'>{verdict}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    except Exception as e:
        import traceback
        st.error("⚠️ Inference error")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag-and-drop a TIF/TIFF/PNG/JPG/JPEG/PDF📌 to analyze.")
