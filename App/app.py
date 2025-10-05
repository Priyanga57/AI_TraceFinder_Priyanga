# app.py

import os, pickle, json, math, re, glob
from pathlib import Path
import numpy as np
import streamlit as st
import cv2, pywt
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp
import tensorflow as tf

# ---------------- CONFIG ----------------
APP_TITLE = "🎯 TraceFinder - Scanner & Tamper Detector"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16
TOPK = 0.3
HIT_THR = 0.85
MIN_HITS = 2

# ---------------- PATHS ----------------
BASE_DIR = Path(r"C:\AI Trace Finder\App\models")  # your folder
ART_SCN = BASE_DIR
ART_TP = BASE_DIR / "artifacts_tamper_patch"
ART_PAIR = BASE_DIR / "artifacts_tamper_pair"

# ---------------- UTILITIES ----------------
def must_exist(p: Path, kind="file"):
    if kind == "file" and not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p}")
    if kind == "dir" and not p.is_dir():
        raise FileNotFoundError(f"Missing required folder: {p}")
    return p

# ---------------- IMAGE LOAD ----------------
def decode_upload_to_bgr(uploaded):
    uploaded.seek(0)
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), name
    buf = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if bgr is None: raise ValueError("Cannot decode file")
    return bgr, name

def load_to_residual_from_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim==3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA,(cH,cV,cD)),"haar")
    return (gray - den).astype(np.float32)

def extract_patches(res, patch=PATCH, stride=STRIDE, limit=MAX_PATCHES, seed=42):
    H, W = res.shape
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    coords = [(y,x) for y in ys for x in xs]
    rng = np.random.RandomState(seed)
    rng.shuffle(coords)
    coords = coords[:min(limit, len(coords))]
    return [res[y:y+patch, x:x+patch] for y,x in coords]

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng<1e-12 else (img - float(np.min(img))) / (rng+1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P+2
    hist,_ = np.histogram(codes, bins=np.arange(n_bins+1), density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h,:w]; r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats = [float(mag[(r>=bins[i])&(r<bins[i+1])].mean() if np.any((r>=bins[i])&(r<bins[i+1])) else 0.0) for i in range(K)]
    return np.asarray(feats, dtype=np.float32)

def residual_stats(img):
    return np.asarray([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))], dtype=np.float32)

def fft_resample_feats(img):
    f = np.fft.fftshift(np.fft.fft2(img)); mag = np.abs(f)
    h,w = mag.shape; cy,cx = h//2,w//2
    yy,xx = np.ogrid[:h,:w]; r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = r.max()+1e-6
    b1 = (r>=0.25*rmax)&(r<0.35*rmax)
    b2 = (r>=0.35*rmax)&(r<0.5*rmax)
    e1 = float(mag[b1].mean() if b1.any() else 0.0)
    e2 = float(mag[b2].mean() if b2.any() else 0.0)
    ratio = float(e2/(e1+1e-8))
    return np.asarray([e1,e2,ratio],dtype=np.float32)

def make_feat_vector(img_patch):
    lbp = lbp_hist_safe(img_patch)
    fft6 = fft_radial_energy(img_patch)
    res3 = residual_stats(img_patch)
    rsp3 = fft_resample_feats(img_patch)
    return np.concatenate([lbp,fft6,res3,rsp3],axis=0)

# ---------------- SCANNER MODEL ----------------
hyb_model = tf.keras.models.load_model(str(must_exist(ART_SCN/"scanner_hybrid.keras")))
with open(must_exist(ART_SCN/"hybrid_label_encoder.pkl"), "rb") as f: le_sc = pickle.load(f)
with open(must_exist(ART_SCN/"hybrid_feat_scaler.pkl"), "rb") as f: sc_sc = pickle.load(f)
with open(must_exist(ART_SCN/"scannerfingerprints.pkl"), "rb") as f: fps = pickle.load(f)
fp_keys = np.load(must_exist(ART_SCN/"fp_keys.npy"), allow_pickle=True).tolist()

def corr2d(a,b):
    a=a.ravel();b=b.ravel()
    a-=a.mean(); b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/d) if d!=0 else 0.0

def make_scanner_feats(res):
    v_corr=[corr2d(res,fps[k]) for k in fp_keys]
    v_fft=fft_radial_energy(res).tolist()
    v_lbp=lbp_hist_safe(res).tolist()
    v=np.array(v_corr+v_fft+v_lbp,dtype=np.float32).reshape(1,-1)
    return sc_sc.transform(v)

def try_scanner_predict(res):
    x_img = np.expand_dims(res,axis=(0,-1))
    x_ft = make_scanner_feats(res)
    preds = hyb_model.predict([x_img, x_ft], verbose=0)
    ps = preds.ravel(); s_idx=int(np.argmax(ps))
    return str(le_sc.classes_[s_idx]), float(ps[s_idx]*100.0)

# ---------------- SINGLE PATCH TAMPER ----------------
with open(must_exist(ART_TP/"patch_scaler.pkl"), "rb") as f: sc_tp = pickle.load(f)
with open(must_exist(ART_TP/"patch_svm_sig_calibrated.pkl"), "rb") as f: clf_tp = pickle.load(f)
with open(must_exist(ART_TP/"thresholds_patch.json"), "r") as f: THRS_TP = json.load(f)

def choose_thr_single(): return THRS_TP.get("global",0.5)

def image_score_topk(patch_probs, frac=TOPK):
    n=len(patch_probs); k=max(1,int(math.ceil(frac*n)))
    top=np.sort(np.asarray(patch_probs))[-k:]
    return float(np.mean(top))

def infer_tamper_single(res):
    patches = extract_patches(res, limit=MAX_PATCHES)
    feats = np.stack([make_feat_vector(p) for p in patches],0)
    feats = sc_tp.transform(feats)
    p_patch = clf_tp.predict_proba(feats)[:,1]
    p_img = image_score_topk(p_patch)
    thr = choose_thr_single()
    hits = int((p_patch>=HIT_THR).sum())
    tampered = int((p_img>=thr) and (hits>=MIN_HITS))
    return bool(tampered), p_img, thr, hits

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <div style='background: linear-gradient(90deg, #6a11cb, #2575fc); padding: 20px; border-radius: 10px'>
        <h1 style='color:white; margin:0;'>🎨 TraceFinder - Scanner & Tamper Detector</h1>
    </div>
    """, unsafe_allow_html=True
)

uploaded = st.file_uploader("Upload scanned page 🖨️", type=["tif","tiff","png","jpg","jpeg","pdf"])

def safe_show_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_container_width=True)

if uploaded:
    try:
        bgr, name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # SCANNER
        scanner_name, scanner_conf = try_scanner_predict(residual)

        # TAMPER
        tampered, p_img, thr, hits = infer_tamper_single(residual)
        verdict = "🛑 Tampered" if tampered else "✅ Clean"

        colL, colR = st.columns([1.5,2])
        with colR: safe_show_image(bgr)
        with colL:
            st.markdown(f"""
            <div style='padding:16px;border-radius:12px;background:#1c1f26;color:white;'>
                <h3>🕵️ Scanner Identification</h3>
                <p style='font-size:18px;'>{scanner_name} ({scanner_conf:.1f}% confidence)</p>
                <div style='background:#333;border-radius:8px; height:15px; width:100%;'>
                    <div style='background:#00ff99; height:15px; width:{scanner_conf}%; border-radius:8px'></div>
                </div>
                <hr style='border:1px solid #555'>
                <h3>📊 Tamper Detection</h3>
                <p style='font-size:18px;'>{verdict}</p>
                <div style='background:#333;border-radius:8px; height:15px; width:100%;'>
                    <div style='background:#ff4d4d; height:15px; width:{p_img*100:.1f}%; border-radius:8px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error("Error during inference")
        st.code(str(e))
else:
    st.info("Drag and drop a scanned TIF/TIFF/PNG/JPG/PDF to detect scanner & tamper 🖨️🕵️")
