# app.py
import os, pickle, json, math, re, glob
import numpy as np
import streamlit as st
import cv2, pywt
from skimage.feature import local_binary_pattern as sk_lbp

# ---------------- CONFIG ----------------
ROOT = r"C:\AI Trace Finder\App\models"
TAMP_PATCH = os.path.join(ROOT, "artifacts_tamper_patch")
TAMP_PAIR  = os.path.join(ROOT, "artifacts_tamper_pair")
TAMP_ROOT  = os.path.join(ROOT, "Tampered images")

APP_TITLE = "🖨️🕵️ TraceFinder 2.0 - Forensic Scanner & Tamper Dashboard"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16
TOPK = 0.30
HIT_THR = 0.85
MIN_HITS = 2

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"""
<div style="background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding:20px; border-radius:10px">
<h1 style='color:white; text-align:center'>{APP_TITLE}</h1>
<p style='color:white; text-align:center'>Upload a scanned page to identify the scanner & check for tampering</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Utils ----------------
def file_exists(file_path, desc="file"):
    if not os.path.exists(file_path):
        st.error(f"🛑 {desc} not found: {file_path}")
        return False
    return True

def decode_upload_to_bgr(uploaded):
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(raw, dpi=300)
            pil = pages[0].convert("RGB")
            rgb = np.array(pil)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), name
        except Exception:
            import fitz
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
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim==3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    cA, (cH,cV,cD) = pywt.dwt2(gray,"haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA,(cH,cV,cD)),"haar")
    return (gray-den).astype(np.float32)

def lbp_hist_safe(img,P=8,R=1.0):
    rng=float(np.ptp(img))
    g=np.zeros_like(img,dtype=np.float32) if rng<1e-12 else (img-float(np.min(img)))/(rng+1e-8)
    g8=(g*255.0).astype(np.uint8)
    codes = sk_lbp(g8,P=P,R=R,method="uniform")
    n_bins=P+2
    hist,_=np.histogram(codes,bins=np.arange(n_bins+1),density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img,K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h,w = mag.shape
    cy,cx = h//2,w//2
    yy,xx=np.ogrid[:h,:w]
    r=np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats=[]
    for i in range(K):
        m=(r>=bins[i]) & (r<bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return np.asarray(feats,dtype=np.float32)

def corr2d(a,b):
    a=a.astype(np.float32).ravel(); b=b.astype(np.float32).ravel()
    a-=a.mean(); b-=b.mean()
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/d) if d!=0 else 0.0

def make_scanner_feats(res, fps, fp_keys, sc_sc):
    v_corr=[corr2d(res,fps[k]) for k in fp_keys]
    v_fft=fft_radial_energy(res,K=6)
    v_lbp=lbp_hist_safe(res,P=8,R=1.0)
    v=np.array(v_corr + v_fft.tolist() + v_lbp.tolist(),dtype=np.float32).reshape(1,-1)
    return sc_sc.transform(v)

# ---------------- Load Artifacts ----------------
# Scanner
scanner_files = [
    "scanner_hybrid.keras",
    "hybrid_label_encoder.pkl",
    "hybrid_feat_scaler.pkl",
    "scannerfingerprints.pkl",
    "fp_keys.npy"
]

if all(file_exists(os.path.join(ROOT,f), f) for f in scanner_files):
    import tensorflow as tf
    hyb_model = tf.keras.models.load_model(os.path.join(ROOT,"scanner_hybrid.keras"))
    le_sc = pickle.load(open(os.path.join(ROOT,"hybrid_label_encoder.pkl"),"rb"))
    sc_sc = pickle.load(open(os.path.join(ROOT,"hybrid_feat_scaler.pkl"),"rb"))
    fps   = pickle.load(open(os.path.join(ROOT,"scannerfingerprints.pkl"),"rb"))
    fp_keys = np.load(os.path.join(ROOT,"fp_keys.npy"),allow_pickle=True).tolist()
    scanner_ready=True
else:
    hyb_model=le_sc=sc_sc=fps=fp_keys=None
    scanner_ready=False

# Tamper single
if all(file_exists(os.path.join(TAMP_PATCH,f), f) for f in ["patch_scaler.pkl","patch_svm_sig_calibrated.pkl","thresholds_patch.json"]):
    sc_tp = pickle.load(open(os.path.join(TAMP_PATCH,"patch_scaler.pkl"),"rb"))
    clf_tp= pickle.load(open(os.path.join(TAMP_PATCH,"patch_svm_sig_calibrated.pkl"),"rb"))
    THRS_TP=json.load(open(os.path.join(TAMP_PATCH,"thresholds_patch.json")))
else:
    sc_tp=clf_tp=THRS_TP=None

# Tamper pair
if all(file_exists(os.path.join(TAMP_PAIR,f), f) for f in ["pair_scaler.pkl","pair_svm_sig.pkl","pair_thresholds_topk.json"]):
    sc_pair = pickle.load(open(os.path.join(TAMP_PAIR,"pair_scaler.pkl"),"rb"))
    pair_clf = pickle.load(open(os.path.join(TAMP_PAIR,"pair_svm_sig.pkl"),"rb"))
    THR_PAIR = json.load(open(os.path.join(TAMP_PAIR,"pair_thresholds_topk.json")))
else:
    sc_pair=pair_clf=THR_PAIR=None

# ---------------- UI ----------------
uploaded = st.file_uploader("📤 Upload scanned page", type=["tif","tiff","png","jpg","jpeg","pdf"])
if uploaded:
    try:
        bgr, display_name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # Scanner ID
        s_lab, s_conf="Unknown",0.0
        if scanner_ready:
            x_img = np.expand_dims(residual,axis=(0,-1))
            x_ft = make_scanner_feats(residual,fps,fp_keys,sc_sc)
            ps = hyb_model.predict([x_img,x_ft],verbose=0).ravel()
            if not np.isnan(ps).all():
                s_idx=int(np.argmax(ps))
                s_lab=le_sc.classes_[s_idx]
                s_conf=float(ps[s_idx]*100.0)

        # Tamper verdict (simplified single-image)
        verdict = "Clean" if (clf_tp is None or sc_tp is None) else "Tampered"  # placeholder

        # Display
        colL,colR = st.columns([1,2],gap="large")
        with colR:
            st.image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),use_container_width=True)
        with colL:
            st.markdown(f"""
            <div style='padding:16px;border-radius:12px;background:linear-gradient(135deg,#0f2027,#203a43);color:white;'>
                <h2>🖨️ Scanner Identification</h2>
                <h1 style='color:#ffda79'>{s_lab}</h1>
                <p>Confidence: <b>{s_conf:.1f}%</b> {"🔹"*int(s_conf//10)}</p>
                <div style='background:#444;border-radius:8px;overflow:hidden;height:12px;width:100%;margin-bottom:10px;'>
                    <div style='width:{s_conf}%;background:#ffda79;height:12px;'></div>
                </div>
                <h2>🕵️ Tamper Verdict</h2>
                <h1 style='color:#70ff70'>{verdict}</h1>
            </div>
            """,unsafe_allow_html=True)
    except Exception as e:
        import traceback
        st.error("⚠️ Inference error")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag & drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze.")
