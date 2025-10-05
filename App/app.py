# app.py

import os, pickle, json, math, re, glob
import numpy as np
import streamlit as st
import cv2, pywt
from skimage.feature import local_binary_pattern as sk_lbp

# ---------------- CONFIG ----------------
ROOT = r"C:\AI Trace Finder\App\models"
APP_TITLE = "🖨️🕵️ TraceFinder 2.0 - Forensic Scanner & Tamper Dashboard"
IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16
TOPK = 0.30
HIT_THR = 0.85
MIN_HITS = 2

# ---------------- Page Setup ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #1f1c2c, #928dab);
            color: white;
        }}
        .card {{
            padding: 16px;
            border-radius: 12px;
            background: rgba(0,0,0,0.4);
            margin-bottom: 16px;
        }}
        .progress-bar {{
            background: #ffda79;
            height: 12px;
            border-radius: 6px;
        }}
    </style>
""", unsafe_allow_html=True)
st.markdown(f"<h1 style='color:white'>{APP_TITLE}</h1>", unsafe_allow_html=True)

# ---------------- File Paths ----------------
ART_SCN = ROOT
ART_TP = os.path.join(ROOT, "artifacts_tamper_patch")
ART_PAIR = os.path.join(ROOT, "artifacts_tamper_pair")

# ---------------- Utils ----------------
def decode_upload_to_bgr(uploaded):
    raw = uploaded.read()
    name = uploaded.name
    ext = os.path.splitext(name.lower())[-1]
    if ext == ".pdf":
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(raw, dpi=300, fmt="png")
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
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (gray - den).astype(np.float32)

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g*255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P+2
    hist,_ = np.histogram(codes, bins=np.arange(n_bins+1), density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h,w = mag.shape
    cy,cx = h//2,w//2
    yy,xx = np.ogrid[:h,:w]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats=[]
    for i in range(K):
        mask = (r >= bins[i]) & (r < bins[i+1])
        feats.append(float(mag[mask].mean() if mask.any() else 0.0))
    return np.asarray(feats,dtype=np.float32)

def residual_stats(img):
    return np.asarray([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))],dtype=np.float32)

def fft_resample_feats(img):
    f = np.fft.fftshift(np.fft.fft2(img)); mag=np.abs(f)
    h,w=mag.shape; cy,cx=h//2,w//2
    yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy-cy)**2 + (xx-cx)**2)
    rmax = r.max()+1e-6
    b1 = (r>=0.25*rmax)&(r<0.35*rmax)
    b2 = (r>=0.35*rmax)&(r<0.50*rmax)
    e1 = float(mag[b1].mean() if b1.any() else 0.0)
    e2 = float(mag[b2].mean() if b2.any() else 0.0)
    ratio = float(e2/(e1+1e-8))
    return np.asarray([e1,e2,ratio],dtype=np.float32)

def make_feat_vector(img_patch):
    lbp = lbp_hist_safe(img_patch,8,1.0)
    fft6 = fft_radial_energy(img_patch,6)
    res3 = residual_stats(img_patch)
    rsp3 = fft_resample_feats(img_patch)
    return np.concatenate([lbp,fft6,res3,rsp3],axis=0)

def corr2d(a,b):
    a,b=a.astype(np.float32).ravel(),b.astype(np.float32).ravel()
    a-=a.mean(); b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/d) if d!=0 else 0.0

def make_scanner_feats(res):
    v_corr=[corr2d(res,fps[k]) for k in fp_keys]
    v_fft=fft_radial_energy(res,6)
    v_lbp=lbp_hist_safe(res,8,1.0)
    v = np.array(v_corr + v_fft.tolist() + v_lbp.tolist(),dtype=np.float32).reshape(1,-1)
    return sc_sc.transform(v)

# ---------------- Load Models ----------------
import tensorflow as tf

# Scanner model
hyb_model=None
cand=[os.path.join(ART_SCN,"scanner_hybrid.keras")]
found=next((p for p in cand if os.path.exists(p)),None)
if found: hyb_model=tf.keras.models.load_model(found)

# Artifacts
with open(os.path.join(ART_SCN,"hybrid_label_encoder.pkl"),"rb") as f: le_sc=pickle.load(f)
with open(os.path.join(ART_SCN,"hybrid_feat_scaler.pkl"),"rb") as f: sc_sc=pickle.load(f)
with open(os.path.join(ART_SCN,"scannerfingerprints.pkl"),"rb") as f: fps=pickle.load(f)
fp_keys=np.load(os.path.join(ART_SCN,"fp_keys.npy"),allow_pickle=True).tolist()

# Tamper single
with open(os.path.join(ART_TP,"patch_scaler.pkl"),"rb") as f: sc_tp=pickle.load(f)
with open(os.path.join(ART_TP,"patch_svm_sig_calibrated.pkl"),"rb") as f: clf_tp=pickle.load(f)
with open(os.path.join(ART_TP,"thresholds_patch.json"),"r") as f: THRS_TP=json.load(f)

# Paired tamper
with open(os.path.join(ART_PAIR,"pair_scaler.pkl"),"rb") as f: sc_pair=pickle.load(f)
with open(os.path.join(ART_PAIR,"pair_svm_sig.pkl"),"rb") as f: pair_clf=pickle.load(f)
with open(os.path.join(ART_PAIR,"pair_thresholds_topk.json"),"r") as f: THR_PAIR=json.load(f)

# ---------------- Patch Extract ----------------
def extract_patches(res,patch=PATCH,stride=STRIDE,limit=MAX_PATCHES,seed=42):
    H,W=res.shape
    ys=list(range(0,H-patch+1,stride))
    xs=list(range(0,W-patch+1,stride))
    coords=[(y,x) for y in ys for x in xs]
    rng=np.random.RandomState(seed)
    rng.shuffle(coords)
    coords=coords[:min(limit,len(coords))]
    return [res[y:y+patch,x:x+patch] for y,x in coords]

# ---------------- Single Tamper Inference ----------------
def choose_thr_single(domain,typ):
    if "by_type" in THRS_TP and typ in THRS_TP["by_type"]: return THRS_TP["by_type"][typ]
    if "by_domain" in THRS_TP and domain in THRS_TP["by_domain"]: return THRS_TP["by_domain"][domain]
    return THRS_TP.get("global",0.5)

def image_score_topk(patch_probs,frac=TOPK):
    n=len(patch_probs); k=max(1,int(math.ceil(frac*n)))
    top=np.sort(np.asarray(patch_probs))[-k:]
    return float(np.mean(top))

def infer_tamper_single_from_residual(residual,domain,typ_hint):
    patches = extract_patches(residual,limit=MAX_PATCHES,seed=123)
    feats=np.stack([make_feat_vector(p) for p in patches],0)
    feats=sc_tp.transform(feats)
    p_patch=clf_tp.predict_proba(feats)[:,1]
    p_img=image_score_topk(p_patch,frac=TOPK)
    thr=choose_thr_single(domain,(typ_hint or "unknown"))
    if domain=="orig_pdf_tif": thr=min(1.0,thr+0.03)
    hits=int((p_patch>=HIT_THR).sum())
    tampered=int((p_img>=thr) and (hits>=MIN_HITS))
    if domain=="orig_pdf_tif": tampered=0
    return tampered,p_img,thr,hits

# ---------------- UI ----------------
uploaded = st.file_uploader("📤 Upload scanned page",type=["tif","tiff","png","jpg","jpeg","pdf"])
if uploaded:
    try:
        bgr, name = decode_upload_to_bgr(uploaded)
        residual = load_to_residual_from_bgr(bgr)

        # Scanner ID
        s_lab,s_conf="Unknown",0.0
        try:
            if hyb_model is not None:
                x_img=np.expand_dims(residual,axis=(0,-1))
                x_ft=make_scanner_feats(residual)
                ps=hyb_model.predict([x_img,x_ft],verbose=0).ravel()
                idx=int(np.argmax(ps))
                s_lab=le_sc.classes_[idx]
                s_conf=float(ps[idx]*100.0)
        except Exception:
            pass

        # Tamper inference
        domain="orig_pdf_tif"
        typ_hint=None
        is_t,p_img,thr_used,hits=infer_tamper_single_from_residual(residual,domain,typ_hint)
        verdict="Tampered" if is_t else "Clean"

        # Display
        colL,colR = st.columns([1.2,1.8],gap="large")
        with colR: st.image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),use_column_width=True)
        with colL:
            st.markdown(f"""
                <div class='card'>
                    <div style='font-size:18px;'>🖨️ <b>Scanner Identification</b></div>
                    <div style='font-size:22px;margin-top:4px;'>{s_lab}</div>
                    <div style='font-size:14px;color:#ccc;margin-top:2px;'>{s_conf:.1f}% confidence 📊</div>
                    <div style='background:#444;border-radius:6px;overflow:hidden;height:12px;width:100%;margin-top:6px;'>
                        <div class='progress-bar' style='width:{s_conf}%;'></div>
                    </div>
                    <hr style='border:none;border-top:1px solid #555;margin:12px 0;'>
                    <div style='font-size:18px;'>🕵️ <b>Tamper Verdict</b></div>
                    <div style='font-size:22px;margin-top:4px;'>{verdict}</div>
                </div>
            """,unsafe_allow_html=True)

    except Exception as e:
        import traceback
        st.error("⚠️ Inference error")
        st.code(traceback.format_exc())
else:
    st.info("📂 Drag & drop a TIF/TIFF/PNG/JPG/JPEG/PDF to analyze.")
