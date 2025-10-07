# App/app.py

import os, json, pickle, numpy as np, cv2, streamlit as st, joblib, tensorflow as tf
from pathlib import Path
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp
from App.inference import (
    make_feats_from_res, corr2d, fft_radial_energy, lbp_hist_safe, predict_from_bytes
)

st.set_page_config(page_title="🔍 AI Trace Finder - Scanner & Tamper Detection", layout="wide")

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
TAMP_PATCH = MODELS / "artifacts_tamper_patch"
TAMP_PAIR = MODELS / "artifacts_tamper_pair"

@st.cache_resource
def load_scanner_model():
    model = tf.keras.models.load_model(str(MODELS / "scanner_hybrid.keras"))
    scaler = joblib.load(MODELS / "hybrid_feat_scaler.pkl")
    with open(MODELS / "scannerfingerprints.pkl", "rb") as f:
        fps = pickle.load(f)
    fp_keys = np.load(MODELS / "fp_keys.npy", allow_pickle=True).tolist()
    le = joblib.load(MODELS / "hybrid_label_encoder.pkl")
    return model, scaler, fps, fp_keys, le

@st.cache_resource
def load_tamper_models():
    try:
        sc_patch = joblib.load(TAMP_PATCH / "patch_scaler.pkl")
        clf_patch = joblib.load(TAMP_PATCH / "patch_svm_sig_calibrated.pkl")
        thr_patch = json.load(open(TAMP_PATCH / "thresholds_patch.json"))
    except Exception as e:
        sc_patch, clf_patch, thr_patch = None, None, None
        st.warning(f"⚠️ Patch-level tamper model not loaded: {e}")
    try:
        sc_pair = joblib.load(TAMP_PAIR / "pair_scaler.pkl")
        clf_pair = joblib.load(TAMP_PAIR / "pair_svm_sig.pkl")
        thr_pair = json.load(open(TAMP_PAIR / "pair_thresholds_topk.json"))
    except Exception as e:
        sc_pair, clf_pair, thr_pair = None, None, None
        st.warning(f"⚠️ Pair-level tamper model not loaded: {e}")
    return sc_patch, clf_patch, thr_patch, sc_pair, clf_pair, thr_pair

def preprocess_image(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return img

def compute_residual(gray):
    import pywt
    cA, (cH, cV, cD) = pywt.dwt2(gray, "haar")
    cH.fill(0); cV.fill(0); cD.fill(
