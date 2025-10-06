# App/app.py
import os, json, pickle, numpy as np, cv2, streamlit as st, joblib, tensorflow as tf
from pathlib import Path
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp

from inference import make_feats_from_res, corr2d, fft_radial_energy, lbp_hist_safe, residualstats, fftresamplefeats  # Updated imports

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
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    return (gray - den).astype(np.float32)

def predict_scanner(residual, model, scaler, fps, fp_keys, le):
    v_corr = [corr2d(residual, fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(residual, 6)
    v_lbp = lbp_hist_safe(residual, 8, 1.0)
    v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    v_scaled = scaler.transform(v)
    x_img = np.expand_dims(residual, axis=(0, -1))
    preds = model.predict([x_img, v_scaled], verbose=0).ravel()
    idx = int(np.argmax(preds))
    return str(le.classes_[idx]), float(preds[idx] * 100.0)

def predict_tamper_patch(residual, sc_patch, clf_patch, thr_patch):
    if sc_patch is None or clf_patch is None:
        return "❌ Not available", 0.0
    patch_feats = []
    for y in range(0, residual.shape[0] - 64, 64):
        for x in range(0, residual.shape[1] - 64, 64):
            p = residual[y:y+64, x:x+64]
            lbp = lbp_hist_safe(p, 8, 1.0)
            fft6 = fft_radial_energy(p, 6)
            res3 = residualstats(p)
            rsp3 = fftresamplefeats(p)
            feat = np.concatenate([lbp, fft6, res3, rsp3], axis=0)
            patch_feats.append(feat)
    if not patch_feats:
        return "❌ No patches", 0.0
    X = np.array(patch_feats, np.float32)

    expected_len = sc_patch.scale_.shape[0] if hasattr(sc_patch, 'scale_') else None
    if expected_len is not None and X.shape[1] != expected_len:
        raise ValueError(f"Feature dimension mismatch: scaler expects {expected_len} features but input has {X.shape[1]} features.")

    Xs = sc_patch.transform(X)
    p = clf_patch.predict_proba(Xs)[:, 1]
    prob = float(np.mean(p))
    thr = thr_patch.get("global", 0.5)
    verdict = "🔴 Tampered" if prob >= thr else "🟢 Clean"
    return verdict, prob

st.title("🔍 AI Trace Finder")
st.markdown("### 🧠 **Scanner Identification & Tamper Detection Tool**")
st.markdown("Upload a scanned page (TIF/PNG/JPG/PDF) to identify its **scanner source** and check for **tampering.**")
uploaded = st.file_uploader("📁 Upload Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("⚠️ Could not decode image.")
    else:
        gray = preprocess_image(img)
        residual = compute_residual(gray)
        model, scaler, fps, fp_keys, le = load_scanner_model()
        sc_patch, clf_patch, thr_patch, sc_pair, clf_pair, thr_pair = load_tamper_models()
        with st.spinner("🔍 Identifying scanner..."):
            label, conf = predict_scanner(residual, model, scaler, fps, fp_keys, le)
        with st.spinner("🧪 Checking tamper status..."):
            verdict, prob = predict_tamper_patch(residual, sc_patch, clf_patch, thr_patch)
        col1, col2 = st.columns([1.2, 1.8])
        with col2:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        with col1:
            st.success(f"🖨️ **Scanner:** {label}")
            st.info(f"📊 Confidence: {conf:.2f}%")
            st.write("---")
            st.write(f"🧾 **Tamper Status:** {verdict}")
            st.write(f"📈 Probability: `{prob:.3f}`")
else:
    st.info("👆 Upload a scanned image to begin analysis.")
