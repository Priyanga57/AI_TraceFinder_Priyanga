import io, pickle, joblib, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image
from skimage.feature import local_binary_pattern as sk_lbp

BASE = Path(__file__).resolve().parent / "models"

def load_any_hybrid():
    for p in [BASE / "scanner_hybrid_14.keras", BASE / "scanner_hybrid.keras", BASE / "scanner_hybrid.h5", BASE / "scanner_hybrid"]:
        if p.exists():
            return tf.keras.models.load_model(str(p))
    raise FileNotFoundError("scanner_hybrid(.keras) not found")

hyb_model = load_any_hybrid()
required_tab_feats = int(hyb_model.inputs[1].shape[-1])

scaler_inf = joblib.load(BASE / "hybrid_feat_scaler.pkl")
if getattr(scaler_inf, "n_features_in_", None) != required_tab_feats:
    raise RuntimeError("Scaler feature size does not match model/tabular input.")

with open(BASE / "scanner_fingerprints.pkl", "rb") as f:
    scanner_fps_inf = pickle.load(f)
fp_keys_inf = np.load(BASE / "fp_keys.npy", allow_pickle=True).tolist()

le_inf = joblib.load(BASE / "hybrid_label_encoder.pkl")

def corr2d(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d != 0 else 0.0

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = np.linspace(0, r.max() + 1e-6, K + 1)
    feats = [float(mag[(r >= bins[i]) & (r < bins[i + 1])].mean() if np.any((r >= bins[i]) & (r < bins[i + 1])) else 0.0) for i in range(K)]
    return np.array(feats, dtype=np.float32)

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - float(np.min(img))) / (rng + 1e-8)
    g8 = (g * 255.0).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.astype(np.float32)

def residualstats(img):
    return np.array([img.mean(), img.std(), np.mean(np.abs(img - img.mean()))], dtype=np.float32)

def fftresamplefeats(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max() + 1e-6
    b1 = (r >= 0.25 * rmax) & (r < 0.35 * rmax)
    b2 = (r >= 0.35 * rmax) & (r < 0.50 * rmax)
    e1 = float(mag[b1].mean() if np.any(b1) else 0.0)
    e2 = float(mag[b2].mean() if np.any(b2) else 0.0)
    return np.array([e1, e2, e2 / (e1 + 1e-8)], dtype=np.float32)

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps_inf[k]) for k in fp_keys_inf]
    v_fft = fft_radial_energy(res, K=6)
    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
    v = np.array(v_corr + list(v_fft) + list(v_lbp), dtype=np.float32).reshape(1, -1)
    return scaler_inf.transform(v)

def predict_from_bytes(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    res = preprocess_residual_from_array(np.array(img))
    x_img = np.expand_dims(res, axis=(0, -1))
    x_ft = make_feats_from_res(res)
    prob = np.array(hyb_model.predict([x_img, x_ft], verbose=0)).ravel()
    idx = int(np.argmax(prob))
    label = le_inf.classes_[idx]
    conf = float(prob[idx] * 100.0)
    k = min(3, prob.size)
    top_idx = np.argpartition(prob, -k)[-k:]
    top_idx = top_idx[np.argsort(prob[top_idx])[::-1]]
    top3 = [(le_inf.classes_[i], float(prob[i] * 100.0)) for i in top_idx]
    return {"label": label, "confidence": conf, "top3": top3}
