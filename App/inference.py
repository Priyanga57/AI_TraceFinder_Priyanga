# inference.py
import numpy as np
import pickle, json, math
from pathlib import Path
from skimage.feature import local_binary_pattern as sk_lbp
import cv2, pywt, tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).parent / "models"
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

# ---------------- MODEL LOADING (CACHED) ----------------
def load_models():
    # Scanner model
    hyb_model = tf.keras.models.load_model(must_exist(ART_SCN/"scanner_hybrid.keras"))
    le_sc = pickle.load(open(must_exist(ART_SCN/"hybrid_label_encoder.pkl"), "rb"))
    sc_sc = pickle.load(open(must_exist(ART_SCN/"hybrid_feat_scaler.pkl"), "rb"))
    fps = pickle.load(open(must_exist(ART_SCN/"scannerfingerprints.pkl"), "rb"))
    fp_keys = np.load(must_exist(ART_SCN/"fp_keys.npy"), allow_pickle=True).tolist()

    # Tamper patch model
    sc_tp = pickle.load(open(must_exist(ART_TP/"patch_scaler.pkl"), "rb"))
    clf_tp = pickle.load(open(must_exist(ART_TP/"patch_svm_sig_calibrated.pkl"), "rb"))
    THRS_TP = json.load(open(must_exist(ART_TP/"thresholds_patch.json"), "r"))

    return {
        "scanner": (hyb_model, le_sc, sc_sc, fps, fp_keys),
        "tamper_patch": (sc_tp, clf_tp, THRS_TP)
    }

# ---------------- FEATURE FUNCTIONS ----------------
def corr2d(a,b):
    a=a.ravel();b=b.ravel()
    a-=a.mean(); b-=b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float((a@b)/d) if d!=0 else 0.0

def lbp_hist_safe(img, P=8, R=1.0):
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng<1e-12 else (img - float(np.min(img))) / (rng+1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P+2
    hist,_ = np.histogram(codes, bins=np.arange(n_bins+1), density=True)
    return hist.astype(np.float32)

def fft_radial_energy(img, K=6):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    h, w = mag.shape; cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h,:w]; r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = np.linspace(0, r.max()+1e-6, K+1)
    feats = [float(mag[(r>=bins[i])&(r<bins[i+1])].mean() if np.any((r>=bins[i])&(r<bins[i+1])) else 0.0) for i in range(K)]
    return np.asarray(feats, dtype=np.float32)

def residual_stats(img):
    return np.asarray([float(img.mean()), float(img.std()), float(np.mean(np.abs(img)))], dtype=np.float32)

def fft_resample_feats(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
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
    return np.concatenate([
        lbp_hist_safe(img_patch),
        fft_radial_energy(img_patch),
        residual_stats(img_patch),
        fft_resample_feats(img_patch)
    ], axis=0)

# ---------------- SCANNER INFERENCE ----------------
def make_scanner_feats(res, fps, fp_keys, sc_sc):
    v_corr=[corr2d(res,fps[k]) for k in fp_keys]
    v_fft=fft_radial_energy(res).tolist()
    v_lbp=lbp_hist_safe(res).tolist()
    v=np.array(v_corr+v_fft+v_lbp,dtype=np.float32).reshape(1,-1)
    return sc_sc.transform(v)

def predict_scanner(res, hyb_model, le_sc, sc_sc, fps, fp_keys):
    x_img = np.expand_dims(res,axis=(0,-1))
    x_ft = make_scanner_feats(res, fps, fp_keys, sc_sc)
    preds = hyb_model.predict([x_img, x_ft], verbose=0)
    ps = preds.ravel(); s_idx=int(np.argmax(ps))
    return str(le_sc.classes_[s_idx]), float(ps[s_idx]*100.0)

# ---------------- TAMPER INFERENCE ----------------
def infer_tamper_single(res, sc_tp, clf_tp, THRS_TP, patch_size=128, stride=64, max_patches=16, topk=0.3, hit_thr=0.85, min_hits=2):
    H, W = res.shape
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    coords = [(y,x) for y in ys for x in xs]
    np.random.shuffle(coords)
    coords = coords[:min(max_patches, len(coords))]
    patches = [res[y:y+patch_size, x:x+patch_size] for y,x in coords]

    feats = np.stack([make_feat_vector(p) for p in patches],0)
    feats = sc_tp.transform(feats)
    p_patch = clf_tp.predict_proba(feats)[:,1]

    # Top-K scoring
    n=len(p_patch); k=max(1,int(np.ceil(topk*n)))
    top_mean = float(np.mean(np.sort(p_patch)[-k:]))
    thr = THRS_TP.get("global",0.5)
    hits = int((p_patch>=hit_thr).sum())
    tampered = bool((top_mean>=thr) and (hits>=min_hits))
    return tampered, top_mean, thr, hits
