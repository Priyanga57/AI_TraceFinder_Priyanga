import os
import pickle
import numpy as np
import cv2
import pywt
import json
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Base model directory
BASE_DIR = r"C:\AI Trace Finder\App\models"

# === Load scanner model and assets ===
def load_scanner_models():
    paths = {
        "model": os.path.join(BASE_DIR, "scanner_hybrid.keras"),
        "encoder": os.path.join(BASE_DIR, "hybrid_label_encoder.pkl"),
        "scaler": os.path.join(BASE_DIR, "hybrid_feat_scaler.pkl"),
        "fp": os.path.join(BASE_DIR, "scannerfingerprints.pkl"),
        "keys": os.path.join(BASE_DIR, "fp_keys.npy"),
    }

    model = tf.keras.models.load_model(paths["model"], compile=False)
    with open(paths["encoder"], "rb") as f:
        encoder = pickle.load(f)
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["fp"], "rb") as f:
        fp = pickle.load(f)
    keys = np.load(paths["keys"], allow_pickle=True)
    return model, encoder, scaler, fp, keys


# === Feature extraction functions ===
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    return gray.astype(np.float32) / 255.0


def extract_features(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    features = np.concatenate([LL.flatten(), LH.flatten(), HL.flatten(), HH.flatten()])
    return features[:5000]  # Limit for consistency


# === Prediction ===
def predict_scanner(img):
    model, encoder, scaler, _, _ = load_scanner_models()
    feat = extract_features(preprocess_image(img))
    feat_scaled = scaler.transform([feat])
    preds = model.predict(feat_scaled)
    label_idx = np.argmax(preds)
    confidence = float(np.max(preds) * 100)
    label = encoder.inverse_transform([label_idx])[0]
    return label, confidence


# === Tamper Detection Models ===
def load_tamper_models():
    patch_dir = os.path.join(BASE_DIR, "artifacts_tamper_patch")
    pair_dir = os.path.join(BASE_DIR, "artifacts_tamper_pair")

    with open(os.path.join(patch_dir, "patch_scaler.pkl"), "rb") as f:
        patch_scaler = pickle.load(f)
    with open(os.path.join(patch_dir, "patch_svm_sig_calibrated.pkl"), "rb") as f:
        patch_svm = pickle.load(f)
    with open(os.path.join(patch_dir, "thresholds_patch.json"), "r") as f:
        patch_thresh = json.load(f)

    with open(os.path.join(pair_dir, "pair_scaler.pkl"), "rb") as f:
        pair_scaler = pickle.load(f)
    with open(os.path.join(pair_dir, "pair_svm_sig.pkl"), "rb") as f:
        pair_svm = pickle.load(f)
    with open(os.path.join(pair_dir, "pair_thresholds_topk.json"), "r") as f:
        pair_thresh = json.load(f)

    return patch_scaler, patch_svm, patch_thresh, pair_scaler, pair_svm, pair_thresh


def predict_tamper(img):
    patch_scaler, patch_svm, patch_thresh, _, _, _ = load_tamper_models()
    feat = extract_features(preprocess_image(img))
    feat_scaled = patch_scaler.transform([feat])
    prob = patch_svm.predict_proba(feat_scaled)[0][1]
    verdict = "Tampered" if prob > patch_thresh["threshold"] else "Clean"
    return verdict, prob * 100
