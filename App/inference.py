import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import cv2
import os

# Paths to models
BASE_MODEL_PATH = os.path.join("..", "models")
SCANNER_MODEL = os.path.join(BASE_MODEL_PATH, "scanner_hybrid.keras")
LABEL_ENCODER = os.path.join(BASE_MODEL_PATH, "hybrid_label_encoder.pkl")
SCALER = os.path.join(BASE_MODEL_PATH, "hybrid_feat_scaler.pkl")
FP_KEYS = os.path.join(BASE_MODEL_PATH, "fp_keys.npy")

# Tamper artifacts
PATCH_DIR = os.path.join(BASE_MODEL_PATH, "artifacts_tamper_patch")
PAIR_DIR = os.path.join(BASE_MODEL_PATH, "artifacts_tamper_pair")

# Load scanner model
scanner_model = load_model(SCANNER_MODEL)
with open(LABEL_ENCODER, "rb") as f:
    label_encoder = pickle.load(f)
with open(SCALER, "rb") as f:
    feat_scaler = pickle.load(f)
fp_keys = np.load(FP_KEYS, allow_pickle=True)

# Load tamper detection models
with open(os.path.join(PATCH_DIR, "patch_scaler.pkl"), "rb") as f:
    patch_scaler = pickle.load(f)
with open(os.path.join(PATCH_DIR, "patch_svm_sig_calibrated.pkl"), "rb") as f:
    patch_svm = pickle.load(f)
with open(os.path.join(PATCH_DIR, "thresholds_patch.json"), "r") as f:
    thresholds_patch = json.load(f)

with open(os.path.join(PAIR_DIR, "pair_scaler.pkl"), "rb") as f:
    pair_scaler = pickle.load(f)
with open(os.path.join(PAIR_DIR, "pair_svm_sig.pkl"), "rb") as f:
    pair_svm = pickle.load(f)
with open(os.path.join(PAIR_DIR, "pair_thresholds_topk.json"), "r") as f:
    pair_thresholds = json.load(f)

# -------------------
# Helper functions
# -------------------
def extract_features(image_path):
    """
    Extract features for scanner classification.
    Here we assume fp_keys are features to be extracted.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features = img.flatten().astype(np.float32)
    # normalize based on saved scaler
    features = feat_scaler.transform([features])
    return features

def predict_scanner(image_path):
    features = extract_features(image_path)
    pred = scanner_model.predict(features)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return label[0]

def predict_tamper_patch(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    feat = img.flatten().astype(np.float32).reshape(1, -1)
    feat = patch_scaler.transform(feat)
    score = patch_svm.decision_function(feat)[0]
    tamper = score > thresholds_patch["threshold"]
    return tamper, float(score)

def predict_tamper_pair(image_path1, image_path2):
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (64, 64)).flatten()
    img2 = cv2.resize(img2, (64, 64)).flatten()
    feat = np.concatenate([img1, img2]).reshape(1, -1)
    feat = pair_scaler.transform(feat)
    score = pair_svm.decision_function(feat)[0]
    tamper = score > pair_thresholds["threshold_topk"]
    return tamper, float(score)
