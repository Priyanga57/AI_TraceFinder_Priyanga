# app/inference.py
import numpy as np
import cv2
from skimage.feature import local_binary_pattern

# ---------------------------
# FEATURE EXTRACTION HELPERS
# ---------------------------

def corr2d(a, b):
    """Compute normalized 2D correlation between two matrices."""
    a_mean, b_mean = np.mean(a), np.mean(b)
    num = np.sum((a - a_mean) * (b - b_mean))
    den = np.sqrt(np.sum((a - a_mean)**2) * np.sum((b - b_mean)**2))
    if den == 0:
        return 0.0
    return float(num / den)


def fft_radial_energy(img, bins=6):
    """
    Compute FFT-based radial energy distribution of the image.
    Returns a histogram-like feature vector of frequency energies.
    """
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    cy, cx = np.array(mag.shape) // 2
    y, x = np.indices(mag.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = (r / np.max(r)) * (bins - 1)
    r = r.astype(np.int32)
    feat = np.zeros(bins, dtype=np.float32)
    for i in range(bins):
        mask = (r == i)
        if np.any(mask):
            feat[i] = np.mean(mag[mask])
    feat /= np.sum(feat) + 1e-8
    return feat.tolist()


def lbp_hist_safe(img, P=8, R=1.0):
    """
    Compute safe Local Binary Pattern histogram (uniform).
    Used for scanner and tamper features.
    """
    lbp = local_binary_pattern(img, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()


def make_feats_from_res(residual, fps, fp_keys):
    """
    Combine correlation, FFT, and LBP features into one hybrid feature vector.
    """
    v_corr = [corr2d(residual, fps[k]) for k in fp_keys]
    v_fft = fft_radial_energy(residual, 6)
    v_lbp = lbp_hist_safe(residual, 8, 1.0)
    return np.array(v_corr + v_fft + v_lbp, dtype=np.float32)
