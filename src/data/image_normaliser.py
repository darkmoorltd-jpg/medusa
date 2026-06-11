import numpy as np

def normalise_xray(img):
    """Min-max scale to [0,1]."""
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    return img

def normalise_ct(img, window_center=40, window_width=400):
    """Apply DICOM windowing for lung CT."""
    img = img.astype(np.float32)
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    return img

def normalise_mri(img, lower=1, upper=99):
    """Percentile-based clipping."""
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [lower, upper])
    img = np.clip(img, p_low, p_high)
    img = (img - p_low) / (p_high - p_low + 1e-8)
    return img