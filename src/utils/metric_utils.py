import numpy as np

def dice_score(pred, target, smooth=1e-6):
    pred = np.asarray(pred > 0.5).astype(np.float32)
    target = np.asarray(target).astype(np.float32)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou(pred, target, smooth=1e-6):
    pred = np.asarray(pred > 0.5).astype(np.float32)
    target = np.asarray(target).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)