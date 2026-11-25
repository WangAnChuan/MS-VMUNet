import numpy as np
import torch
from medpy.metric.binary import hd95

def compute_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    dice = (2.0 * intersection) / (union + 1e-7) if union > 0 else (1.0 if np.sum(pred) == 0 else 0.0)
    iou = intersection / (np.sum(pred | gt) + 1e-7) if np.sum(pred | gt) > 0 else (1.0 if np.sum(pred) == 0 else 0.0)

    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))

    sensitivity = tp / (tp + fn + 1e-7) if (tp + fn) > 0 else 1.0
    specificity = tn / (tn + fp + 1e-7) if (tn + fp) > 0 else 1.0

    if np.sum(pred) > 0 and np.sum(gt) > 0:
        try:
            hd = hd95(pred, gt)
        except Exception:
            hd = 50.0
    elif np.sum(pred) == 0 and np.sum(gt) == 0:
        hd = 0.0
    else:
        hd = 50.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "hd95": float(hd)
    }
