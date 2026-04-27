import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from skimage import measure


def compute_roc_curve_data(gt, scores):
    gt = np.asarray(gt).astype(int).ravel()
    scores = np.asarray(scores).ravel()
    return roc_curve(gt, scores)


def compute_pro_curve_data(masks, amaps, num_th=200, fpr_limit=0.3):
    masks = np.asarray(masks).astype(int)
    amaps = np.asarray(amaps)

    assert masks.ndim == 3, "masks.ndim must be 3"
    assert amaps.ndim == 3, "amaps.ndim must be 3"
    assert masks.shape == amaps.shape, "masks.shape and amaps.shape must match"
    assert set(np.unique(masks)).issubset({0, 1}), "masks must be binary"

    min_th = float(amaps.min())
    max_th = float(amaps.max())

    if max_th == min_th:
        return np.array([0.0]), np.array([0.0])

    binary_amaps = np.zeros_like(amaps, dtype=bool)
    fprs = []
    pros_all = []

    for th in np.linspace(min_th, max_th, num_th, endpoint=False):
        binary_amaps[:] = amaps > th

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                coords = region.coords
                pros.append(binary_amap[coords[:, 0], coords[:, 1]].sum() / region.area)

        inverse_masks = 1 - masks
        denom = inverse_masks.sum()
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / denom if denom > 0 else 0.0

        fprs.append(fpr)
        pros_all.append(np.mean(pros) if pros else 0.0)

    fprs = np.asarray(fprs)
    pros_all = np.asarray(pros_all)

    keep = fprs < fpr_limit
    fprs = fprs[keep]
    pros_all = pros_all[keep]

    if len(fprs) == 0:
        return np.array([0.0]), np.array([0.0])

    if fprs.max() > 0:
        fprs = fprs / fprs.max()

    return fprs, pros_all


def plot_roc_curve(gt, scores, out_dir, filename="roc_curve.png"):
    fpr, tpr, _ = compute_roc_curve_data(gt, scores)

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close()


def plot_pro_curve(masks, amaps, out_dir, filename="pro_curve.png", num_th=200, fpr_limit=0.3):
    fpr, pro = compute_pro_curve_data(masks, amaps, num_th=num_th, fpr_limit=fpr_limit)

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, pro, label="PRO")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Per-Region Overlap")
    plt.title("PRO Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close()


def plot_roc_and_pro(
    gt,
    scores,
    masks,
    amaps,
    out_dir,
    filename="roc_pro_cplot.png",
    num_th=200,
    fpr_limit=0.3,
):
    fpr_roc, tpr_roc, _ = compute_roc_curve_data(gt, scores)
    fpr_pro, pro = compute_pro_curve_data(masks, amaps, num_th=num_th, fpr_limit=fpr_limit)

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_roc, tpr_roc, label="ROC")
    plt.plot(fpr_pro, pro, label="PRO")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Performance")
    plt.title("ROC and PRO Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close()