from statistics import mean

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.metrics import roc_auc_score, auc
from skimage import measure


def collect_eval_data(dataloader, anomaly_maps):
    """
    Collects pixel-wise ground truth, anomaly scores and masks for evaluation.
    """
    gt_list_px = []
    pr_list_px = []
    masks = []

    for (_, gt, label, _, _), anomaly_map in zip(dataloader, anomaly_maps): 
        # Binarize ground truth mask (threshold at 0.5)
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        gt_np = gt.squeeze().cpu().numpy().astype(int)

        gt_list_px.extend(gt_np.ravel())
        pr_list_px.extend(anomaly_map.ravel())

        masks.append(gt_np)

    masks = np.stack(masks)

    return gt_list_px, pr_list_px, masks


def compute_pixel_auroc(gt_pixels, score_pixels):
    """
    Compute pixel-level AUROC (P-AUROC).
    """
    return round(roc_auc_score(gt_pixels, score_pixels), 4)


def compute_image_auroc(gt_images, score_images):
    """
    Compute image-level AUROC (I-AUROC).
    """
    return round(roc_auc_score(gt_images, score_images), 4)


def compute_pro_auc(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """
    Compute the area under the PRO (Per-Region Overlap) curve for a single anomalous sample.

    PRO measures region-wise overlap between predicted anomaly maps
    and ground truth defect regions, aggregated over multiple thresholds.

    The final AUC is computed for false positive rates up to 0.3.
    """
    # Input validation
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    # Storage for PRO curve values
    records = {"pro": [], "fpr": [], "threshold": []}
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    # Threshold range
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []

        # Compute region-wise overlap
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]

                # True positive pixels inside each defect region
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        # Compute false positive rate
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        records["pro"].append(mean(pros))
        records["fpr"].append(fpr)
        records["threshold"].append(th)

    # Restrict evaluation to low-FPR regime
    df = pd.DataFrame(records)
    df = df[df["fpr"] <= 0.3]

    # Compute AUPRO by integrating the PRO curve up to FPR=0.3 and normalizing by the integration limit
    alpha = 0.3
    pro_auc = auc(df["fpr"], df["pro"]) / alpha

    return pro_auc  


def compute_mean_pro_auc(pro_auc_scores):
    """
    Compute mean AUPRO over all anomalous samples.
    """
    return round(float(np.mean(pro_auc_scores)), 4)



def compute_ad_metrics(dataloader, anomaly_maps):
    """
    Compute standard anomaly detection metrics.

    Returns:
        - Pixel-level AUROC
        - Sample-level AUROC
        - Mean AUPRO over anomalous samples
    """
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    pro_auc_list = []

    for (_, gt, label, _, _), anomaly_map in zip(dataloader, anomaly_maps):
        # Binarize ground truth mask
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0

        # Calculate PRO score for anomalous samples
        if label.item() != 0:
            pro_auc_list.append(
                compute_pro_auc(
                    gt.squeeze(0).cpu().numpy().astype(int),
                    anomaly_map[np.newaxis, :, :],
                )
            )

        # Collect pixel-level predictions
        gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
        pr_list_px.extend(anomaly_map.ravel())

        # Collect sample-level predictions
        gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
        pr_list_sp.append(np.max(anomaly_map))

    auroc_px = compute_pixel_auroc(gt_list_px, pr_list_px)
    auroc_sp = compute_image_auroc(gt_list_sp, pr_list_sp)
    mean_pro_auc = compute_mean_pro_auc(pro_auc_list)

    return auroc_px, auroc_sp, mean_pro_auc

