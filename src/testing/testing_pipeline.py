import os
from collections import Counter

import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.model.backbone_registry import BACKBONE_FNS
from src.model.multiscale_projection_loss import MultiProjectionLayer
from src.data.dataset import (
    AnomalyTestDataset,
    get_data_transforms,
)
from src.utils.checkpoint_loader import safe_torch_load
from src.inference.infer_anomaly_maps import infer_anomaly_maps_from_dataset
from src.evaluation.metrics import compute_ad_metrics, collect_eval_data
from src.postprocessing.image_level_output import (
    predict_image_label,
    save_image_level_decision,
)
from src.postprocessing.confusion_case_split import (
    save_confusion_case_split,
    get_confusion_case,
)
from src.postprocessing.pixel_level_output import save_pixel_level_outputs
from src.evaluation.evaluation_plots import plot_roc_and_pro


def run_test(dataset_path, test_metadata, output_dir): 
    """
    Evaluation pipeline with ground truth.

    Runs inference on labeled test data, computes anomaly detection metrics
    (AUROC sample/pixel, AUPRO), and generates pixel-level visualizations,
    image-level predictions, and confusion case analysis (TP/FP/TN/FN),
    returning test metadata, an aggregated evaluation summary, and per-image scores.
    """
    # --------------------------------------------------
    # Initialize testing setup
    # --------------------------------------------------
    device = torch.device(test_metadata["device"])

    dataset_path = test_metadata["paths"]["dataset"]
    image_size = test_metadata["dataset"]["image_size"]

    backbone = test_metadata["model"]["backbone"]
    base = test_metadata["model"]["base"]
    checkpoint_path = test_metadata["model"]["checkpoint_path"]

    image_quantile = test_metadata["thresholds"]["image_quantile"]
    image_threshold = test_metadata["thresholds"]["image_threshold"]
    pixel_threshold = test_metadata["thresholds"]["pixel_threshold"]
    min_region_area = test_metadata["thresholds"]["min_region_area"]
    polygon_epsilon_ratio = test_metadata["polygon"]["polygon_epsilon_ratio"]

    morphology_config = test_metadata.get("morphology", {})
    morphology_enabled = morphology_config.get("enabled", False)
    morphology_kernel_size = morphology_config.get("kernel_size", 3)
    morphology_open_iterations = morphology_config.get("open_iterations", 1)
    morphology_close_iterations = morphology_config.get("close_iterations", 1)

    image_level_counter = Counter()
    error_counter = Counter()
    image_scores = [] 

    # --------------------------------------------------
    # Dataset & transforms (ground truth available for evaluation)    
    # --------------------------------------------------
    test_path = dataset_path
    data_transform, gt_transform = get_data_transforms(image_size)

    test_data = AnomalyTestDataset(
        dataset_path=test_path,
        transform=data_transform,
        gt_transform=gt_transform,
        image_size=image_size,
    )
    
    # Record dataset size (known at runtime)
    test_metadata["dataset"]["num_test_images"] = len(test_data)

    # --------------------------------------------------
    # Dataloader
    # --------------------------------------------------
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    # Build model components 
    encoder_fn, decoder_fn = BACKBONE_FNS[backbone]
    encoder, bn = encoder_fn(pretrained=True)
    decoder = decoder_fn(pretrained=False)
    proj_layer = MultiProjectionLayer(base=base)

    # Move to device
    encoder, bn = encoder.to(device), bn.to(device)
    decoder = decoder.to(device)
    proj_layer = proj_layer.to(device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    ckpt = safe_torch_load(checkpoint_path, device)
    ckpt_backbone = ckpt.get("backbone")
    if ckpt_backbone is not None and ckpt_backbone != backbone:
        raise ValueError(
            f"Backbone mismatch: config uses '{backbone}', "
            f"but checkpoint was trained with '{ckpt_backbone}'."
        )
    proj_layer.load_state_dict(ckpt["proj"])
    decoder.load_state_dict(ckpt["decoder"])
    bn.load_state_dict(ckpt["bn"])

    # --------------------------------------------------
    # Switch to evaluation mode
    # --------------------------------------------------
    # Freeze encoder (feature extractor is not trained)
    encoder.eval()
    proj_layer.eval()
    bn.eval()
    decoder.eval()

    # --------------------------------------------------
    # Model inference and anomaly map generation
    # --------------------------------------------------
    with torch.no_grad():
        anomaly_maps = infer_anomaly_maps_from_dataset(
            encoder=encoder,
            proj_layer=proj_layer,
            bn=bn,
            decoder=decoder,
            dataloader=test_dataloader,
            device=device,
        )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    auroc_px, auroc_sp, aupro_px = compute_ad_metrics(
        test_dataloader,
        anomaly_maps,
    )

    gt_px, pr_px, masks = collect_eval_data(test_dataloader, anomaly_maps)
    amaps = np.array(anomaly_maps)

    # --------------------------------------------------
    # Pixel-level visualization
    # --------------------------------------------------
    for idx, anomaly_map in tqdm(
        enumerate(anomaly_maps),
        total=len(anomaly_maps),
        desc="Pixel-level visualization",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} images [{elapsed}<{remaining}, {rate_fmt}]",
        leave=False,
        mininterval=0.5,
    ):
        img_path = test_data.img_paths[idx]
        filename = os.path.basename(img_path)
        
        sample = test_data[idx]
        category = sample[3]  # "good_clean", "good_noisy", "cracks", ...
        
        original_image = cv2.imread(img_path)
        if original_image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        save_pixel_level_outputs(
            anomaly_map=anomaly_map,
            pixel_threshold=pixel_threshold,            
            min_region_area=min_region_area,
            morphology_enabled=morphology_enabled,
            morphology_kernel_size=morphology_kernel_size,
            morphology_open_iterations=morphology_open_iterations,
            morphology_close_iterations=morphology_close_iterations,
            original_image=original_image,
            save_root=output_dir,
            filename=filename,
            category=category,
            polygon_epsilon_ratio=polygon_epsilon_ratio,
        )

    # --------------------------------------------------
    # Image-level decision and confusion case analysis (GT-based)
    # --------------------------------------------------    
    for idx, amap in tqdm(
        enumerate(anomaly_maps),
        total=len(anomaly_maps),
        desc="Image-level decision & confusion case split",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} images [{elapsed}<{remaining}, {rate_fmt}]",        
        leave=False,
        mininterval=0.5,
    ):
        img_path = test_data.img_paths[idx]
        sample = test_data[idx]
        
        gt_label = sample[2]  # 0 = normal, 1 = anomal
        category = sample[3]  # "good", "crack", ...
        
        pred_label = predict_image_label(
            anomaly_map=amap,
            image_threshold=image_threshold,
            image_quantile=image_quantile,
        )
        
        image_score = float(np.quantile(amap, image_quantile))
        image_scores.append({
            "image_name": os.path.basename(img_path),
            "category": category,
            "gt_label": gt_label,
            "predicted_label": pred_label,
            "anomaly_score": image_score,
        })
        
        image_level_counter[pred_label] += 1
        case = get_confusion_case(pred_label, gt_label)
        error_counter[case] += 1
        save_image_level_decision(img_path, pred_label, output_dir)
        save_confusion_case_split(
            image_path=img_path,
            case=case,
            save_root=output_dir,
            category=category,
        )

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_roc_and_pro(
        gt_px,
        pr_px,
        masks,
        amaps,
        output_dir,
    )


    # --------------------------------------------------
    # Structured test summary as JSON
    # --------------------------------------------------
    summary = {
        "metrics": {
            "AUROC_sample": auroc_sp,
            "AUROC_pixel": auroc_px,
            "AUPRO_pixel": aupro_px,
        },
        "image_level": {
            "predicted_normal": image_level_counter["normal"],
            "predicted_anomal": image_level_counter["anomal"],
        },
        "confusion_case_split": {
            "TP": error_counter["TP"],
            "FP": error_counter["FP"],
            "TN": error_counter["TN"],
            "FN": error_counter["FN"],
        },
    }

    return {
        "test_metadata": test_metadata,
        "summary": summary,
        "image_scores": image_scores,
    }

