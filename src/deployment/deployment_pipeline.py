import os
from collections import Counter

import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.model.backbone_registry import BACKBONE_FNS
from src.model.multiscale_projection_loss import MultiProjectionLayer
from src.data.dataset import (
    AnomalyDeploymentDataset,
    get_data_transforms,
)
from src.utils.checkpoint_loader import safe_torch_load
from src.inference.infer_anomaly_maps import infer_anomaly_maps_from_dataset
from src.postprocessing.image_level_output import (
    predict_image_label,
    save_image_level_decision,
)
from src.postprocessing.pixel_level_output import save_pixel_level_outputs


def run_deployment(deploy_metadata, output_dir):
    """
    Deployment pipeline without ground truth.

    Applies a trained model to unseen data, generates anomaly maps,
    and produces pixel- and image-level predictions,
    returning deployment metadata, a prediction summary, and per-image scores.
    """
    # --------------------------------------------------
    # Initialize deployment setup
    # --------------------------------------------------
    device = torch.device(deploy_metadata["device"])

    dataset_path = deploy_metadata["paths"]["dataset"]
    image_size = deploy_metadata["dataset"]["image_size"]

    backbone = deploy_metadata["model"]["backbone"]
    base = deploy_metadata["model"]["base"]
    checkpoint_path = deploy_metadata["model"]["checkpoint_path"]

    image_quantile = deploy_metadata["thresholds"]["image_quantile"]
    image_threshold = deploy_metadata["thresholds"]["image_threshold"]
    pixel_threshold = deploy_metadata["thresholds"]["pixel_threshold"]
    min_region_area = deploy_metadata["thresholds"]["min_region_area"]
    polygon_epsilon_ratio = deploy_metadata["polygon"]["polygon_epsilon_ratio"]

    morphology_config = deploy_metadata.get("morphology", {})
    morphology_enabled = morphology_config.get("enabled", False)
    morphology_kernel_size = morphology_config.get("kernel_size", 3)
    morphology_open_iterations = morphology_config.get("open_iterations", 1)
    morphology_close_iterations = morphology_config.get("close_iterations", 1)

    image_quantile = deploy_metadata["thresholds"]["image_quantile"]
    image_threshold = deploy_metadata["thresholds"]["image_threshold"]

    image_level_counter = Counter()
    image_scores = []

    # --------------------------------------------------
    # Dataset & transforms (deployment uses NO ground truth)    
    # --------------------------------------------------
    test_path = dataset_path  
    data_transform, _ = get_data_transforms(image_size)

    test_data = AnomalyDeploymentDataset(
        dataset_path=test_path, 
        transform=data_transform,
        image_size=image_size,
    )

    # Record dataset size (known at runtime)
    deploy_metadata["dataset"]["num_deploy_images"] = len(test_data)

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
    # Switch to eval mode
    # --------------------------------------------------
    encoder.eval()
    proj_layer.eval()
    bn.eval()
    decoder.eval()

    # --------------------------------------------------
    # Inference (anomaly map generation + runtime measurement)
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
    # Pixel-level visualization
    # --------------------------------------------------
    for idx, anomaly_map in tqdm(
        enumerate(anomaly_maps),
        total=len(anomaly_maps),
        desc="Pixel-level visualization",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} image [{elapsed}<{remaining}, {rate_fmt}]",        
        leave=False,
        mininterval=0.5,
    ):
        img_path = test_data.img_paths[idx]
        filename = os.path.basename(img_path)
        
        sample = test_data[idx]
        category = sample[1] # "good_clean", "good_noisy", "cracks", ...
        
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
    # Image-level decision: binary classification without ground truth or error analysis
    # --------------------------------------------------
    for idx, amap in tqdm(
        enumerate(anomaly_maps),
        total=len(anomaly_maps),
        desc="Image-level decision",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} images [{elapsed}<{remaining}, {rate_fmt}]",        
        leave=False,
        mininterval=0.5,
    ):
        img_path = test_data.img_paths[idx]

        pred_label = predict_image_label(
            anomaly_map=amap,
            image_threshold=image_threshold,
            image_quantile=image_quantile,
        )

        image_score = float(np.quantile(amap, image_quantile))
        
        image_scores.append({
            "image_name": os.path.basename(img_path),
            "predicted_label": pred_label,
            "anomaly_score": image_score,
        })

        image_level_counter[pred_label] += 1
        save_image_level_decision(img_path, pred_label, output_dir)

    # --------------------------------------------------
    # Structured deployment summary as JSON
    # --------------------------------------------------
    summary = {
        "image_level": {
            "predicted_normal": image_level_counter["normal"],
            "predicted_anomal": image_level_counter["anomal"],
        },
        "checkpoint": checkpoint_path,
    }

    return {
        "deployment_metadata": deploy_metadata,
        "summary": summary,
        "image_scores": image_scores,
    }

