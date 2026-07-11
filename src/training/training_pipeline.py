import os

import torch
from tqdm import tqdm
import copy

from src.inference.infer_anomaly_maps import infer_anomaly_maps_from_dataset
from src.evaluation.metrics import (
    compute_ad_metrics,
    collect_eval_data,
)
from src.evaluation.noise_plots import plot_noisy_examples_grid
from src.model.backbone_registry import BACKBONE_FNS
from src.model.multiscale_projection_loss import (
    MultiProjectionLayer,
    Revisit_RDLoss,
    loss_function,
)
from src.data.dataset import (
    AnomalyTrainDataset,
    AnomalyTestDataset,
    get_data_transforms,
)

def run_training(train_metadata):
    """
    Training pipeline for anomaly detection.

    Trains projection, bottleneck, and decoder modules with a frozen encoder.
    Performs epoch-wise evaluation and returns the best checkpoint state,
    best evaluation metrics, full evaluation history and training metadata.
    """
    # --------------------------------------------------
    # Initialize training setup
    # --------------------------------------------------
    device = torch.device(train_metadata["device"])

    dataset_path = train_metadata["paths"]["dataset"] 
    image_size = train_metadata["dataset"]["image_size"]
    output_dir = train_metadata["paths"]["output_dir"]

    backbone = train_metadata["model"]["backbone"]
    base = train_metadata["model"]["base"]

    batch_size = train_metadata["dataloader"]["batch_size"]
    shuffle_train = train_metadata["dataloader"]["shuffle_train"]

    epochs = train_metadata["training"]["epochs"]
    accumulation_steps = train_metadata["training"]["gradient_accumulation_steps"]

    optimizer_cfg = train_metadata["training"]["optimizer"]
    loss_cfg = train_metadata["training"]["loss"]
    betas = tuple(optimizer_cfg["betas"])

    best_checkpoint = None

    # --------------------------------------------------
    # Dataset & transforms (ground truth available for evaluation)
    # --------------------------------------------------
    train_path = dataset_path 
    test_path = dataset_path   
    data_transform, gt_transform = get_data_transforms(image_size)

    train_data = AnomalyTrainDataset(
        dataset_path=train_path,
        transform=data_transform,
        image_size=image_size,
        pseudo_anomaly_cfg=train_metadata["training"]["pseudo_anomaly"],
    )

    test_data = AnomalyTestDataset(
        dataset_path=test_path,
        transform=data_transform,
        gt_transform=gt_transform,
        image_size=image_size,
    )

    # Record dataset size (known at runtime)
    train_metadata["dataset"]["num_train_images"] = len(train_data)

    # --------------------------------------------------
    # Dataloader
    # --------------------------------------------------
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

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

    # Freeze encoder
    encoder.eval()

    # Losses and optimizers
    proj_loss = Revisit_RDLoss(
        ssot_weight=loss_cfg["ssot_weight"],
        reconstruct_weight=loss_cfg["reconstruct_weight"],
        contrast_weight=loss_cfg["contrast_weight"],
    )

    optimizer_proj = torch.optim.Adam(
        list(proj_layer.parameters()),
        lr=optimizer_cfg["proj_lr"],
        betas=betas,
    )
    optimizer_distill = torch.optim.Adam(
        list(decoder.parameters()) + list(bn.parameters()),
        lr=optimizer_cfg["distill_lr"],
        betas=betas,
    )

    # --------------------------------------------------
    # Evaluation tracking
    # --------------------------------------------------
    best = {
        "epoch": 0,
        "score": 0.0,
        "AUROC_sample": 0.0,
        "AUROC_pixel": 0.0,
        "AUPRO_pixel": 0.0,
    }
    eval_history = {
        "epoch": [],
        "pixel_auroc": [],
        "image_auroc": [],
        "aupro": [],
        "loss_proj": [],
        "loss_distill": [],
        "total_loss": [],
    }

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, epochs + 1):

        # Enable training mode for trainable parts
        bn.train()
        proj_layer.train()
        decoder.train()

        running = {"loss_proj": 0.0, "loss_distill": 0.0, "total_loss": 0.0}

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{epochs}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]",
            leave=False,
            mininterval=0.5,
        )

        for i, (img, img_noise, _) in enumerate(pbar):
            # Debug: save example noisy images (only once)
            if epoch == 1 and i == 0:
                save_path = os.path.join(output_dir, "noisy_examples.png")
                plot_noisy_examples_grid(
                    img,
                    img_noise,
                    save_path,
                )

            # Move data to device (GPU/CPU)
            img = img.to(device)
            img_noise = img_noise.to(device)

            # Feature extraction: encoder is frozen, so autograd is disabled to avoid computing unused gradients
            with torch.no_grad():
                inputs = encoder(img)
                inputs_noise = encoder(img_noise)

            # Projection and distillation losses
            fs_noise, fs = proj_layer(inputs, features_noise=inputs_noise)
            L_proj = proj_loss(inputs_noise, fs_noise, fs)
            outputs = decoder(bn(fs))
            L_distill = loss_function(inputs, outputs)

            # Total loss and backpropagation
            loss = L_distill + loss_cfg["weight_proj"] * L_proj
            loss.backward()

            # Gradient accumulation: update after `accumulation_steps` OR on the final batch to avoid dropping gradients
            is_accumulation_step = (i + 1) % accumulation_steps == 0
            is_last_batch = (i + 1) == len(train_dataloader)

            if is_accumulation_step or is_last_batch:
                optimizer_proj.step()
                optimizer_distill.step()
                optimizer_proj.zero_grad()
                optimizer_distill.zero_grad()

            running["total_loss"] += loss.item()
            running["loss_proj"] += L_proj.item()
            running["loss_distill"] += L_distill.item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "proj": f"{L_proj.item():.4f}",
                    "distill": f"{L_distill.item():.4f}",
                }
            )

        # --------------------------------------------------
        # Model inference and anomaly map generation
        # --------------------------------------------------
        # Eval mode
        proj_layer.eval()
        bn.eval()
        decoder.eval()

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
        # Collect evaluation data
        gt_pixels, pixel_anomaly_scores, masks, gt_images, image_anomaly_scores = collect_eval_data(
            test_dataloader,
            anomaly_maps,
        )
        
        # Compute anomaly detection metrics:
        pixel_auroc, image_auroc, aupro = compute_ad_metrics(
            gt_pixels,
            pixel_anomaly_scores,
            gt_images,
            image_anomaly_scores,
            masks,
            anomaly_maps,
        )

        # Store eval_history
        eval_history["epoch"].append(epoch)
        eval_history["pixel_auroc"].append(pixel_auroc)
        eval_history["image_auroc"].append(image_auroc)
        eval_history["aupro"].append(aupro)
        eval_history["loss_proj"].append(running["loss_proj"] / len(train_dataloader))
        eval_history["loss_distill"].append(
            running["loss_distill"] / len(train_dataloader)
        )
        eval_history["total_loss"].append(running["total_loss"] / len(train_dataloader))

        # Update best-performing model
        score = (pixel_auroc + image_auroc + aupro) / 3
        if score > best["score"]:
            best.update(
                {
                    "score": score,
                    "epoch": epoch,
                    "AUROC_sample": image_auroc,
                    "AUROC_pixel": pixel_auroc,
                    "AUPRO_pixel": aupro,
                }
            )

            best_checkpoint = {
                "backbone": backbone,
                "proj": copy.deepcopy(proj_layer.state_dict()),
                "decoder": copy.deepcopy(decoder.state_dict()),
                "bn": copy.deepcopy(bn.state_dict()),
                "metrics": best,
            }

    return {
        "best_checkpoint": best_checkpoint,
        "best": best,
        "eval_history": eval_history,
        "train_metadata": train_metadata,
    }