import os
import json
import time
from datetime import datetime

import torch
import pandas as pd

from src.utils.cli import parse_cli_args
from src.utils.config import load_yaml_config, override
from src.deployment.deployment_pipeline import run_deployment


def main():
    """
    Script entry point.

    Resolves configuration into deployment metadata,
    runs the deployment pipeline, and persists all results.
    """
    # --------------------------------------------------
    # Load config & args
    # --------------------------------------------------
    args = parse_cli_args(default_config="config_deployment.yaml")
    cfg = load_yaml_config(args.config)

# --------------------------------------------------
    # Resolve dataset path (CLI > config)
    # --------------------------------------------------
    dataset_path = override(
        args.dataset,
        cfg["paths"]["dataset"],
    )
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    # --------------------------------------------------
    # Resolve checkpoint paths (CLI > config)
    # --------------------------------------------------

    checkpoint_path = override(
        args.checkpoint_path,
        cfg["paths"]["checkpoint_path"],
    )   
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # --------------------------------------------------
    # Setup output directory (CLI > config)
    # --------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root = override(
        args.save_folder,
        os.path.join(
            cfg["paths"]["output_root"],
            "deployment",
            timestamp,
        ),
    )
    os.makedirs(save_root, exist_ok=True)

    output_dir = os.path.join(save_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Define and record the concrete training run (resolved from YAML and CLI)   
    # --------------------------------------------------
    deploy_metadata = {
        "device": cfg["project"]["device"],
        "run": {
            "dataset": dataset_name,   
        },
        "paths": {
            "dataset": dataset_path,
            "output_dir": output_dir,
        },
        "model": {
            "backbone": cfg["model"]["backbone"],
            "base": cfg["model"]["backbones"][cfg["model"]["backbone"]]["base"],
            "checkpoint_path": checkpoint_path,
        },
        "dataset": {
            "image_size": cfg["dataset"]["image_size"],
        },
        "thresholds": {
            "image_quantile": cfg["thresholds"].get("image_quantile", 0.99),
            "image_threshold": cfg["thresholds"].get("image_threshold", 0.0),
            "pixel_threshold": cfg["thresholds"].get("pixel_threshold", 0.8),
            "min_region_area": cfg["thresholds"].get("min_region_area", 50),
        },
        "polygon": {
            "polygon_epsilon_ratio": cfg["thresholds"].get(
                "polygon_epsilon_ratio", 0.02
            ),
        },
        "morphology": cfg.get("morphology", {}),
    }

    # Resolve runtime objects    
    device = torch.device(deploy_metadata["device"])

   # --------------------------------------------------
    # Run test pipeline with end-to-end runtime measurement
    # (inference, anomaly map generation and postprocessing)
    # --------------------------------------------------
    print("=" * 80)
    print("[Deployment]")
    print(f"Dataset             : {dataset_name}")   
    print(f"Backbone            : {deploy_metadata['model']['backbone']}")
    print(f"Checkpoint          : {checkpoint_path}")
    print("Running deployment...", end="", flush=True)

    # --- End-to-End start ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    result = run_deployment(
        deploy_metadata=deploy_metadata,
        output_dir=output_dir,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    # --- End-to-End end ---

    print("\r\033[2K", end="", flush=True)

    # --------------------------------------------------
    # Unpack results
    # --------------------------------------------------
    deployment_metadata = result["deployment_metadata"]
    summary = result["summary"]
    image_scores = result["image_scores"]

    num_images = deployment_metadata["dataset"]["num_deploy_images"]
    e2e_time = t_end - t_start

    summary["runtime"] = {
        "end_to_end_time_sec": e2e_time,
        "end_to_end_fps": num_images / e2e_time if num_images > 0 else 0.0,
        "end_to_end_ms_per_image": (
            (e2e_time / num_images) * 1000 if num_images > 0 else 0.0
        ),
    }

    # --------------------------------------------------
    # IO: Deploy metadata
    # --------------------------------------------------
    with open(os.path.join(output_dir, "deployment_metadata.json"), "w") as deployment_metadata_file:
        json.dump(deployment_metadata, deployment_metadata_file, indent=2)
    
    # --------------------------------------------------
    # IO: Save summary (JSON)
    # --------------------------------------------------
    with open(os.path.join(output_dir, "deployment_summary.json"), "w") as deployment_summary_file:
        json.dump(summary, deployment_summary_file, indent=2)

    # --------------------------------------------------
    # IO: Save image-level scores
    # --------------------------------------------------
    df = pd.DataFrame(image_scores)
    df.to_csv(
        os.path.join(output_dir, "image_scores.csv"),
        index=False,
    )

    # --------------------------------------------------
    # Print summary to console
    # --------------------------------------------------
    print(f"Images                : {num_images}")
    
    print("=" * 80)
    print("[Deployment results]")

    il = summary["image_level"]
    print(f"Predicted normal      : {il['predicted_normal']}")
    print(f"Predicted anomal      : {il['predicted_anomal']}")

    runtime = summary["runtime"]
    print(f"End-to-End time [s]   : {runtime['end_to_end_time_sec']:.2f}")
    print(f"End-to-End FPS        : {runtime['end_to_end_fps']:.2f}")
    print(f"End-to-End ms / image : {runtime['end_to_end_ms_per_image']:.1f}")
    print()

    print(f"All results saved under:")
    print(f"{output_dir}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
