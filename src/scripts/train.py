import sys
import os
import json
from datetime import datetime

import torch
import pandas as pd

from src.utils.cli import parse_cli_args
from src.utils.config import load_yaml_config, override
from src.utils.reproducibility import setup_seed
from src.training.training_pipeline import run_training


def main():
    """
    Script entry point.

    Resolves configuration, runs training, and saves results.
    """
    # --------------------------------------------------
    # Load config & args
    # --------------------------------------------------
    args = parse_cli_args(default_config="config_training.yaml")
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
    # Setup output directory (CLI > config)
    # --------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root = override(  
        args.save_folder,  
        os.path.join(  
            cfg["paths"]["output_root"],
            "training",
            timestamp,
        ),
    )
    os.makedirs(save_root, exist_ok=True)
    
    output_dir = os.path.join(save_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Define and record the concrete training run (resolved from YAML and CLI)   
    # --------------------------------------------------
    train_metadata = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": cfg["project"]["device"],
        "seed": cfg["project"]["seed"],
        "run": {
            "dataset": dataset_name,
        },
        "paths": {
            "dataset": dataset_path,
            "output_dir": output_dir,
        },
        "dataset": {
            "image_size": cfg["dataset"]["image_size"],
        },
        "model": {
            "backbone": cfg["model"]["backbone"],
            "base": cfg["model"]["backbones"][cfg["model"]["backbone"]]["base"],
        },
        "dataloader": {
            "batch_size": override(
                args.batch_size,
                cfg["dataloader"]["batch_size"],
            ),
            "shuffle_train": cfg["dataloader"]["shuffle_train"],
        },
        "training": {
            "epochs": override(
                args.epochs,
                cfg["training"]["epochs"],
            ),
            "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
            "optimizer": cfg["training"]["optimizer"],
            "loss": cfg["training"]["loss"],
            "pseudo_anomaly": cfg["training"]["pseudo_anomaly"],
        },
    }

    setup_seed(train_metadata["seed"])

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    print("=" * 80)
    print("[Training]")
    print(f"Dataset         : {dataset_name}")
    print(f"Backbone        : {train_metadata['model']['backbone']}")
    print("Training running...", end=" ", flush=True)

    result = run_training(dataset_path, train_metadata)

    # --------------------------------------------------
    # IO: Training metadata
    # --------------------------------------------------
    with open(os.path.join(output_dir, "train_metadata.json"), "w") as metadata_file:
        json.dump(result["train_metadata"], metadata_file, indent=2)

    # --------------------------------------------------
    # IO: Checkpoint
    # --------------------------------------------------
    checkpoint_name = cfg["paths"]["checkpoint_name"].format(  
        model=train_metadata["model"]["backbone"],
        dataset=dataset_name,  
    )
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    torch.save(result["checkpoint_state"], checkpoint_path)

    # --------------------------------------------------
    # IO: Best metrics
    # --------------------------------------------------
    with open(os.path.join(output_dir, "best_model.json"), "w") as best_model_file:
        json.dump(result["best"], best_model_file, indent=2)

    # --------------------------------------------------
    # IO: CSV history
    # --------------------------------------------------
    df = pd.DataFrame(result["eval_history"])
    df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    # --------------------------------------------------
    # Print summary to console
    # --------------------------------------------------
    print(f"Training images : {train_metadata['dataset']['num_train_images']}")
    print(f"Total epochs    : {train_metadata['training']['epochs']}")
    print(f"Batch size      : {train_metadata['dataloader']['batch_size']}")
    print("=" * 80)
    print("[Best model]")
    print(f"Best epoch      : {result['best']['epoch']}")
    print(f"AUROC (image)   : {result['best']['AUROC_sample']:.4f}")
    print(f"AUROC (pixel)   : {result['best']['AUROC_pixel']:.4f}")
    print(f"AUPRO (pixel)   : {result['best']['AUPRO_pixel']:.4f}")
    print()
    print("Best checkpoint saved at:")
    print(f"{checkpoint_path}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()