from argparse import ArgumentParser


def parse_cli_args(default_config=None):
    """
    Central CLI argument definition.

    All scripts may use a subset of these arguments.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config filename in `configs/` (e.g. `config_training.yaml`)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path (e.g. ./datasets/shearography)",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="Override output directory (e.g. `./output` or `/home/user/experiments`)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="[TRAINING ONLY] Override training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="[TRAINING ONLY] Override number of training epochs"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="[TESTING / DEPLOYMENT ONLY] Override path to trained model checkpoint (e.g. `./output/training/2026-03-30_16-00-21/shearography/best_model_resnet34_shearography.pth`)",
    )

    args = parser.parse_args()

    if args.config is None and default_config is not None:
        args.config = default_config

    return args
