import random
import numpy as np
import torch


def setup_seed(seed: int) -> None:
    """
    Sets a fixed random seed to ensure reproducible results.
    Same settings should lead to similar results across runs.
    """

    # Set random seed for PyTorch operations on CPU
    torch.manual_seed(seed)

    # Set random seed for all available GPUs
    torch.cuda.manual_seed_all(seed)

    # Set random seed for NumPy random operations
    np.random.seed(seed)

    # Set random seed for Python's built-in random module
    random.seed(seed)

    # Enforce deterministic GPU operations
    torch.backends.cudnn.deterministic = True

    # Disable CuDNN benchmarking to avoid varying results
    # caused by automatic algorithm selection
    torch.backends.cudnn.benchmark = False