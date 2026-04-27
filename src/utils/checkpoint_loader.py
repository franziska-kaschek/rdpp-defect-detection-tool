import torch

def safe_torch_load(checkpoint_path, device):
    """
    Safely loads a PyTorch checkpoint in a version-compatible way.

    It first tries to load only the model weights (supported in newer PyTorch versions).
    If this fails due to an unsupported argument, it falls back to standard loading
    for older PyTorch versions.
    """
    try:
        # For newer PyTorch versions (supports weights_only)
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # For older PyTorch versions (weights_only not supported)
        return torch.load(checkpoint_path, map_location=device)