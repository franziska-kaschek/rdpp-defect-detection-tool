import os
import shutil
import numpy as np


def save_image_level_decision(
    image_path,
    predicted_label,
    save_root,
):
    """
    Save an image according to its image-level prediction.

    Images are grouped into folders based on the predicted label
    (e.g. "normal" / "anomal") to support qualitative inspection
    of image-level decisions.
    """
    target_dir = os.path.join(
        save_root,
        "image_level",
        predicted_label,
    )
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(image_path, target_dir)


def predict_image_label(anomaly_map, image_threshold, image_quantile):
    """
    Derive an image-level label from a pixel-level anomaly map.

    The image-level score is computed as a high quantile of the anomaly map
    to reduce sensitivity to isolated noisy pixels. This score is then
    compared against a fixed image-level threshold.
    """
    image_score = np.quantile(anomaly_map, image_quantile)
    
    return "anomal" if image_score >= image_threshold else "normal"

