import os
import json
import cv2
import numpy as np


# ============================================================
# Image normalization & mask utilities
# ============================================================
def normalize_to_uint8(anomaly_map: np.ndarray) -> np.ndarray:
    """
    Normalize anomaly map to uint8 [0, 255].
    """
    amap_norm = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )
    
    return (amap_norm * 255).astype(np.uint8)


def create_binary_mask(anomaly_map: np.ndarray, pixel_threshold: float) -> np.ndarray:
    """
    Convert a continuous anomaly map into a binary mask by thresholding:
    pixels with anomaly score >= pixel_threshold are marked as anomalous (255),
    all others as normal (0).
    """
    mask = np.zeros_like(anomaly_map, dtype=np.uint8)
    mask[anomaly_map >= pixel_threshold] = 255

    return mask


def resize_mask_to_image(mask: np.ndarray, image_shape) -> np.ndarray:
    """
    Resize binary mask to original image resolution.
    """
    h, w = image_shape[:2]
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (resized > 0).astype(np.uint8) * 255

def cleanup_binary_mask(
    mask: np.ndarray,
    enabled: bool,
    kernel_size: int,
    open_iterations: int,
    close_iterations: int,
) -> np.ndarray:
    """
    Remove small noise and fill small holes in a binary mask
    using morphological opening and closing.
    """

    if not enabled:
        return mask

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    if open_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=open_iterations,
        )

    if close_iterations > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )

    return mask


# ============================================================
# Visualization utilities (heatmaps & overlays)
# ============================================================
def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a JET heatmap of the anomaly map onto the original image.
    """
    # Normalize anomaly map to [0, 255]
    amap_uint8 = normalize_to_uint8(anomaly_map)

    # Apply JET colormap (low resolution)
    heatmap = cv2.applyColorMap(amap_uint8, cv2.COLORMAP_JET)

    # Resize heatmap to original image resolution
    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Alpha blend heatmap and original image
    return cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple = (0, 0, 255),  # red (BGR)
) -> np.ndarray:
    """
    Overlay binary mask on original image.
    """
    overlay = image.copy()
    overlay[mask == 255] = color

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


# ============================================================
# Contour extraction, filtering & polygon generation
# ============================================================
def extract_valid_contours(binary_mask: np.ndarray, min_region_area: int):
    """
    Extract contours from a binary mask and keep only regions
    that satisfy the minimum area threshold.
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = []

    # Iterate over all detected contours
    for cnt in contours:
        # Keep only contours that meet the minimum area requirement
        if cv2.contourArea(cnt) >= min_region_area:
            valid_contours.append(cnt)

    return valid_contours


def contours_to_mask(valid_contours, mask_shape) -> np.ndarray:
    """
    Render filtered contours into a binary mask.
    """
    filtered_mask = np.zeros(mask_shape, dtype=np.uint8)

    for cnt in valid_contours:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=-1)

    return filtered_mask


def contours_to_polygons(valid_contours, epsilon_ratio: float):
    """
    Convert filtered contours into polygon point lists.
    """
    polygons = []

    for cnt in valid_contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * perimeter, True)
        polygons.append(approx.squeeze(1).tolist())

    return polygons


def save_labelme_json(
    polygons,
    image_filename,
    image_height,
    image_width,
    save_path,
    label="anomaly",
):
    """
    Save polygons in LabelMe JSON format.
    """
    # Create a list of polygon annotation entries for the JSON file
    shapes = [
        {
            "label": label,
            "points": poly,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        }
        for poly in polygons
    ]

    # Build the complete LabelMe JSON structure including image metadata and annotations
    data = {
        "version": "5.8.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }

    # Write the JSON annotation file to disk
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# Orchestrates all pixel-level outputs for one image
# ============================================================
def save_pixel_level_outputs(
    anomaly_map: np.ndarray,
    pixel_threshold: float,
    min_region_area: int,
    morphology_enabled: bool,
    morphology_kernel_size: int,
    morphology_open_iterations: int,
    morphology_close_iterations: int,
    original_image: np.ndarray,
    save_root: str,
    filename: str,
    category: str,
    polygon_epsilon_ratio: float = 0.02,
):
    """
    Generate and save all pixel-level outputs for a single image.

    Includes:
        - heatmap overlay
        - binary mask overlay
        - polygon annotations (LabelMe)

    Logic:
        - Create binary mask from anomaly map
        - Optionally clean mask with morphology
        - Extract and filter contours once
        - Reuse contours for binary mask and polygon generation
    """
    pixel_root = os.path.join(save_root, "pixel_level")

    # --------------------------------------------------
    # Step 1: Heatmap overlay generation
    # --------------------------------------------------
    heatmap_overlay_dir = os.path.join(pixel_root, "heatmap_overlays", category)
    os.makedirs(heatmap_overlay_dir, exist_ok=True)

    heatmap_overlay = overlay_heatmap(original_image, anomaly_map)
    cv2.imwrite(os.path.join(heatmap_overlay_dir, filename), heatmap_overlay)

    # --------------------------------------------------
    # Step 2: Initial binary mask (threshold-based, includes noise)
    # --------------------------------------------------
    binary_mask = create_binary_mask(anomaly_map, pixel_threshold)

    overlay_dir = os.path.join(pixel_root, "binary_overlays", category)
    os.makedirs(overlay_dir, exist_ok=True)

    # No anomalous pixels at all
    if not np.any(binary_mask):
        cv2.imwrite(os.path.join(overlay_dir, filename), original_image)
        return

    # Resize mask to image resolution
    resized_binary_mask = resize_mask_to_image(binary_mask, original_image.shape)

    # --------------------------------------------------
    # Step 3: Optional mask refinement (morphology)
    # --------------------------------------------------
    if morphology_enabled:
        resized_binary_mask = cleanup_binary_mask(
            resized_binary_mask,
            enabled=True,
            kernel_size=morphology_kernel_size,
            open_iterations=morphology_open_iterations,
            close_iterations=morphology_close_iterations,
        )

    # --------------------------------------------------
    # Step 4: Extract and filter valid anomaly regions (contours)    
    # --------------------------------------------------
    valid_contours = extract_valid_contours(
        resized_binary_mask,
        min_region_area,
    )

    # No valid anomaly regions found → save original image and skip processing
    if not valid_contours:
        cv2.imwrite(os.path.join(overlay_dir, filename), original_image)
        return

    # --------------------------------------------------
    # Step 5: Final binary mask (contour-filtered, noise removed)
    # --------------------------------------------------
    filtered_mask = contours_to_mask(
        valid_contours,
        resized_binary_mask.shape,
    )

    # --------------------------------------------------
    # Step 6: Polygon generation from filtered contours
    # --------------------------------------------------
    polygons = contours_to_polygons(
        valid_contours,
        polygon_epsilon_ratio,
    )

    # --------------------------------------------------
    # Step 7: Mask overlay visualization
    # --------------------------------------------------
    overlay = overlay_mask(original_image, filtered_mask)
    cv2.imwrite(os.path.join(overlay_dir, filename), overlay)

    # --------------------------------------------------
    # Step 8: Annotation export (LabelMe)
    # --------------------------------------------------
    json_dir = os.path.join(pixel_root, "polygons", category)
    os.makedirs(json_dir, exist_ok=True)

    save_labelme_json(
        polygons=polygons,
        image_filename=filename,
        image_height=original_image.shape[0],
        image_width=original_image.shape[1],
        save_path=os.path.join(json_dir, filename.replace(".png", ".json")),
    )
