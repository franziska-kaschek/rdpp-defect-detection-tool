"""
Helper script. Convert LabelMe JSON annotations to binary mask images.

Usage:
    python3 -m src.scripts.labelme_to_mask --input <json_dir> --output <mask_dir>

Example:
    python3 -m src.scripts.labelme_to_mask \
        --input datasets/labelme_labels \
        --output datasets/ground_truth
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


def create_mask(data):
    """
    Create a binary mask from LabelMe JSON data.
    All annotated regions are filled with value 255 (foreground).
    """
    # Create empty grayscale mask (0 = background)
    mask = Image.new("L", (data["imageWidth"], data["imageHeight"]), 0)
    draw = ImageDraw.Draw(mask)

    # Draw all annotated shapes
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")

        # Skip invalid annotations
        if not points:
            continue

        # Rectangle defined by two corner points
        if shape_type == "rectangle" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[:2]
            draw.rectangle([round(x1), round(y1), round(x2), round(y2)], fill=255)

        # Polygon defined by at least three points
        elif shape_type == "polygon" and len(points) >= 3:
            polygon_points = [(round(x), round(y)) for x, y in points]
            draw.polygon(polygon_points, fill=255)

        # Ignore unsupported shape types
        else:
            continue

    return mask


def process_file(json_file, output_dir):
    """
    Load a JSON file, generate its mask, and save it as PNG.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mask = create_mask(data)

    # Save mask with same name as input file
    mask.save(output_dir / (json_file.stem + ".png"))


def main(input_dir, output_dir):
    """
    Process all JSON files in the input directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {input_dir}")
        return

    # Process all files with progress bar
    for json_file in tqdm(json_files, desc="Generating masks"):
        process_file(json_file, output_dir)

    print(f"{len(json_files)} masks created!")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to LabelMe JSON folder")
    parser.add_argument("--output", required=True, help="Output folder for binary masks")
    args = parser.parse_args()

    main(args.input, args.output)