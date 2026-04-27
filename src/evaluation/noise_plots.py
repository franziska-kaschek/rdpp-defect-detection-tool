import numpy as np
import matplotlib.pyplot as plt


def normalize_for_display(x):
    """
    Normalize image to [0, 1] for visualization. Ensures consistent contrast when displaying images with matplotlib.
    """
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def plot_noisy_examples_grid(
    img,
    img_noise,
    save_path,
    max_images=4,
    normalize=True,
):
    """
    Plot and save original vs noisy images in a grid.

    Layout:
    - top row: original images
    - bottom row: corresponding noisy images
    """

    # --------------------------------------------------
    # Determine how many images to show
    # --------------------------------------------------
    batch_size = img.shape[0]
    num_images = min(batch_size, max_images)

    # Safety check (should not happen, but keeps function robust)
    if num_images == 0:
        return

    # --------------------------------------------------
    # Create grid: 2 rows (original / noisy), N columns
    # --------------------------------------------------
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))

    # Handle edge case: matplotlib returns 1D axes if num_images == 1
    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # --------------------------------------------------
    # Iterate over images in batch
    # --------------------------------------------------
    for i in range(num_images):
        # Convert tensors to numpy arrays (H, W, C)
        img_np = img[i].detach().cpu().permute(1, 2, 0).numpy()
        noise_np = img_noise[i].detach().cpu().permute(1, 2, 0).numpy()

        # Optional normalization for correct visualization
        if normalize:
            img_np = normalize_for_display(img_np)
            noise_np = normalize_for_display(noise_np)

        # Plot original image (top row)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")

        # Plot corresponding noisy image (bottom row)
        axes[1, i].imshow(noise_np)
        axes[1, i].set_title(f"Noisy {i+1}")
        axes[1, i].axis("off")

    # --------------------------------------------------
    # Finalize and save figure
    # --------------------------------------------------
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()