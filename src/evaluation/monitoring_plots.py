import os
import matplotlib.pyplot as plt


def plot_training_progress(eval_history, best_epoch, out_dir):
    """
    Plot training and evaluation metrics over epochs.

    Args:
        eval_history (dict): Dictionary containing per-epoch metrics
        out_dir (str): Output directory for saving the plot
        best_epoch (int): Best epoch (1-based index)
    """

    epochs = range(1, len(next(iter(eval_history.values()))) + 1)  # <-- angepasst

    best_epoch_idx = best_epoch 

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    
    # --------------------------------------------------
    # AUROC / AUPRO metrics
    # --------------------------------------------------
    ax[0][0].plot(epochs, eval_history["pixel_auroc"])
    ax[0][0].set_title("AUROC (pixel)")

    ax[0][1].plot(epochs, eval_history["image_auroc"])
    ax[0][1].set_title("AUROC (image)")

    ax[0][2].plot(epochs, eval_history["aupro"])
    ax[0][2].set_title("AUPRO (pixel)")

    # --------------------------------------------------
    # Loss curves
    # --------------------------------------------------
    ax[1][0].plot(epochs, eval_history["loss_proj"])
    ax[1][0].set_title("Projection loss")

    ax[1][1].plot(epochs, eval_history["loss_distill"])
    ax[1][1].set_title("Distillation loss")

    ax[1][2].plot(epochs, eval_history["total_loss"])
    ax[1][2].set_title("Total loss")

    # --------------------------------------------------
    # Mark best epoch on all subplots
    # --------------------------------------------------
    for row in ax:
        for axis in row:
            axis.axvline(
                best_epoch_idx,
                linestyle="--",
                color="red",
                linewidth=1,
            )
            axis.set_xticks(list(epochs)) 

    ax[0][0].axvline(
        best_epoch_idx,
        linestyle="--",
        color="red",
        linewidth=1,
        label="Best epoch (avg AUROC/AUPRO)",
    )
    ax[0][0].legend()

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=100)
    plt.close(fig)
