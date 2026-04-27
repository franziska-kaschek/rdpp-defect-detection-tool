import os
import shutil


def save_confusion_case_split(
    image_path,
    case,
    save_root,
    category,
):
    """
    Save an image into a folder structure based on its confusion matrix case.

    This is used for qualitative error analysis by grouping images into
    TP / FP / TN / FN folders per category.
    """
    out_dir = os.path.join(
        save_root,
        "error_analysis",
        category,
        case,
    )
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy(
        image_path,
        os.path.join(out_dir, os.path.basename(image_path)),
    )


def get_confusion_case(pred_label, gt_label):
    """
    Determine the confusion matrix case for a single prediction.
    """
    if pred_label == "anomal" and gt_label == 1:
        return "TP"
    elif pred_label == "anomal" and gt_label == 0:
        return "FP"
    elif pred_label == "normal" and gt_label == 0:
        return "TN"
    else:
        return "FN"
