import csv
import os


class ImageScoreLogger:
    """
    Logs image-level anomaly scores to a CSV file.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._init_file()

    def _init_file(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_name",
                "category",
                "gt_label",
                "predicted_label",
                "anomaly_score",
            ])

    def log(
        self,
        image_name,
        category,
        gt_label,
        predicted_label,
        anomaly_score,
    ):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                image_name,
                category,
                gt_label,
                predicted_label,
                anomaly_score,
            ])
