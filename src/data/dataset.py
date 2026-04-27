import os
import glob

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.data.noise import Simplex_CLASS


class ToTensor(object):
    """
    Convert a NumPy image array (H, W, C) to a PyTorch tensor (C, H, W).

    Assumes input images are already normalized to [0, 1].
    """
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print(
                "Invalid_transpose, please make sure images have shape (H, W, C) before transposing"
            )
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    """
    Normalize image channels using ImageNet statistics.

    This is required when using ImageNet-pretrained backbones
    such as ResNet variants.
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


def get_data_transforms(image_size):
    """
    Create transformation pipelines for input images and ground truth masks.
    """
    # resizing is handled explicitly via cv2.resize in the dataset
    data_transforms = transforms.Compose([Normalize(), ToTensor()])
    gt_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    return data_transforms, gt_transforms


class AnomalyTrainDataset(torch.utils.data.Dataset):
    """
    Dataset used for training anomaly detection models.

    Synthetic anomalies are generated on-the-fly using Simplex noise
    and injected into normal images.
    """
    def __init__(self, dataset_path, transform, image_size, pseudo_anomaly_cfg):
        self.img_path = os.path.join(dataset_path, "train")
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        self.image_size = image_size
        self.img_paths = self.load_dataset()

        # -------------------------------
        # Store pseudo anomaly config
        # -------------------------------
        patch_cfg = pseudo_anomaly_cfg["patch"]
        simplex_cfg = pseudo_anomaly_cfg["simplex"]

        self.noise_enabled = pseudo_anomaly_cfg.get("enabled", True)

        self.patch_min_size = patch_cfg["min_size"]
        self.patch_max_size_divisor = patch_cfg["max_size_divisor"]

        self.simplex_octaves = simplex_cfg["octaves"]
        self.simplex_persistence = simplex_cfg["persistence"]
        self.simplex_frequency = simplex_cfg["frequency"]
        self.simplex_amplitude = simplex_cfg["amplitude"]

    def load_dataset(self):
        """
        Recursively collect all training images (PNG format).
        """
        return glob.glob(
            os.path.join(self.img_path, "**", "*.png"), recursive=True
        ) + glob.glob(os.path.join(self.img_path, "**", "*.PNG"), recursive=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # --------------------------------------------------
        # Load and preprocess input image
        # --------------------------------------------------
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255.0, (self.image_size, self.image_size))
        img_normal = self.transform(img)

        # No-Noise Fall-Case
        if not self.noise_enabled:
            return img_normal, img_normal.clone(), os.path.basename(img_path)

        # --------------------------------------------------
        # Generate synthetic anomaly using Simplex noise
        # --------------------------------------------------
        image_size = self.image_size
        
        # Random anomaly patch size
        max_patch = max(self.patch_min_size + 1, int(image_size // self.patch_max_size_divisor))
        h_noise = np.random.randint(self.patch_min_size, max_patch)
        w_noise = np.random.randint(self.patch_min_size, max_patch)
        
        # Random anomaly position
        start_h_noise = np.random.randint(1, image_size - h_noise)
        start_w_noise = np.random.randint(1, image_size - w_noise)
        
        noise_size = (h_noise, w_noise)
        
        # Generate 3-channel Simplex noise
        simplex_noise = self.simplexNoise.rand_3d_octaves(
            (3, *noise_size),
            self.simplex_octaves,
            self.simplex_persistence,
            self.simplex_frequency,
        )
        
        # Inject anomaly into an otherwise zero mask
        init_zero = np.zeros((image_size, image_size, 3))
        init_zero[
            start_h_noise : start_h_noise + h_noise,
            start_w_noise : start_w_noise + w_noise,
            :,
        ] = self.simplex_amplitude * simplex_noise.transpose(1, 2, 0)
        
        # Create anomalous image        
        img_noise = img + init_zero
        img_noise = self.transform(img_noise)

        return img_normal, img_noise, os.path.basename(img_path)


class AnomalyTestDataset(torch.utils.data.Dataset):
    """
    Dataset used for evaluation with ground truth annotations.

    Supports both normal ("good") samples and anomalous samples
    with pixel-level ground truth masks.
    """
    def __init__(self, dataset_path, transform, gt_transform, image_size):
        self.img_path = os.path.join(dataset_path, "test")
        self.gt_path = os.path.join(dataset_path, "ground_truth")
        self.transform = transform
        self.gt_transform = gt_transform
        self.image_size = image_size

        self.img_paths, self.gt_paths, self.labels, self.categories = self.load_dataset()

    def load_dataset(self):
        """
        Load image paths, corresponding ground truth masks,
        image-level labels and defect categories.
        """
        tot_img_paths = []
        tot_gt_paths = []
        tot_labels = []
        tot_categories = []
        
        root_categories = os.listdir(self.img_path)

        for root_category in root_categories:
            img_root = os.path.join(self.img_path, root_category)

            # Recursively collect all test images
            img_paths = glob.glob(
                os.path.join(img_root, "**", "*.png"), recursive=True
            ) + glob.glob(os.path.join(img_root, "**", "*.PNG"), recursive=True)

            if root_category == "good":
                # Normal samples have no ground truth mask
                for img_path in img_paths:
                    category = os.path.basename(os.path.dirname(img_path))
                    tot_img_paths.append(img_path)
                    tot_gt_paths.append(0)
                    tot_labels.append(0)
                    tot_categories.append(category)
            else:
                # Anomalous samples require ground truth masks
                gt_root = os.path.join(self.gt_path, root_category)

                for img_path in img_paths:
                    category = os.path.basename(os.path.dirname(img_path))
                    rel_path = os.path.relpath(img_path, img_root)
                    gt_path = os.path.join(gt_root, rel_path)

                    if not os.path.exists(gt_path):
                        raise FileNotFoundError(
                            f"GT not found for image:\n{img_path}\nExpected:\n{gt_path}"
                        )

                    tot_img_paths.append(img_path)
                    tot_gt_paths.append(gt_path)
                    tot_labels.append(1)
                    tot_categories.append(category)

        return tot_img_paths, tot_gt_paths, tot_labels, tot_categories

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        category = self.categories[idx]
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255.0, (self.image_size, self.image_size))
        img = self.transform(img)

        # Load or create ground truth mask
        if gt == 0:
            gt = torch.zeros([1, self.image_size, self.image_size])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        return img, gt, label, category, os.path.basename(img_path)


class AnomalyDeploymentDataset(torch.utils.data.Dataset):
    """
    Dataset used for deployment / inference.

    No ground truth is required. Images are grouped by category
    to support per-class evaluation or reporting.
    """
    def __init__(self, dataset_path, transform, image_size):
        self.img_path = os.path.join(dataset_path, "test")        
        self.transform = transform
        self.image_size = image_size
        self.img_paths, self.categories = self.load_dataset()

    def load_dataset(self):
        """
        Recursively collect all test images and their categories.
        """
        tot_img_paths = []
        tot_categories = []

        root_categories = os.listdir(self.img_path)

        for root_category in root_categories:
            img_root = os.path.join(self.img_path, root_category)
            # Recursively collect all test images
            img_paths = glob.glob(
                os.path.join(img_root, "**", "*.png"), recursive=True
            ) + glob.glob(os.path.join(img_root, "**", "*.PNG"), recursive=True)

            for img_path in img_paths:
                category = os.path.basename(os.path.dirname(img_path))
                tot_img_paths.append(img_path)
                tot_categories.append(category)

        return tot_img_paths, tot_categories

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        category = self.categories[idx]

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255.0, (self.image_size, self.image_size))
        img = self.transform(img)

        return img, category, os.path.basename(img_path)