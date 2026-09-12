#!/usr/bin/env python3
"""YOLO-compatible augmentation using Albumentations."""
import albumentations as A
import numpy as np
import cv2
import random


def basic_aug(image, bboxes, labels):
    """
    Apply YOLO-compatible augmentation using Albumentations.
    Args:
        image (np.ndarray): Input image (H, W, C)
        bboxes (List[List[int]]): Pascal VOC bboxes [xmin, ymin, xmax, ymax]
        labels (List[int]): Class labels
    Returns:
        aug_img (np.ndarray): Augmented image
        aug_bboxes (np.ndarray): Augmented bboxes (VOC format)
        aug_labels (List[int]): Labels (unchanged)
    """
    random.seed(42)  # For reproducibility
    np.random.seed(42)

    transform = A.Compose(
        [
            A.PadIfNeeded(
                min_height=image.shape[0],
                min_width=image.shape[1],
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                translate_percent=0.1,
                scale=0.1,   # scale 0.1 means ±10%
                rotate=(-30, 0),
                fit_output=False,
                interpolation=cv2.INTER_NEAREST,
                p=0.5
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0
        ),
        seed=42
    )

    augmented = transform(image=image, bboxes=bboxes, labels=labels)

    aug_img = augmented["image"]
    aug_bboxes = np.array(augmented["bboxes"])
    aug_labels = augmented["labels"]

    return aug_img, aug_bboxes, aug_labels
