#!/usr/bin/env python3
"""Custom YOLO-compatible augmentation using Albumentations."""
import albumentations as A
import numpy as np


def custom_aug(image, bboxes, labels):
    """
    Apply YOLO‑compatible Albumentations‑exclusive augmentation.

    Args:
        image (np.ndarray): Input image (H, W, C)
        bboxes (List[List[int]]): Pascal VOC bboxes [xmin, ymin, xmax, ymax]
        labels (List[int]): Class labels

    Returns:
        aug_img (np.ndarray): Augmented image
        aug_bboxes (np.ndarray): Augmented bboxes (VOC format)
        aug_labels (List[int]): Labels (unchanged)
    """

    transform = A.Compose(
        [
            # Motion blur
            A.MotionBlur(blur_limit=5, p=0.9),

            # One-of block: elastic or optical distortion
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1,
                                       sigma=50, alpha_affine=0, p=0.2),
                    A.OpticalDistortion(distort_limit=0.05,
                                        shift_limit=0.0, p=0.2),
                ],
                p=0.9  # OneOf block applied with p=0.9
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
