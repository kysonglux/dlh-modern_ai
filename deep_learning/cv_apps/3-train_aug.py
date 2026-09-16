#!/usr/bin/env python3
"""Train YOLOv8 with optional custom Albumentations augmentation."""
import albumentations as A
import numpy as np
from ultralytics import YOLO


def train_with_augmentation(data, model_path="yolov8n.pt", aug=None,
                            custom_albu=None, epochs=50, imgsz=640, batch=16,
                            albumentations_transforms=None,
                            yolo_aug_params=None,
                            save=True, plots=True, verbose=True):
    """
    Train YOLO with optional custom Albumentations augmentation.

    Args:
        data_yaml (str): Path to dataset YAML.
        model (str): YOLO model(yolov8n.pt) or weights.
        aug (bool): Enable/disable augmentation.
        custom_albu (list): Custom Albumentations transforms.
        epochs (int): Training epochs.
        imgsz (int or tuple): Image size.
        batch (int): Batch size.
        optional
        albumentations_transforms (list):
        yolo_aug_params (dict): override.
        save (bool): Save checkpoints.
        plots (bool): Save training plots.
        verbose (bool): Print training logs.

    Returns:
        model: Trained YOLO model.
        results: Full training output.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Base training arguments
    train_args = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }

    # Add YOLO augmentation overrides
    if yolo_aug_params:
        train_args.update(yolo_aug_params)

    results = model.train(**train_args)
    return model, results
