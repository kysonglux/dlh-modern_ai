#!/usr/bin/env python3
"""Train YOLOv8 with optional custom Albumentations augmentation."""
from ultralytics import YOLO


def train_with_augmentation(data, model_path="yolov8n.pt", augmentation=None,
                            custom_albu=None, epochs=100, imgsz=640, batch=16,
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
    if albumentations_transforms is not None:
        train_args["augmentations"] = albumentations_transforms
    elif yolo_aug_params is not None:
        train_args.update(yolo_aug_params)
    elif augmentation is False:
        train_args.update({"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
                           "degrees": 0.0, "translate": 0.0, "scale": 0.0,
                           "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
                           "fliplr": 0.0, "bgr": 0.0, "cutmix": 0.0,
                           "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
                           "erasing": 0.0, "auto_augment": None})

    results = model.train(**train_args)
    return model, results
