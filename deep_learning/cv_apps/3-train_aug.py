#!/usr/bin/env python3
"""Train YOLOv8 with optional custom Albumentations augmentation."""
from ultralytics import YOLO
import albumentations as A
import numpy as np


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
    yolo = YOLO(model_path)

    # ---------------------------------------------------------
    # 1. Albumentations custom augmentation pipeline
    # ---------------------------------------------------------
    if custom_albu:
        # Build Albumentations pipeline
        albu_pipeline = A.Compose(
            custom_albu,
            bbox_params=A.BboxParams(
                format="yolo",  # YOLO expects normalized xywh
                label_fields=["cls"]
            ),
            seed=42
        )

        # Albumentations callback
        def albu_callback(data):
            imgs = data["img"]
            bboxes = data["bboxes"]
            cls = data["cls"]

            new_imgs, new_bboxes, new_cls = [], [], []

            for img, bbox_list, cls_list in zip(imgs, bboxes, cls):
                # Convert CHW → HWC
                img = img.transpose(1, 2, 0)

                # Albumentations expects uint8
                img = (img * 255).astype(np.uint8)

                augmented = albu_pipeline(image=img,
                                          bboxes=bbox_list, cls=cls_list)

                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_cls = augmented["cls"]

                # Convert back HWC → CHW and normalize
                aug_img = aug_img.astype(np.float32) / 255.0
                aug_img = aug_img.transpose(2, 0, 1)

                new_imgs.append(aug_img)
                new_bboxes.append(aug_bboxes)
                new_cls.append(aug_cls)

            data["img"] = np.stack(new_imgs)
            data["bboxes"] = new_bboxes
            data["cls"] = new_cls

        # Register callback
        yolo.add_callback("on_preprocess_batch", albu_callback)

    # ---------------------------------------------------------
    # 2. YOLO native augmentation override
    # ---------------------------------------------------------
    if yolo_aug_params and aug:
        yolo.overrides.update(yolo_aug_params)

    # Disable YOLO augmentation if aug=False
    if not aug:
        yolo.overrides.update({"augment": False})

    # ---------------------------------------------------------
    # 3. Train YOLO
    # ---------------------------------------------------------
    results = yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        save=save,
        plots=plots,
        verbose=verbose
    )

    return yolo, results
