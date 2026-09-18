#!/usr/bin/env python3
"""Tune inference parameters for YOLOv8 model."""
from ultralytics import YOLO


def tune_inference(data_yaml, model_path,
                   conf_list=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                   iou_list=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                   imgsz=640):
    """
    Performs inference parameter tuning
    to find optimal confidence and IoU thresholds.

    Args:
        model (str or YOLO): Path to trained model or YOLO object.
        val_images_path (str): Path to validation images directory.
        conf_thresholds (List[float]): Confidence thresholds to test.
        iou_thresholds (List[float]): IoU thresholds for NMS.
        imgsz (int): Inference image size.

    Returns:
        dict: {
            "best_conf": float,
            "best_iou": float,
            "best_metrics": {...},
            "all_results": DataFrame
        }
    """

# Load model if path is given
    model = YOLO(model_path)

    results = []

    # Grid search over conf × IoU
    for conf in conf_list:
        for iou in iou_list:

            metrics = model.val(
                data=data_yaml,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                split="val",
                save_json=False,
                verbose=False
            )

            results.append({
                "conf": conf,
                "iou": iou,
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "F1": float(metrics.box.mf1)
            })

    return results
