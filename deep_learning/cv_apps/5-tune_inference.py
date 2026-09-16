#!/usr/bin/env python3
"""Tune inference parameters for YOLOv8 model."""
import os
import pandas as pd
from ultralytics import YOLO


def tune_inference(model, val_images_path,
                   conf_thresholds=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                   iou_thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
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
    if isinstance(model, str):
        model = YOLO(model)

    results_list = []

    # Grid search over conf × IoU
    for conf in conf_thresholds:
        for iou in iou_thresholds:
            print(f"Testing conf={conf}, iou={iou}")

            metrics = model.val(
                data=None,                 # use model's internal data config
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                split="val",
                save_json=False,
                verbose=False
            )

            # Extract metrics
            mAP50 = metrics.box.map50
            mAP5095 = metrics.box.map
            precision = metrics.box.mp
            recall = metrics.box.mr
            f1 = metrics.box.mf1

            results_list.append({
                "conf": conf,
                "iou": iou,
                "mAP50": mAP50,
                "mAP50-95": mAP5095,
                "precision": precision,
                "recall": recall,
                "F1": f1
            })

    # Convert to DataFrame
    df = pd.DataFrame(results_list)

    # Find best combination (maximize mAP50-95)
    best_row = df.loc[df["mAP50-95"].idxmax()]

    best_conf = float(best_row["conf"])
    best_iou = float(best_row["iou"])

    best_metrics = {
        "mAP50": float(best_row["mAP50"]),
        "mAP50-95": float(best_row["mAP50-95"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "F1": float(best_row["F1"])
    }

    return {
        "best_conf": best_conf,
        "best_iou": best_iou,
        "best_metrics": best_metrics,
        "all_results": df
    }
