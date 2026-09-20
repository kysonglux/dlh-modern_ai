#!/usr/bin/env python3
"""Two-phase hyperparameter tuning and training for YOLOv8."""
from ultralytics import YOLO
import os
import shutil


def tune_hyperparameters():
    """
    Performs two-phase hyperparameter tuning and training:
    Phase 1: Lightweight hyperparameter search
    Phase 2: Continue training from best checkpoint
    """

    # -----------------------------
    # Phase 1 — Lightweight tuning
    # -----------------------------

    model = YOLO("yolov8n.pt")   # base model f

    data_yaml = "datasets/detection/data.yaml"

    model.tune(
        data=data_yaml,
        epochs=10,                # short trials
        iterations=20,            # number of hyperparameter configs
        optimizer="auto",         # let YOLO explore optimizers
        plots=False,
        save=True,
        verbose=False
    )

    # -----------------------------
    # Load best hyperparameters
    # -----------------------------
    best_hyp_path = "runs/detect/tune/best_hyperparameters.yaml"
    best_weights = "runs/detect/tune/weights/best.pt"

    if not os.path.exists(best_hyp_path):
        raise FileNotFoundError("best_hyperparameters.yaml not found.")

    if not os.path.exists(best_weights):
        raise FileNotFoundError("best.pt not found — tuning may have failed.")

    # -----------------------------
    # Phase 2 — Full training
    # -----------------------------
    final_model = YOLO(best_weights)

    results = final_model.train(
        data=data_yaml,
        epochs=150,               # full convergence
        imgsz=640,
        batch=16,
        patience=20,              # early stopping
        cfg=best_hyp_path,        # load tuned hyperparameters
        plots=False,
        verbose=False,
    )

    # -----------------------------
    # Save final model as best_model.pt
    # -----------------------------
    trained_best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    if not os.path.exists(trained_best_weights):
        raise FileNotFoundError(f"Expected trained weights at"
                                f"{trained_best_weights} but none found.")
    shutil.copy(trained_best_weights, "best_model.pt")


"""
if __name__ == "__main__":
    tune_hyperparameters()
"""
