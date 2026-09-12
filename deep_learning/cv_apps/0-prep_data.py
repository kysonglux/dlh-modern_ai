#!/usr/bin/env python3
"""prepare VOC2012 dataset for YOLOv8 training."""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------
# CONFIG — adjust only if your folder changes
# ---------------------------------------------------------
VOC_ROOT = "VOC2012"
IMAGESETS_MAIN = f"{VOC_ROOT}/ImageSets/Main"

OUTPUT_ROOT = "datasets/detection"

CLASSES = ["person", "car", "bicycle"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

# ---------------------------------------------------------
# STEP 1 — Build train_samples.txt and val_samples.txt
# ---------------------------------------------------------


def extract_ids(split):
    """extract image IDs for a given split (train/val)"""
    ids = set()
    for cls in CLASSES:
        file_path = os.path.join(IMAGESETS_MAIN, f"{cls}_{split}.txt")
        if not os.path.exists(file_path):
            print(f"[WARN] Missing: {file_path}")
            continue

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                img_id, label = parts
                if label == "1":  # keep only images containing the class
                    ids.add(img_id)
    return ids


print("Extracting train/val image IDs...")
train_ids = extract_ids("train")
val_ids = extract_ids("val")

with open("train_samples.txt", "w") as f:
    f.write("\n".join(sorted(train_ids)))

with open("val_samples.txt", "w") as f:
    f.write("\n".join(sorted(val_ids)))

print(f"train_samples.txt: {len(train_ids)} images")
print(f"val_samples.txt:   {len(val_ids)} images")

# ---------------------------------------------------------
# STEP 2 — VOC → YOLO conversion
# ---------------------------------------------------------


def voc_to_yolo_bbox(size, box):
    """converts Pascal VOC bbox format to YOLO format."""
    w_img, h_img = size
    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0 / w_img
    y_center = (ymin + ymax) / 2.0 / h_img
    w = (xmax - xmin) / w_img
    h = (ymax - ymin) / h_img

    return x_center, y_center, w, h


def process_split(split_name, image_list):
    """split and convert the images and labels to YOLO format."""
    images_out = Path(OUTPUT_ROOT) / "images" / split_name
    labels_out = Path(OUTPUT_ROOT) / "labels" / split_name

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_id in image_list:
        xml_path = Path(VOC_ROOT) / "Annotations" / f"{img_id}.xml"
        img_path = Path(VOC_ROOT) / "JPEGImages" / f"{img_id}.jpg"

        if not xml_path.exists() or not img_path.exists():
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w_img = int(size.find("width").text)
        h_img = int(size.find("height").text)

        yolo_lines = []

        for obj in root.findall("object"):
            cls = obj.find("name").text.lower()

            if cls not in CLASS_TO_ID:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            x_center, y_center, w, h = voc_to_yolo_bbox(
                (w_img, h_img), (xmin, ymin, xmax, ymax)
            )

            yolo_lines.append(
                f"{CLASS_TO_ID[cls]}"
                f"{x_center:.6f}{y_center:.6f} {w:.6f} {h:.6f}"
            )

        if len(yolo_lines) == 0:
            continue

        shutil.copy(img_path, images_out / f"{img_id}.jpg")

        with open(labels_out / f"{img_id}.txt", "w") as f:
            f.write("\n".join(yolo_lines))


print("Converting VOC → YOLO format...")
process_split("train", train_ids)
process_split("val", val_ids)

# ---------------------------------------------------------
# STEP 3 — Write data.yaml
# ---------------------------------------------------------

yaml_path = Path(OUTPUT_ROOT) / "data.yaml"
yaml_content = """path: datasets/detection/
train: images/train
val: images/val

nc: 3
names: ["person", "car", "bicycle"]
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("Done! YOLOv8 dataset created successfully.")
