#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
basic_aug = __import__('1-basic_aug').basic_aug

def visualize_boxes(image, boxes, labels=None, title="Image"):
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if labels:
            cv2.putText(img, labels[i], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()



image_path = "datasets/detection/images/train/2008_000008.jpg"
label_path = "datasets/detection/labels/train/2008_000008.txt"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YOLO label → Pascal VOC
h, w, _ = image.shape
bboxes = []
labels = []

with open(label_path) as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        bboxes.append([x1, y1, x2, y2])
        labels.append(str(int(cls)))

aug_img, aug_boxes, aug_labels = basic_aug(image, bboxes, labels)

visualize_boxes(image, bboxes, labels, "Original")
visualize_boxes(aug_img, aug_boxes, aug_labels, "Augmented")
