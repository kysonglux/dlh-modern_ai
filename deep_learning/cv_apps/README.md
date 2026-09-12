# Object Detection & Segmentation — Learning Objectives Q&A

## 1. What is object detection?

Object detection is a computer vision task that involves both **locating** and **classifying** objects within an image. Unlike simple image classification (which assigns a single label to an entire image), object detection outputs, for each object present:

- A **bounding box** (the coordinates of a rectangle tightly enclosing the object).
- A **class label** (what the object is, e.g., "dog", "car").
- Usually a **confidence score** (how certain the model is about that detection).

An image can contain multiple objects of multiple classes, and object detection must find all of them, distinguishing "there's a dog and a car in this image, here's where each one is" rather than just "this image contains a dog."

## 2. What is a single-shot detector?

A **single-shot detector (SSD)** is a category of object detection architecture that predicts bounding boxes and class labels in a **single forward pass** through the network, rather than using a two-stage pipeline.

This contrasts with two-stage detectors (like Faster R-CNN), which first generate candidate regions ("region proposals") in one stage and then classify/refine those regions in a second stage. Single-shot detectors instead directly predict a dense set of bounding boxes (typically based on a grid of predefined anchor boxes at multiple scales) and their associated class probabilities in one pass.

Key characteristics of single-shot detectors:

- **Faster inference** — no separate region proposal step, making them well-suited to real-time applications.
- **Simpler end-to-end training** — the whole pipeline is one network trained jointly.
- Often a small trade-off in accuracy for small objects compared to two-stage detectors, though modern single-shot architectures (like YOLO variants) have largely closed this gap.

Examples of single-shot detectors: **SSD** (the original architecture bearing this name) and **YOLO**.

## 3. What is the YOLO algorithm?

**YOLO (You Only Look Once)** is a single-shot object detection algorithm that frames detection as a single regression problem: directly predicting bounding boxes and class probabilities from the full image in one evaluation of the network.

At a high level, YOLO works by:

1. Dividing the input image into an **S × S grid**.
2. For each grid cell, predicting a fixed number of bounding boxes, each with:
    - Coordinates (center x, center y, width, height) relative to the cell.
    - An **objectness score** (confidence that an object exists in that box).
    - Class probabilities for what the object is.
3. Combining these predictions across the whole grid and applying **non-max suppression** to remove duplicate/overlapping detections, leaving a final set of bounding boxes.

Because the entire image is processed in one pass, YOLO is extremely fast, which made it popular for real-time detection tasks (e.g., video, robotics, autonomous driving). Over its many versions (YOLOv1 through modern releases like YOLOv8/v11), it has improved accuracy while retaining its core single-pass, grid-based design, and modern versions incorporate anchor boxes (or anchor-free variants), multi-scale feature maps, and more sophisticated backbones.

## 4. What is IoU and how do you calculate it?

**Intersection over Union (IoU)** is a metric that measures how well two bounding boxes (or masks) overlap — typically a predicted box compared to a ground-truth box. It's used to determine whether a detection counts as "correct" and to guide non-max suppression.

**Formula:**

IoU = Area of Overlap / Area of Union

Where:

- **Area of Overlap** = the area where the predicted box and ground-truth box intersect.
- **Area of Union** = the total area covered by both boxes combined (overlap counted only once).

IoU ranges from **0** (no overlap at all) to **1** (perfect overlap, identical boxes). In practice, a detection is often considered a "true positive" if its IoU with a ground-truth box exceeds a chosen threshold (commonly 0.5).

## 5. What is non-max suppression?

**Non-Max Suppression (NMS)** is a post-processing technique used to eliminate duplicate, overlapping detections of the same object, keeping only the single best (highest-confidence) box.

Detectors typically output many candidate boxes for the same object (since multiple grid cells or anchors may all detect the same object with slightly different boxes). NMS cleans this up as follows:

1. Sort all candidate boxes by their confidence score, highest first.
2. Select the box with the highest confidence and add it to the final output list.
3. Compare this box's IoU against all remaining boxes; **discard** any box whose IoU with the selected box exceeds a chosen threshold (meaning it's likely detecting the same object).
4. Repeat steps 2–3 with the remaining boxes until none are left.

The result is a clean set of detections with one box per detected object, rather than a cluttered pile of overlapping duplicates.

## 6. What are anchor boxes?

**Anchor boxes** are a set of predefined bounding box shapes (widths and heights) placed at each location on a detector's output grid, used as reference templates for predicting objects of various sizes and aspect ratios.

Instead of predicting a bounding box's coordinates entirely from scratch, the network predicts **offsets/adjustments relative to these anchor boxes** — e.g., "this anchor should be shifted this much and scaled by this factor to tightly fit the object." This makes learning easier because the network only needs to learn small refinements rather than absolute coordinates.

Anchor boxes are usually chosen based on common object shapes in the training dataset (e.g., via clustering of ground-truth box dimensions, as YOLO does with k-means), and multiple anchors of different aspect ratios/sizes are typically placed at each grid location so the model can detect both small, tall, wide, and large objects effectively. Some modern detectors are **anchor-free**, predicting box parameters directly without this template mechanism.

## 7. What is mAP and how do you calculate it?

**Mean Average Precision (mAP)** is the standard metric for evaluating object detection models, summarizing both localization accuracy (via IoU) and classification accuracy (via precision/recall) into a single number.

**Calculation, step by step:**

1. For each class, and for a given IoU threshold (e.g., 0.5), sort all predicted detections by confidence score.
2. Match predictions to ground-truth boxes: a prediction is a **true positive** if its IoU with a ground-truth box of the same class exceeds the threshold and that ground-truth hasn't already been matched; otherwise it's a **false positive**. Unmatched ground-truth boxes are **false negatives**.
3. Compute **precision** (true positives / all predictions) and **recall** (true positives / all ground truths) at each confidence threshold, producing a **precision-recall curve** for that class.
4. Compute the **Average Precision (AP)** for that class as the area under the precision-recall curve (often approximated by averaging precision at several recall levels).
5. Compute **mAP** by averaging the AP values across all classes.

Common variants:

- **mAP@0.5** — AP computed using a fixed IoU threshold of 0.5.
- **mAP@[0.5:0.95]** — AP averaged across multiple IoU thresholds (0.5 to 0.95 in steps of 0.05), a stricter, more comprehensive metric popularized by the COCO benchmark.

## 8. What is the difference between object detection and image segmentation?

- **Object detection** locates objects with coarse **bounding boxes** and assigns each a class label. It answers "what objects are here, and roughly where?" but doesn't describe the object's exact shape — the box may include background pixels around the object.
- **Image segmentation** classifies images at the **pixel level**, producing a precise outline (mask) of each object's exact shape, rather than a rectangular approximation. It answers "which exact pixels belong to which object/class?"

In short: detection gives you a rough rectangular location, segmentation gives you a pixel-accurate shape. Segmentation is more computationally expensive and provides finer-grained information, useful when precise shape matters (e.g., medical imaging, autonomous driving lane boundaries).

## 9. What are semantic and instance segmentation?

- **Semantic segmentation** assigns a class label to every pixel in the image, but does **not distinguish between separate instances** of the same class. For example, if there are three cars in an image, semantic segmentation labels all car pixels simply as "car" — it can't tell you there are three distinct cars, just that those pixels are car-pixels collectively.
    
- **Instance segmentation** goes further: it not only labels pixels by class, but also **distinguishes individual object instances**. Each of the three cars would get its own separate mask, so you know there are three distinct car objects and exactly which pixels belong to each one.
    

A useful mental model: semantic segmentation answers "what is at this pixel?", while instance segmentation answers "what is at this pixel, and which specific object does it belong to?" (There's also **panoptic segmentation**, which combines both — labeling every pixel by class _and_ distinguishing instances, even for "stuff" classes like sky or road that don't have countable instances.)

## 10. What is a segmentation mask?

A **segmentation mask** is a pixel-level map — typically the same height and width as the input image — where each pixel is labeled to indicate which class (semantic segmentation) or which specific object instance (instance segmentation) it belongs to.

Practically, a mask is often represented as:

- A **binary mask** for a single object: 1 for pixels belonging to that object, 0 for everything else.
- A **multi-class label map** for semantic segmentation: each pixel holds an integer representing its class ID.
- A **stack of binary masks** for instance segmentation: one binary mask per detected object instance.

Masks are the ground truth (and prediction output) for segmentation tasks, analogous to how bounding boxes are the ground truth/output for detection tasks.

## 11. How are objects represented in instance segmentation?

In instance segmentation, each detected object is typically represented by a combination of:

- A **bounding box** (roughly localizing the object, often produced as an intermediate step, e.g., in Mask R-CNN's region proposal stage).
- A **class label** (what the object is).
- A **confidence score** (how certain the model is).
- A **binary segmentation mask**, cropped to (or aligned with) the bounding box, indicating the precise pixels within that region that belong to the object.

So rather than a single flat label map (as in semantic segmentation), instance segmentation produces a **list of individual object instances**, each with its own box, class, score, and mask — allowing overlapping objects of the same class to be represented and distinguished separately.

## 12. What is polygon-based annotation?

**Polygon-based annotation** is a labeling approach where an object's boundary is marked by a series of connected points (vertices) forming a polygon that tightly traces the object's outline — as opposed to a simple axis-aligned rectangle (bounding box).

Compared to bounding boxes:

- Polygons capture the **actual shape** of an object (including concave regions, irregular edges, etc.), which is essential for training segmentation models.
- Bounding boxes are faster to annotate but include extra background area within the rectangle, especially for irregularly shaped or diagonally oriented objects.

Polygon annotations are commonly converted into segmentation masks (by filling the interior of the polygon) to serve as ground truth for training semantic or instance segmentation models.

## 13. What is mask IoU and how is it calculated?

**Mask IoU** is the segmentation equivalent of bounding box IoU — it measures the overlap between a predicted segmentation mask and a ground-truth mask, but computed at the **pixel level** rather than using rectangular box coordinates.

**Formula:**

Mask IoU = (Number of pixels where predicted mask and ground-truth mask overlap) / (Number of pixels in the union of predicted mask and ground-truth mask)

Concretely: count the pixels that are "on" (part of the object) in _both_ masks (intersection), and divide by the count of pixels that are "on" in _either_ mask (union). Because it accounts for the actual pixel-level shape rather than a rectangular approximation, mask IoU gives a much stricter and more precise measure of how well a segmentation prediction matches the true object shape than box IoU does.

## 14. Bounding box mAP vs Segmentation mAP?

Both metrics follow the same underlying mAP calculation process (precision-recall curves averaged across classes and IoU thresholds), but they differ in **how a "match" between prediction and ground truth is determined**:

- **Bounding box mAP** uses **box IoU** — a predicted detection is a true positive if its bounding box overlaps the ground-truth box above the IoU threshold.
- **Segmentation mAP** uses **mask IoU** — a predicted detection is a true positive only if its pixel-level segmentation mask overlaps the ground-truth mask above the IoU threshold.

Because mask IoU is a stricter, pixel-accurate measure, segmentation mAP is typically **harder to achieve high scores on** than box mAP for the same model — a model might get a good bounding box roughly around an object but still produce an imprecise mask (e.g., missing thin parts, fuzzy edges), lowering segmentation mAP while box mAP stays high. Models like Mask R-CNN report both metrics separately, since they perform both detection and segmentation.

## 15. How is non-max suppression applied when masks overlap?

NMS in instance segmentation works the same way conceptually as in detection, but the overlap criterion is typically based on **mask IoU** (or a combination of box IoU followed by mask IoU) instead of purely box IoU:

1. Sort candidate object instances by confidence score, highest first.
2. Select the highest-confidence instance and keep it in the final output.
3. Compare its mask (or box, depending on implementation) against remaining candidates using **mask IoU**; suppress (discard) any candidate whose mask IoU with the kept instance exceeds a threshold, since it likely represents the same physical object.
4. Repeat until all candidates are processed.

Using mask-based overlap (rather than just box overlap) is important for instance segmentation because two genuinely distinct, non-overlapping objects can have overlapping bounding boxes (e.g., two people standing close together) — mask IoU correctly recognizes that their actual pixel masks don't overlap much, preventing one of them from being wrongly suppressed as a "duplicate."

## 16. When to use object detection vs segmentation?

The right choice depends on how much spatial precision the downstream task actually needs, weighed against computational cost and annotation effort:

**Use object detection when:**

- You only need to know **what objects are present and roughly where** (e.g., counting objects, triggering an alert when something appears in a region, coarse localization for tracking).
- Speed/real-time performance is critical, and bounding boxes are "good enough" (detection is generally faster and cheaper to train/annotate than segmentation).
- Objects are mostly compact, and a rectangular approximation doesn't lose much useful information (e.g., detecting faces, vehicles for counting).

**Use segmentation when:**

- You need the **precise shape or boundary** of objects (e.g., measuring the exact area of a tumor in a medical scan, determining drivable road surface in autonomous driving, separating foreground objects from background for image editing).
- Objects have irregular shapes, overlap significantly, or a rectangular box would include too much irrelevant background to be useful.
- Downstream tasks require pixel-level reasoning (e.g., precise robotic grasping, augmented reality object replacement).

In practice, many pipelines start with detection (cheaper, faster) and only add segmentation where the extra pixel-level precision materially improves the end application, given the higher annotation and computational cost segmentation requires.