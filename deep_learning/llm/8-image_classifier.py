#!/usr/bin/env python3
"""creates a high-level interface to perform image calssification
using a pre-trained llm adapted for computer-vision applications"""
import transformers


def image_classifier(model):
    """creates a high-level interface to perform image classification
    """

    classifier = transformers.pipeline(
        "image-classification",
        model=model,
    )
    return classifier
