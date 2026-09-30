#!/usr/bin/env python3
"""generates a textual description of a given image
using a pre-trained BLIP model and a pre-trained language model (LLM)"""
import transformers
from PIL import Image


def image_captioner(model, image_path, max_new_tokens):
    """generates a textual description of a given image
    using a pre-trained BLIP model and a pre-trained language model (LLM)."""

    processor = transformers.BlipProcessor.from_pretrained(model)
    blip_model = (
                transformers.BlipForConditionalGeneration.
                from_pretrained(model))

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
