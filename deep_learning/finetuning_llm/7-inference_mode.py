#!/usr/bin/env python3
"""initializes a text classifcation pipeline"""
import transformers


def inference_mode(model_path, top_k):
    """loads the model and tokenizer and returns the model and tokenizer"""

    model = (transformers.AutoModelForSequenceClassification.
             from_pretrained(model_path))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    pipeline = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=top_k
    )

    return pipeline
