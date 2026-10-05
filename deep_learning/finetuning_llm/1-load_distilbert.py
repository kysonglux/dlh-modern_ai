#!/usr/bin/env python3
"""load tokenizer and sequence classification model"""
import transformers


def load_distilbert(model_name, num_classes, id_to_label, label_to_id):
    """
    Load the DistilBERT tokenizer and sequence classification model.
    """

    # Load the tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)

    # Load the sequence classification model
    model = transformers.DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes,
        id2label=id_to_label, label2id=label_to_id
    )

    return tokenizer, model
