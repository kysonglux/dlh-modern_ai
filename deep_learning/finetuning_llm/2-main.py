#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert
tokenize_and_map = __import__('2-tokenize_and_map').tokenize_and_map

from transformers import logging
logging.set_verbosity_error()

dataset = load_emotion_dataset()

model_name = "distilbert-base-uncased"
num_classes = 6
id_to_label = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
label_to_id = {v: k for k, v in id_to_label.items()}

tokenizer, _ = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

tokenized_train, tokenized_val, tokenized_test = tokenize_and_map(
    dataset,
    tokenizer,
    max_length=13,
    truncation=True,
    batched=True
)

# Inspect first examples
print("First tokenized train example:", tokenized_train[0])
print("First tokenized validation example:", tokenized_val[0])
print("First tokenized test example:", tokenized_test[0])